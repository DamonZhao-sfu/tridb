"""Fetch a pinned Wikipedia + Wikidata slice for the GEM demo.

This is the ONLY module in the demo that touches the network, and it never runs
inside a scenario: it writes files and a manifest, and ``scenario.py`` reads
them. Wikipedia changes under you; a demo that re-fetches at run time produces
a different result every morning and cannot be debugged.

WHAT IT COLLECTS AND WHY EACH PIECE EXISTS
------------------------------------------
articles.jsonl   one row per article: title, pageid, QID, and the INTRO extract.
                 Intro rather than full text on purpose — a HotpotQA context
                 paragraph *is* an article's opening paragraph, so intro-grain
                 units line up with the gold evidence that grades act 2.
links.tsv        src_title -> dst_title hyperlinks, restricted to the slice.
                 These become ASSOCIATION edges: relatedness, no propagation.
classes.jsonl    every P31 / P106 / P279 target reachable from the slice, with
                 its label and its own P279 parents. These become CLASS UNITS,
                 and the membership/subsumption relations between them become
                 EXTENSION edges — the only kind `revise` may traverse.
revisions.jsonl  real Wikidata entity revision history for slice QIDs. The edit
                 comments are structured (`/* wbsetdescription-set:1|ca */ ...`)
                 and carry the new value, so they are genuine field-level
                 supersession evidence for units that actually exist in the
                 slice. This is what makes act 3 real rather than staged.

THE EXPANSION RULE
------------------
The frontier is not "the first N links" — MediaWiki returns links in
alphabetical order, so that silently biases a slice toward titles starting with
"A". Instead each round scores every unfetched link target by HOW MANY fetched
articles link to it and takes the top ones (ties broken by title, so the run is
deterministic). Co-citation keeps the slice topically coherent, which is what
makes the class hierarchy in it dense enough for C3 to be worth testing.

CACHING
-------
Every raw API response is written under ``cache/`` keyed by a hash of the URL.
A re-run with the same arguments is served entirely from cache and issues zero
requests, so the slice is reproducible and the fetch is resumable.

USAGE
-----
    python -m bench.agent_memory.demo.wiki_source \
        --out data/wiki_demo --articles 400 --revision-entities 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "tridb-gem-demo/0.1 (https://github.com/tridb; benchmark fetcher)"

#: Wikidata properties that make a unit a MEMBER of a class. A change to the
#: class entails re-evaluating its members, so these yield extension edges
#: oriented class -> member.
MEMBERSHIP_PROPERTIES = ("P31", "P106")

#: Subclass-of. Orientation parent -> child, same entailment argument.
SUBCLASS_PROPERTY = "P279"

#: Topic seeds: computing and the mathematics it grew out of. A single coherent
#: domain gives the P106/P279 hierarchy real depth, which is what makes the
#: extension graph — and therefore C3 — worth testing.
#:
#: These do NOT make the slice gradeable. Measured 2026-07-30: a 400-article
#: slice grown from these seeds fully resolved 0 of 1500 HotpotQA dev questions,
#: because HotpotQA is mostly films, bands and athletes. The gold evidence has
#: to come from the question set itself — see ``hotpot_link
#: --seeds-for-questions`` and the ``--seed-file`` flag, which is how the demo
#: slice is actually built.
DEFAULT_SEEDS = (
    "Alan Turing",
    "Claude Shannon",
    "John von Neumann",
    "Ada Lovelace",
    "Grace Hopper",
    "Donald Knuth",
    "Edsger W. Dijkstra",
    "Barbara Liskov",
    "Alonzo Church",
    "Kurt Gödel",
    "Turing machine",
    "Cryptanalysis of the Enigma",
    "ENIAC",
    "Lambda calculus",
    "Information theory",
    "Bletchley Park",
    "Von Neumann architecture",
    "Computer science",
    "Artificial intelligence",
    "Halting problem",
)

_API_PAUSE_SECONDS = 0.1


# ---------------------------------------------------------------------------
# HTTP with an on-disk cache
# ---------------------------------------------------------------------------


@dataclass
class Fetcher:
    """GET JSON, caching every response by URL hash.

    ``requests_made`` and ``cache_hits`` go into the manifest: a run that was
    entirely cache-served and a run that hit the live API are different
    provenance, and the reader should not have to guess which happened.
    """

    cache_dir: Path
    retries: int = 4
    timeout: int = 30
    requests_made: int = 0
    cache_hits: int = 0

    def get(self, url: str) -> dict[str, Any]:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))

        payload = self._get_live(url)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _get_live(self, url: str) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": USER_AGENT}
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                self.requests_made += 1
                time.sleep(_API_PAUSE_SECONDS)
                return body
            except Exception as exc:  # noqa: BLE001 — re-raised after retries
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"GET failed after {self.retries} tries: {url}\n  {last}")


def _url(endpoint: str, params: dict[str, Any]) -> str:
    return f"{endpoint}?{urllib.parse.urlencode(params)}"


# ---------------------------------------------------------------------------
# Title normalisation — must match tools/wiki_extract so HotpotQA gold keys in
# ---------------------------------------------------------------------------


def normalize_title(title: str) -> str:
    """MediaWiki-faithful title key, identical to ``tools.wiki_extract``.

    enwiki sets ``$wgCapitalLinks=true``, so only the first character is
    case-insensitive. Reusing the exact rule is what lets ``hotpot_link``
    resolve a HotpotQA gold title onto the same key this fetcher wrote.
    """
    collapsed = " ".join(title.replace("_", " ").split())
    if not collapsed:
        return ""
    return collapsed[0].upper() + collapsed[1:]


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------


@dataclass
class Article:
    title: str
    pageid: int
    qid: str | None
    extract: str
    links: tuple[str, ...] = ()

    def to_row(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "pageid": self.pageid,
            "qid": self.qid,
            "extract": self.extract,
        }


def _chunks(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def fetch_articles(
    fetcher: Fetcher,
    titles: Sequence[str],
    *,
    batch: int = 20,
    max_link_pages: int = 12,
) -> list[Article]:
    """Intro extract + ns0 links + QID for each title, batched and paginated.

    Two MediaWiki behaviours make the obvious version of this function wrong:

    ``exlimit=max`` is what permits more than one extract per request. Without
    it the API silently returns the extract for a single page and leaves the
    rest empty, which reads as missing articles rather than as a bad query.

    ``pllimit`` bounds the links returned by the WHOLE query, not per page, so
    a 20-title batch truncates after ~500 links and the survivors are the
    alphabetically-early ones. That is precisely the bias the co-citation
    expansion rule exists to avoid, so the ``continue`` cursor is followed and
    each page's link list is accumulated across responses. ``max_link_pages``
    bounds that walk; a hit is recorded in the manifest rather than silently
    truncating (some hub articles have thousands of links).
    """
    out: list[Article] = []
    for group in _chunks(list(titles), batch):
        params: dict[str, Any] = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "prop": "extracts|links|pageprops",
            "explaintext": 1,
            "exintro": 1,
            "exlimit": "max",
            "pllimit": "max",
            "plnamespace": 0,
            "redirects": 1,
            "titles": "|".join(group),
        }
        merged: dict[int, dict[str, Any]] = {}
        for _ in range(max_link_pages):
            payload = fetcher.get(_url(WIKIPEDIA_API, params))
            for page in payload.get("query", {}).get("pages", []):
                if page.get("missing"):
                    continue
                slot = merged.setdefault(
                    int(page["pageid"]),
                    {
                        "title": page["title"],
                        "pageid": int(page["pageid"]),
                        "qid": (page.get("pageprops") or {}).get("wikibase_item"),
                        "extract": "",
                        "links": [],
                    },
                )
                # The extract arrives on the first response only; later
                # continuation pages carry links alone.
                if page.get("extract") and not slot["extract"]:
                    slot["extract"] = page["extract"].strip()
                slot["links"].extend(
                    normalize_title(link["title"]) for link in page.get("links", [])
                )
            cursor = payload.get("continue")
            if not cursor:
                break
            params = {**params, **cursor}

        for slot in merged.values():
            if not slot["extract"]:
                continue
            out.append(
                Article(
                    title=normalize_title(slot["title"]),
                    pageid=slot["pageid"],
                    qid=slot["qid"],
                    extract=slot["extract"],
                    links=tuple(dict.fromkeys(link for link in slot["links"] if link)),
                )
            )
    return out


def expand(
    fetcher: Fetcher,
    seeds: Sequence[str],
    *,
    target: int,
    batch: int = 20,
    log: Any = print,
) -> dict[str, Article]:
    """Co-citation BFS until ``target`` articles are collected or nothing grows.

    The frontier is the most-linked-to unfetched titles, NOT the alphabetically
    first ones (see module docstring). Ties break on title so two runs with the
    same arguments produce the same slice.
    """
    collected: dict[str, Article] = {}
    # Every title ever REQUESTED, which is not the same as every title
    # collected: a link target can redirect onto an article already held, or
    # come back with no lead extract at all. Such a title never enters
    # ``collected``, so filtering the frontier on ``collected`` alone re-proposes
    # it every round and the walk spins forever without growing. Ask for nothing
    # twice.
    attempted: set[str] = set()
    frontier = [normalize_title(seed) for seed in seeds]

    while frontier and len(collected) < target:
        wanted = [t for t in frontier if t not in attempted][
            : max(0, target - len(collected))
        ]
        if not wanted:
            break
        attempted.update(wanted)
        before = len(collected)
        for article in fetch_articles(fetcher, wanted, batch=batch):
            collected.setdefault(article.title, article)
            attempted.add(article.title)  # redirects land under the real title
        gained = len(collected) - before
        log(
            f"  articles: {len(collected)}/{target} (+{gained} from {len(wanted)} asked)"
        )

        counts: Counter[str] = Counter()
        for article in collected.values():
            for link in article.links:
                if link and link not in collected and link not in attempted:
                    counts[link] += 1
        if not counts:
            break
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        frontier = [title for title, _ in ranked[: max(batch, 40)]]

    return collected


# ---------------------------------------------------------------------------
# Wikidata: classes (the extension graph) and revisions (the C1/C4 evidence)
# ---------------------------------------------------------------------------


@dataclass
class ClassNode:
    qid: str
    label: str
    parents: tuple[str, ...] = ()  # P279 targets

    def to_row(self) -> dict[str, Any]:
        return {"qid": self.qid, "label": self.label, "parents": list(self.parents)}


@dataclass
class EntityFacts:
    qid: str
    label: str
    description: str
    classes: tuple[str, ...] = ()  # P31 + P106 targets


def _claim_targets(entity: dict[str, Any], prop: str) -> tuple[str, ...]:
    out: list[str] = []
    for claim in entity.get("claims", {}).get(prop, []):
        snak = claim.get("mainsnak", {})
        value = (snak.get("datavalue") or {}).get("value")
        if isinstance(value, dict) and "id" in value:
            out.append(str(value["id"]))
    return tuple(out)


def fetch_entities(
    fetcher: Fetcher, qids: Sequence[str], *, batch: int = 50
) -> dict[str, EntityFacts]:
    """Label, English description and class memberships for each QID."""
    out: dict[str, EntityFacts] = {}
    for group in _chunks(list(qids), batch):
        payload = fetcher.get(
            _url(
                WIKIDATA_API,
                {
                    "action": "wbgetentities",
                    "format": "json",
                    "ids": "|".join(group),
                    "props": "labels|descriptions|claims",
                    "languages": "en",
                },
            )
        )
        for qid, entity in (payload.get("entities") or {}).items():
            if "missing" in entity:
                continue
            classes: list[str] = []
            for prop in MEMBERSHIP_PROPERTIES:
                classes.extend(_claim_targets(entity, prop))
            out[qid] = EntityFacts(
                qid=qid,
                label=((entity.get("labels") or {}).get("en") or {}).get("value", ""),
                description=(
                    ((entity.get("descriptions") or {}).get("en") or {}).get(
                        "value", ""
                    )
                ),
                classes=tuple(dict.fromkeys(classes)),
            )
    return out


def fetch_class_closure(
    fetcher: Fetcher, seed_classes: Sequence[str], *, levels: int = 2
) -> dict[str, ClassNode]:
    """Walk P279 upward ``levels`` times, collecting labels and parents.

    Two levels is enough for the demo's purpose: it puts a real subsumption
    chain (cryptographer -> scientist -> ...) in the slice without dragging in
    the whole Wikidata ontology, which would make the C3 frontier meaningless.
    """
    known: dict[str, ClassNode] = {}
    frontier = list(dict.fromkeys(seed_classes))

    for _ in range(max(1, levels)):
        pending = [qid for qid in frontier if qid not in known]
        if not pending:
            break
        next_frontier: list[str] = []
        for group in _chunks(pending, 50):
            payload = fetcher.get(
                _url(
                    WIKIDATA_API,
                    {
                        "action": "wbgetentities",
                        "format": "json",
                        "ids": "|".join(group),
                        "props": "labels|claims",
                        "languages": "en",
                    },
                )
            )
            for qid, entity in (payload.get("entities") or {}).items():
                if "missing" in entity:
                    continue
                parents = _claim_targets(entity, SUBCLASS_PROPERTY)
                known[qid] = ClassNode(
                    qid=qid,
                    label=(
                        ((entity.get("labels") or {}).get("en") or {}).get("value", "")
                    ),
                    parents=parents,
                )
                next_frontier.extend(parents)
        frontier = next_frontier

    return known


def fetch_revisions(
    fetcher: Fetcher, qids: Sequence[str], *, limit: int = 30
) -> list[dict[str, Any]]:
    """Wikidata entity revision history — one request per entity.

    ``rvlimit`` above 1 is only valid for a single page, so these cannot be
    batched. That is why the demo bounds how many entities get histories:
    the cost is one request each, and the value saturates quickly.
    """
    rows: list[dict[str, Any]] = []
    for qid in qids:
        payload = fetcher.get(
            _url(
                WIKIDATA_API,
                {
                    "action": "query",
                    "format": "json",
                    "formatversion": 2,
                    "prop": "revisions",
                    "titles": qid,
                    "rvlimit": limit,
                    "rvprop": "ids|timestamp|user|comment",
                },
            )
        )
        for page in payload.get("query", {}).get("pages", []):
            for revision in page.get("revisions", []):
                rows.append(
                    {
                        "qid": qid,
                        "revid": int(revision["revid"]),
                        "timestamp": revision["timestamp"],
                        "user": revision.get("user", ""),
                        "comment": revision.get("comment", ""),
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass
class SliceParams:
    out: Path
    articles: int = 400
    revision_entities: int = 60
    revision_class_entities: int = 20
    revision_limit: int = 30
    class_levels: int = 2
    seeds: tuple[str, ...] = DEFAULT_SEEDS
    batch: int = 20
    extra: dict[str, Any] = field(default_factory=dict)


def build(params: SliceParams, *, log: Any = print) -> dict[str, Any]:
    """Fetch the slice and write it, returning the manifest."""
    cache = params.out / "cache"
    fetcher = Fetcher(cache_dir=cache)

    log(f"[1/5] expanding from {len(params.seeds)} seeds to {params.articles} articles")
    articles = expand(
        fetcher, params.seeds, target=params.articles, batch=params.batch, log=log
    )

    # Hyperlinks, restricted to the slice — a link to an article we did not
    # fetch has no unit to point at, so it is dropped rather than dangling.
    in_slice = set(articles)
    link_rows = [
        (article.title, target)
        for article in articles.values()
        for target in article.links
        if target in in_slice and target != article.title
    ]

    qids = [a.qid for a in articles.values() if a.qid]
    log(f"[2/5] entity facts for {len(qids)} QIDs")
    entities = fetch_entities(fetcher, qids)

    seed_classes = sorted({c for facts in entities.values() for c in facts.classes})
    log(
        f"[3/5] class closure over {len(seed_classes)} classes, {params.class_levels} levels"
    )
    classes = fetch_class_closure(fetcher, seed_classes, levels=params.class_levels)

    # Revision histories for the most-connected ARTICLE entities supply the
    # association-only negative control for C3 and the bulk of C1/C4's real
    # supersessions.
    degree: Counter[str] = Counter()
    for src, dst in link_rows:
        degree[src] += 1
        degree[dst] += 1
    ranked_titles = sorted(articles, key=lambda t: (-degree[t], t))
    article_revision_qids = [
        articles[title].qid for title in ranked_titles if articles[title].qid
    ][: params.revision_entities]

    # Extension edges are oriented class -> member because `revise` walks
    # out-edges. Fetching histories only for articles therefore gives every
    # changed unit an empty extension frontier and makes the positive half of
    # C3 impossible to falsify. Include the highest-frontier CLASS entities as
    # real revision sources too. This was discovered at G2 when the original
    # 60 article-only histories first met the live graph.
    extension_degree: Counter[str] = Counter()
    for facts in entities.values():
        for qid in facts.classes:
            extension_degree[qid] += 1
    for node in classes.values():
        for parent in node.parents:
            extension_degree[parent] += 1
    class_revision_qids = [
        qid
        for qid, _count in sorted(
            extension_degree.items(), key=lambda item: (-item[1], item[0])
        )
        if qid in classes
    ][: params.revision_class_entities]

    revision_qids = list(dict.fromkeys([*article_revision_qids, *class_revision_qids]))
    log(
        "[4/5] revision history for "
        f"{len(article_revision_qids)} articles + "
        f"{len(class_revision_qids)} extension classes"
    )
    revisions = fetch_revisions(fetcher, revision_qids, limit=params.revision_limit)

    log("[5/5] writing")
    params.out.mkdir(parents=True, exist_ok=True)
    article_rows: list[dict[str, Any]] = []
    for title in sorted(articles):
        article = articles[title]
        facts = entities.get(article.qid or "")
        article_rows.append(
            {
                **article.to_row(),
                # Empty when the article has no QID or Wikidata has no English
                # label — recorded as empty rather than dropped, so the demo's
                # coverage numbers stay honest.
                "label": facts.label if facts else "",
                "description": facts.description if facts else "",
                "classes": list(facts.classes) if facts else [],
            }
        )
    _write_jsonl(params.out / "articles.jsonl", article_rows)
    _write_jsonl(
        params.out / "classes.jsonl", [classes[q].to_row() for q in sorted(classes)]
    )
    _write_tsv(params.out / "links.tsv", sorted(set(link_rows)))
    _write_jsonl(
        params.out / "revisions.jsonl",
        sorted(revisions, key=lambda r: (r["qid"], r["revid"])),
    )

    manifest = {
        "generator": "bench.agent_memory.demo.wiki_source",
        "generator_version": "0.1.0",
        "endpoints": {"wikipedia": WIKIPEDIA_API, "wikidata": WIKIDATA_API},
        "params": {
            "articles_requested": params.articles,
            "revision_entities": params.revision_entities,
            "revision_class_entities": params.revision_class_entities,
            "revision_limit": params.revision_limit,
            "class_levels": params.class_levels,
            "seeds": list(params.seeds),
            "membership_properties": list(MEMBERSHIP_PROPERTIES),
            "subclass_property": SUBCLASS_PROPERTY,
            "expansion_rule": "co-citation: top unfetched link targets by in-slice link count",
            "extract": "exintro (article lead), plaintext",
        },
        "counts": {
            "articles": len(articles),
            "articles_with_qid": len(qids),
            "links_in_slice": len(set(link_rows)),
            "classes": len(classes),
            "revisions": len(revisions),
            "revision_entities": len(revision_qids),
            "revision_article_entities": len(article_revision_qids),
            "revision_class_entities": len(class_revision_qids),
        },
        "fetch": {
            "requests_made": fetcher.requests_made,
            "cache_hits": fetcher.cache_hits,
            "cache_dir": str(cache),
        },
        "files": {
            "articles": "articles.jsonl",
            "classes": "classes.jsonl",
            "links": "links.tsv",
            "revisions": "revisions.jsonl",
        },
    }
    (params.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_tsv(path: Path, rows: Sequence[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for src, dst in rows:
            handle.write(f"{src}\t{dst}\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/wiki_demo"))
    parser.add_argument("--articles", type=int, default=400)
    parser.add_argument("--revision-entities", type=int, default=60)
    parser.add_argument(
        "--revision-class-entities",
        type=int,
        default=20,
        help=(
            "also pin histories for N high-frontier class units so real "
            "revisions exercise the positive extension half of C3"
        ),
    )
    parser.add_argument("--revision-limit", type=int, default=30)
    parser.add_argument("--class-levels", type=int, default=2)
    parser.add_argument(
        "--seed",
        action="append",
        default=None,
        help="override the default seed list (repeatable)",
    )
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        help=(
            "file of seed titles, one per line, ADDED to --seed/the defaults. "
            "Feed it `hotpot_link --seeds-for-questions N` so the slice "
            "contains the gold evidence act 2 is graded on."
        ),
    )
    parser.add_argument(
        "--no-default-seeds",
        action="store_true",
        help="do not include DEFAULT_SEEDS (use only --seed / --seed-file)",
    )
    args = parser.parse_args(argv)

    seeds: list[str] = []
    if args.seed:
        seeds.extend(args.seed)
    elif not args.no_default_seeds and not args.seed_file:
        seeds.extend(DEFAULT_SEEDS)
    if args.seed_file:
        seeds.extend(
            line.strip()
            for line in args.seed_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not args.no_default_seeds and not args.seed:
            seeds.extend(DEFAULT_SEEDS)
    # De-duplicate while preserving order: a seed list assembled from a question
    # set repeats titles, and asking twice would only waste requests.
    seeds = list(dict.fromkeys(seeds))

    manifest = build(
        SliceParams(
            out=args.out,
            articles=args.articles,
            revision_entities=args.revision_entities,
            revision_class_entities=args.revision_class_entities,
            revision_limit=args.revision_limit,
            class_levels=args.class_levels,
            seeds=tuple(seeds),
        )
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(f"manifest -> {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
