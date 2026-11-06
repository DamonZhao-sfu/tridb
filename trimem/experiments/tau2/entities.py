"""τ²-bench task identity for L1 isolation: domain, family and shared entity ids.

Entities are derived only from what defines the task's world: the user scenario (reason for call,
known / unknown info, task instructions, persona) and the initial state. Ids are resolved against the
official domain DB so that a task naming an order or reservation is tied to its owner, and a task that
identifies the customer by name + zip or e-mail (retail) is tied to the user id:

* ``user:<user_id>``  -- explicit id, or every DB user whose full name / e-mail appears in the text
* ``order:#Wnnnnnnn`` -- retail order ids (+ owner user)
* ``reservation:XXXXXX`` -- airline reservation ids that exist in the DB (+ owner user)
* ``customer:/line:/phone:`` -- telecom ids from initialization actions and text

Telecom has ONE synthetic customer in every task (C1001 / L1002 / John Smith / 555-123-2002), so a
literal shared-user rule would remove all telecom memory. Entities present in every task of a domain
(and the lines owned by such a constant customer) are therefore *domain constants*: they carry no
task-specific record and are dropped unless ``strict=True``. Airline / retail have no constants. Telecom tasks that differ only by ``[PERSONA:...]`` form one family.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "vendor/tau2-bench/data/tau2/domains"
DOMAINS = ("airline", "retail", "telecom")
USER_ID = re.compile(r"\b[a-z]+_[a-z]+_\d{4}\b")
ORDER_ID = re.compile(r"#W\d{7}\b")
RESERVATION_ID = re.compile(r"\b[A-Z0-9]{6}\b")
PHONE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
PERSONA = re.compile(r"\[PERSONA:[^\]]*\]$")
TELECOM_KEYS = {"customer_id": "customer", "line_id": "line", "phone_number": "phone"}


def task_key(domain: str, task_id: str) -> str:
    return f"{domain}:{task_id}"


def family(domain: str, task_id: str) -> str:
    """Isolation family: the task itself; telecom tasks differing only by persona share one."""
    if domain == "telecom":
        return f"telecom:{PERSONA.sub('', task_id)}"
    return task_key(domain, task_id)


class DomainDB:
    """Read-only id resolver over one official domain DB."""

    def __init__(self, domain: str, data_dir: Path = DATA_DIR, db: dict | None = None):
        self.domain = domain
        if db is None:
            path = data_dir / domain / ("db.toml" if domain == "telecom" else "db.json")
            db = (
                tomllib.loads(path.read_text())
                if path.suffix == ".toml"
                else json.loads(path.read_text())
            )
        self.users: dict = db.get("users", {}) if domain != "telecom" else {}
        self.records: dict = db.get("orders") or db.get("reservations") or {}
        self.names: dict[str, set[str]] = defaultdict(set)
        self.emails: dict[str, str] = {}
        for uid, u in self.users.items():
            n = u.get("name") or {}
            if n.get("first_name") and n.get("last_name"):
                self.names[f"{n['first_name']} {n['last_name']}".lower()].add(uid)
            if u.get("email"):
                self.emails[u["email"].lower()] = uid
        self.phone_owner: dict[str, str] = {}
        self.customer_lines: dict[str, list[str]] = {}
        for c in db.get("customers", []) if domain == "telecom" else []:
            if c.get("phone_number"):
                self.phone_owner[c["phone_number"]] = c["customer_id"]
            self.customer_lines[c["customer_id"]] = list(c.get("line_ids") or [])

    def entities(self, text: str) -> set[str]:
        out: set[str] = set()
        low = text.lower()
        if self.domain == "telecom":
            for p in PHONE.findall(text):
                out.add(f"phone:{p}")
                if p in self.phone_owner:
                    out.add(f"customer:{self.phone_owner[p]}")
            return out
        for uid in USER_ID.findall(text):
            if uid in self.users:
                out.add(f"user:{uid}")
        for name, ids in self.names.items():
            if name in low and re.search(r"\b" + re.escape(name) + r"\b", low):
                out |= {f"user:{i}" for i in ids}
        for email, uid in self.emails.items():
            if email in low:
                out.add(f"user:{uid}")
        pattern, kind = (
            (ORDER_ID, "order")
            if self.domain == "retail"
            else (RESERVATION_ID, "reservation")
        )
        for rid in pattern.findall(text):
            if rid in self.records:
                out.add(f"{kind}:{rid}")
                owner = self.records[rid].get("user_id")
                if owner:
                    out.add(f"user:{owner}")
        return out


def scenario_text(task: dict) -> str:
    """User scenario + initial state only (never the evaluation criteria)."""
    return json.dumps(
        {
            "user_scenario": task.get("user_scenario"),
            "initial_state": task.get("initial_state"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def raw_entities(task: dict, db: DomainDB) -> set[str]:
    out = db.entities(scenario_text(task))
    init = (task.get("initial_state") or {}).get("initialization_actions") or []
    for a in init:
        for k, v in (a.get("arguments") or {}).items():
            if k in TELECOM_KEYS and isinstance(v, str):
                out.add(f"{TELECOM_KEYS[k]}:{v}")
    return out


@lru_cache(maxsize=8)
def _db(domain: str, data_dir: str) -> DomainDB:
    return DomainDB(domain, Path(data_dir))


@lru_cache(maxsize=8)
def domain_constants(domain: str, data_dir: str = str(DATA_DIR)) -> frozenset[str]:
    """Entities shared by every task of the domain's official task file, plus the lines owned by such a
    constant customer (telecom: the single synthetic customer's line L1002 is not task-specific)."""
    tasks = json.loads((Path(data_dir) / domain / "tasks.json").read_text())
    db = _db(domain, data_dir)
    sets = [raw_entities(t, db) for t in tasks]
    const = set.intersection(*sets) if sets else set()
    for e in list(const):
        if e.startswith("customer:"):
            const |= {
                f"line:{x}" for x in db.customer_lines.get(e.split(":", 1)[1], [])
            }
    return frozenset(const)


def task_entities(
    domain: str,
    task: dict,
    *,
    strict: bool = False,
    data_dir: str = str(DATA_DIR),
    db: DomainDB | None = None,
    constants: frozenset[str] | None = None,
) -> frozenset[str]:
    ents = raw_entities(task, db or _db(domain, data_dir))
    if not strict:
        ents -= (
            constants if constants is not None else domain_constants(domain, data_dir)
        )
    return frozenset(ents)


def identity(domain: str, task: dict, **kw) -> dict:
    """Everything the adapter needs to isolate a query task (task: official tasks.json dict)."""
    tid = str(task["id"])
    return {
        "key": task_key(domain, tid),
        "env": domain,
        "task_id": tid,
        "family": family(domain, tid),
        "entities": sorted(task_entities(domain, task, **kw)),
    }
