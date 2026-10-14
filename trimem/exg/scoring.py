"""Answer EM/F1, matching each dataset's pinned official normalization rules."""

from collections import Counter

from exg.collection.scoring import normalize


def score(answer, gold, dataset, valid=True):
    pred = normalize(answer)
    ems, f1s = [], []
    for target in [gold["answer"]] + gold.get("aliases", []):
        ref = normalize(target)
        em = int(pred == ref)
        ems.append(em)
        p, r = pred.split(), ref.split()
        common = sum((Counter(p) & Counter(r)).values())
        if dataset == "hotpotqa":
            special = {"yes", "no", "noanswer"}
            if not common or ((pred in special or ref in special) and not em):
                f1 = 0.0
            else:
                f1 = 2 * common / (len(p) + len(r))
        elif dataset == "musique":
            f1 = float(p == r) if not p or not r else 2 * common / (len(p) + len(r))
        else:
            raise ValueError(dataset)
        f1s.append(f1)
    return dict(
        em=max(ems),
        f1=max(f1s),
        passed=bool(valid and max(ems)),
        valid=valid,
        scorer=f"{dataset}-official-answer-rules-v1",
    )
