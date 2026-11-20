"""
Step 1 – Build the labeled Kazakh profanity dataset from KazSAnDRA reviews.

    python src/prepare_data.py

Input : data/raw/01_pc_train_ib.zip, 04_pc_valid.zip, 05_pc_test.zip
        (KazSAnDRA, CC BY 4.0, https://github.com/IS2AI/KazSAnDRA)
        data/manual_labels.csv  (my hand-checked labels / corrections)
        data/peer/peer_labels.csv (classmate's labels for 3,061 KazSAnDRA reviews)
Output: data/processed/train.csv, val.csv, test.csv, train_augmented.csv,
        split_info.json, peer_agreement.json

What happens, in order:
  1. load all ~168k reviews and remove personal info (phones, e-mails, links, surnames)
  2. label every review: 1 = profanity (obscene words OR insults), 0 = clean
     (lexicon for candidates + manual review, see README "Labeling rules")
  3. remove exact duplicates, group near-duplicates
  4. keep ALL profane reviews + ALL classmate-labeled reviews + a sample of clean ones
  5. split by group into train / val / test (70 / 15 / 15, seed 42)
  6. create clearly-marked synthetic (augmented) copies of TRAIN reviews -> train_augmented.csv
     (used only when training with --augment 1, see augment.py)
"""
import csv
import io
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from augment import make_augmented  # noqa: E402
from lexicon import (has_insult, has_lookalike, match_insults, match_terms,  # noqa: E402
                     profanity_type, script_type, words)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SEED = 42
RAW_FILES = ["01_pc_train_ib.zip", "04_pc_valid.zip", "05_pc_test.zip"]
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
# how many clean reviews to keep for every profane review, by type
NEG_PER_POS = {"insult_word_literal": 1.0, "lookalike": 0.3, "neg_sentiment": 0.8, "random": 0.8}


# ---------------------------------------------------------------- 1. load + PII
def load_raw():
    rows = []
    for name in RAW_FILES:
        with zipfile.ZipFile(RAW / name) as z:
            with z.open(z.namelist()[0]) as f:
                rows += list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
    return rows


PII = [
    (re.compile(r"\S+@\S+\.\w+"), "<EMAIL>"),
    (re.compile(r"(https?://|www\.)\S+"), "<URL>"),
    (re.compile(r"\+?\d[\d\s\-()]{8,}\d"), "<NUMBER>"),          # phones, IINs, cards
    (re.compile(r"(?<!\w)@\w+"), "<USER>"),
]
# Kazakh/Russian surnames (Capitalised, typical endings) -> <NAME>
SURNAME = re.compile(r"\b[А-ЯӘҒҚҢӨҰҮҺІ][а-яәғқңөұүһі]{2,}(ова|ева|ов|ев|ұлы|улы|қызы|кызы)\b")


def _name(m):
    # never hide a profane word that happens to look like a surname (e.g. "Далбаев")
    return m.group(0) if match_terms(m.group(0))[0] else "<NAME>"


def scrub(text):
    text = " ".join(str(text).split())        # one line, single spaces
    for rx, tag in PII:
        text = rx.sub(tag, text)
    return SURNAME.sub(_name, text)


# ---------------------------------------------------------------- 2. labels
def load_manual():
    path = ROOT / "data" / "manual_labels.csv"
    with open(path, encoding="utf-8") as f:
        return {r["custom_id"]: int(r["label"]) for r in csv.DictReader(f)}


def load_peer():
    path = ROOT / "data" / "peer" / "peer_labels.csv"
    with open(path, encoding="utf-8") as f:
        return {r["kazsandra_id"]: int(r["peer_label"]) for r in csv.DictReader(f)}


KNOWN_PROFANE = set()   # filled in main(): every word the lexicon marks as obscene


def label_review(row, manual):
    """Return 1, 0, or None (= excluded: ambiguous / not reviewed)."""
    if row["custom_id"] in manual:                 # my hand label always wins
        lab = manual[row["custom_id"]]
        return None if lab == -1 else lab
    strong, review = match_terms(row["text"], KNOWN_PROFANE)   # also checks masked words
    if strong:                                     # obscene word
        return 1
    if review:                                     # every such case is in manual_labels
        return None
    ins_strong, ins_review = match_insults(row["text"])
    if ins_strong and row["domain"] == "appstore":  # insult in an app/game review
        return 1
    if ins_strong or ins_review:                   # possible literal meaning -> hand-checked
        return None
    return 0


# ---------------------------------------------------------------- 3. duplicates
def norm_key(text):
    """Lower-case letters only, repeated letters collapsed -> exact-duplicate key."""
    return "".join(words(re.sub(r"[\d_]", " ", text)))


def bag_key(text):
    """Same words in any order / with small edits -> near-duplicate key."""
    ws = sorted(set(w for w in words(text) if len(w) > 2 and not w.isdigit()))
    return " ".join(ws)


def group_near_duplicates(items):
    """Union-find: items sharing an exact key OR a bag-of-words key get one group id."""
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for key_fn in (norm_key, bag_key):
        first = {}
        for i, it in enumerate(items):
            k = key_fn(it["text"])
            if not k:
                continue
            if k in first:
                parent[find(i)] = find(first[k])
            else:
                first[k] = i
    for i, it in enumerate(items):
        it["group"] = find(i)


# ---------------------------------------------------------------- 7. second annotator
def peer_agreement(data):
    """Compare my labels with the classmate's labels on the reviews we both labeled."""
    pairs = [(int(i["peer_label"]), i["label"]) for i in data if i["peer_label"] != ""]
    n = len(pairs)
    agree = sum(a == b for a, b in pairs) / n
    p_peer1 = sum(a for a, _ in pairs) / n
    p_me1 = sum(b for _, b in pairs) / n
    p_chance = p_peer1 * p_me1 + (1 - p_peer1) * (1 - p_me1)
    kappa = (agree - p_chance) / (1 - p_chance)       # Cohen's kappa
    table = Counter(f"peer{a}_me{b}" for a, b in pairs)
    res = {"n_both_labeled": n, "percent_agreement": round(agree, 4),
           "cohen_kappa": round(kappa, 4), "table": dict(table)}
    print("agreement with classmate:", res)
    with open(OUT / "peer_agreement.json", "w") as f:
        json.dump(res, f, indent=2)


# ---------------------------------------------------------------- main
def main():
    rng = random.Random(SEED)
    manual = load_manual()
    peer = load_peer()
    raw = load_raw()
    print(f"loaded {len(raw)} KazSAnDRA reviews")
    for r in raw:                                  # vocabulary for un-masking "бл**", "с*ка"
        KNOWN_PROFANE.update(w for _, w in match_terms(r["text"])[0])

    items, excluded = [], 0
    for r in raw:
        lab = label_review(r, manual)
        if lab is None:
            excluded += 1
            continue
        items.append({
            "id": r["custom_id"], "text": scrub(r["text"]), "label": lab,
            "domain": r["domain"], "sentiment": "positive" if r["label"] == "1" else "negative",
            "profanity_type": profanity_type(r["text"], KNOWN_PROFANE) if lab == 1 else "none",
            "script": script_type(r["text"]),
            "peer_label": peer.get(r["custom_id"], ""),
        })
    print(f"excluded (ambiguous): {excluded}")

    # 3a. exact duplicates: keep the first copy
    seen, uniq = set(), []
    for it in items:
        k = norm_key(it["text"])
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append(it)
    print(f"after removing exact duplicates: {len(uniq)} "
          f"(profane: {sum(i['label'] for i in uniq)})")

    # 4. keep all profane + all classmate-labeled + sampled clean reviews
    pos = [i for i in uniq if i["label"] == 1]
    neg = [i for i in uniq if i["label"] == 0]
    for i in pos:
        i["source"] = "profane_insult" if i["profanity_type"] == "insult_only" else "profane_obscene"
    chosen = {}
    for i in neg:                                  # classmate's clean reviews are always kept
        if i["peer_label"] != "":
            i["source"] = "classmate_clean"
            chosen[i["id"]] = i
    pools = {
        "insult_word_literal": [i for i in neg if has_insult(i["text"])],   # мал базары, поп ит
        "lookalike": [i for i in neg if has_lookalike(i["text"]) and not has_insult(i["text"])],
        "neg_sentiment": [i for i in neg if i["sentiment"] == "negative"],
        "random": neg,
    }
    for name, ratio in NEG_PER_POS.items():
        pool = [i for i in pools[name] if i["id"] not in chosen]
        k = min(len(pool), round(ratio * len(pos)))
        for i in rng.sample(pool, k):
            i["source"] = name
            chosen[i["id"]] = i
    data = pos + list(chosen.values())

    # 3b. near-duplicate groups (inside the final data)
    group_near_duplicates(data)

    # 5. grouped, label-stratified split
    groups = defaultdict(list)
    for it in data:
        groups[it["group"]].append(it)
    split_of = {}
    for lab in (0, 1):
        gids = sorted(g for g, its in groups.items() if max(i["label"] for i in its) == lab)
        rng.shuffle(gids)
        total = sum(len(groups[g]) for g in gids)
        acc = 0
        for g in gids:
            frac = acc / total
            split_of[g] = "train" if frac < SPLIT["train"] else (
                "val" if frac < SPLIT["train"] + SPLIT["val"] else "test")
            acc += len(groups[g])
    splits = {s: [] for s in SPLIT}
    for it in data:
        it["is_synthetic"] = 0
        splits[split_of[it["group"]]].append(it)

    # 6. augmented copies of TRAIN reviews (separate file, train only)
    augmented = make_augmented(splits["train"], seed=SEED)

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["id", "text", "label", "profanity_type", "script", "domain", "sentiment", "source",
            "peer_label", "is_synthetic", "group"]
    info = {"seed": SEED, "source": "KazSAnDRA (CC BY 4.0) polarity files 01, 04, 05",
            "raw_reviews": len(raw), "excluded_ambiguous": excluded, "splits": {}}
    for name, rows in splits.items():
        rng.shuffle(rows)
        with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        c = Counter(r["label"] for r in rows)
        info["splits"][name] = {
            "total": len(rows), "profane_1": c[1], "clean_0": c[0],
            "synthetic": sum(r["is_synthetic"] for r in rows),
            "by_source": dict(Counter(r["source"] for r in rows)),
            "profane_by_type": dict(Counter(r["profanity_type"] for r in rows if r["label"] == 1)),
            "by_script": dict(Counter(r["script"] for r in rows)),
            "by_domain": dict(Counter(r["domain"] for r in rows)),
        }
        print(f"{name:5s}: {len(rows):5d} rows | profane {c[1]:4d} | clean {c[0]:5d}")
    with open(OUT / "train_augmented.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(augmented)
    info["train_augmented"] = {"total": len(augmented),
                               "by_source": dict(Counter(r["source"] for r in augmented)),
                               "profane_1": sum(r["label"] for r in augmented)}
    print(f"augmented (train only, synthetic): {len(augmented)} rows")
    # sanity check: no near-duplicate group is in two splits
    g_split = defaultdict(set)
    for name, rows in splits.items():
        for r in rows:
            g_split[r["group"]].add(name)
    assert all(len(s) == 1 for s in g_split.values()), "group leakage between splits!"
    with open(OUT / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    peer_agreement(data)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
