"""
Data augmentation (TRAIN set only, every example marked is_synthetic = 1).

People hide or change profane words in many ways. Real examples are rare in our data,
so we create copies of training reviews with the same changes:

  mask      : one letter -> '*'                      қотақбас -> қ*тақбас
  symbols   : letters -> look-alike symbols/digits    қотақ   -> қ0т@қ,  сігейін -> с&гейін
  latin     : whole review written in Latin           Мына ойын қотақ -> Myna oiyn qotaq
  no_kazakh : Kazakh letters typed as Russian ones    қотақ   -> котак (no Kazakh keyboard)

Each change is applied to the SAME number of profane and clean reviews, so the model
cannot learn "a symbol / Latin text means profane". Labels never change: a masked
swear word is still a swear word, and a masked clean word is still clean.
"""
import random
import re

from lexicon import match_terms, words

CYR_TO_LAT = {"а": "a", "ә": "a", "б": "b", "в": "v", "г": "g", "ғ": "g", "д": "d", "е": "e", "ё": "yo",
              "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "қ": "q", "л": "l", "м": "m", "н": "n",
              "ң": "n", "о": "o", "ө": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ұ": "u",
              "ү": "u", "ф": "f", "х": "h", "һ": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh",
              "ъ": "", "ы": "y", "і": "i", "ь": "", "э": "e", "ю": "yu", "я": "ya"}
KAZ_TO_RUS = str.maketrans("әғқңөұүһіӘҒҚҢӨҰҮҺІ", "агкноуухиАГКНОУУХИ")
SYMBOLS = {"о": ["0"], "а": ["@", "4"], "с": ["$"], "з": ["3"], "і": ["1", "!"], "и": ["&", "1"],
           "е": ["3", "&"], "т": ["7"], "қ": ["&", "%"], "к": ["&"], "ы": ["#"]}
VOWELS = set("аәеиоөуұүыіэюя")


def _target_word(text, label, rng):
    """For profane reviews: the swear word. For clean reviews: a random word (len >= 4)."""
    if label == 1:
        strong, _ = match_terms(text)
        cands = [w for _, w in strong if len(w) >= 3]
    else:
        cands = [w for w in words(text) if len(w) >= 4]
    return rng.choice(cands) if cands else None


def _replace_word(text, word, new):
    out = re.sub(re.escape(word), new, text, count=1, flags=re.I)
    return out if out != text else None


def aug_mask(text, label, rng):
    w = _target_word(text, label, rng)
    if not w:
        return None
    idx = [i for i in range(1, len(w) - 1) if w[i] in VOWELS] or [1]
    i = rng.choice(idx)
    return _replace_word(text, w, w[:i] + "*" * rng.choice([1, 1, 2]) + w[i + 1:])


def aug_symbols(text, label, rng):
    w = _target_word(text, label, rng)
    if not w:
        return None
    pos = [i for i, ch in enumerate(w) if ch in SYMBOLS and i > 0]
    if not pos:
        return None
    new = list(w)
    for i in rng.sample(pos, min(len(pos), rng.choice([1, 2]))):
        new[i] = rng.choice(SYMBOLS[new[i]])
    return _replace_word(text, w, "".join(new))


def aug_latin(text, label, rng):
    out = "".join(CYR_TO_LAT.get(ch.lower(), ch) if ch.isalpha() else ch for ch in text)
    return out if out != text else None


def aug_no_kazakh(text, label, rng):
    out = text.translate(KAZ_TO_RUS)
    return out if out != text else None


AUGMENTATIONS = {"mask": aug_mask, "symbols": aug_symbols, "latin": aug_latin, "no_kazakh": aug_no_kazakh}


def make_augmented(train, seed=42, per_type_per_class=150):
    """Return new synthetic rows: for each augmentation, `per_type_per_class` profane + the same
    number of clean reviews (sampled from TRAIN only)."""
    rng = random.Random(seed)
    out = []
    for name, fn in AUGMENTATIONS.items():
        for label in (1, 0):
            pool = [r for r in train if r["label"] == label and r["is_synthetic"] == 0]
            rng.shuffle(pool)
            made = 0
            for r in pool:
                if made >= per_type_per_class:
                    break
                new = fn(r["text"], label, rng)
                if new:
                    out.append({**r, "id": f"{r['id']}_{name}", "text": new,
                                "source": f"synthetic_{name}", "is_synthetic": 1})
                    made += 1
    return out
