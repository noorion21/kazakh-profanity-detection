"""
Text normalization for Kazakh reviews written in different scripts.

Problem: the same word appears as  қотақ / котак / kotak / qotaq / qotaǵ ...
Idea   : bring every review to ONE script (Kazakh Cyrillic) before the model sees it,
         so the model does not have to learn each spelling separately.

Rules (applied in this order):
  1. Latin-script reviews -> Kazakh Cyrillic transliteration
     (informal Latin: sh, zh, ch, q, w=ш, y=ы  and the 2021 official alphabet: á ǵ ı ń ó ú ý ş ç ...)
     Only when the review is MOSTLY Latin, or for a Latin word that uses Kazakh-Latin letters.
     -> English brand names inside Cyrillic reviews ("Free Fire", "TikTok") are kept.
  2. Latin look-alike letters inside Cyrillic words -> Cyrillic  ("кoтак" with Latin o -> "котак")
  3. Symbols used as letters inside a word -> letters            ("к0тақ" -> "котақ", "с@ка" -> "сака")
     0->о, @->а, $->с, 3->з  (only between two letters, so "100%" or "@user" are not touched)
  4. A letter repeated 3+ times -> 2 times                       ("бляяяяя" -> "бляя")

We do NOT remove Kazakh letters (қ->к, ө->о ...): "көт" (ass) and "кот" (cat) would become the
same word. We also keep case, punctuation, emoji and '*' masks.
"""
import re

KAZ_LATIN_SPECIAL = set("áǵıńóúýşçğüöäñÁǴIŃÓÚÝŞÇĞÜÖÄÑ")

# longest patterns first
LATIN_TO_CYR = [
    ("shch", "щ"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("kh", "х"), ("ts", "ц"),
    ("ya", "я"), ("yu", "ю"), ("yo", "ё"),
    ("á", "ә"), ("ä", "ә"), ("ǵ", "ғ"), ("ğ", "ғ"), ("ı", "ы"), ("ń", "ң"), ("ñ", "ң"),
    ("ó", "ө"), ("ö", "ө"), ("ú", "ү"), ("ü", "ү"), ("ý", "у"), ("ş", "ш"), ("ç", "ч"),
    ("a", "а"), ("b", "б"), ("c", "с"), ("d", "д"), ("e", "е"), ("f", "ф"), ("g", "г"),
    ("h", "һ"), ("i", "и"), ("j", "ж"), ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"),
    ("o", "о"), ("p", "п"), ("q", "қ"), ("r", "р"), ("s", "с"), ("t", "т"), ("u", "у"),
    ("v", "в"), ("w", "ш"), ("x", "х"), ("y", "ы"), ("z", "з"),
]
_LAT_RX = re.compile("|".join(re.escape(a) for a, _ in LATIN_TO_CYR))
_LAT_MAP = dict(LATIN_TO_CYR)

HOMOGLYPHS = str.maketrans("aeopcxykmtABEKMHOPCTXY", "аеорсхукмтАВЕКМНОРСТХУ")

_LATIN_LETTER = re.compile(r"[a-zA-Záǵıńóúýşçğüöäñ]")
_CYR_LETTER = re.compile(r"[а-яА-ЯёЁәғқңөұүһіӘҒҚҢӨҰҮҺІ]")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_REPEAT = re.compile(r"(.)\1{2,}")
LEET = {"0": "о", "@": "а", "$": "с", "3": "з"}
_LEET_RX = re.compile(r"(?<=[^\W\d_])[0@$3](?=[^\W\d_])")


def _translit_word(w):
    low = w.lower()
    low = re.sub(r"(?<=[aeouáóú])[yi]", "й", low)    # "oyin"/"oiyn" -> "ойин"/"ойын"
    out = _LAT_RX.sub(lambda m: _LAT_MAP[m.group(0)], low)
    return out.capitalize() if w[:1].isupper() else out


def normalize_text(text, leet=True):
    text = " ".join(str(text).split())
    if leet:
        text = _LEET_RX.sub(lambda m: LEET[m.group(0)], text)
    mostly_latin = len(_LATIN_LETTER.findall(text)) > len(_CYR_LETTER.findall(text))

    def fix(m):
        w = m.group(0)
        has_lat = bool(_LATIN_LETTER.search(w))
        has_cyr = bool(_CYR_LETTER.search(w))
        if has_lat and has_cyr:                       # mixed word -> Latin look-alikes to Cyrillic
            return w.translate(HOMOGLYPHS)
        if has_lat and (mostly_latin or set(w) & KAZ_LATIN_SPECIAL):
            return _translit_word(w)
        return w

    text = _WORD.sub(fix, text)
    return _REPEAT.sub(r"\1\1", text)


if __name__ == "__main__":
    for t in ["Qotagimdi je", "kotakbas oyin", "shesheńdi mynaý ne aǵylshynsha tur",
              "Free Fire керемет ойын", "кoтак", "бляяяяя", "Mal eken birsk zor eken", "sygylynderw", "oiyn kyzyk", "Boq ekengoy", "к0тақ", "100% ұнады", "с@ка"]:
        print(f"{t!r:45} -> {normalize_text(t)!r}")
