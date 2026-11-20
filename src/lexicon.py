"""
Profanity lexicon used ONLY to help label the data (weak labeling).

The trained model does NOT use this file at prediction time. We use it to:
  1. find candidate profane reviews in KazSAnDRA quickly,
  2. split matches into "strong" (clearly profane) and "review" (ambiguous,
     checked by hand -> data/manual_labels.csv),
  3. find "hard negatives" (look-alike words and rude-but-not-profane insults).

Matching works word by word on lowercased text. Repeated letters are
collapsed first ("бляяя" -> "бля", "боққққ" -> "боқ").
"""
import re

from normalize import normalize_text

# ---------------------------------------------------------------------------
# 1. STRONG: clearly obscene words (Kazakh + Russian mat used in Kazakh text)
#    Each pattern is matched at the START of a word (prefix match).
# ---------------------------------------------------------------------------
STRONG_PATTERNS = {
    # Kazakh
    "қотақ (penis)":      r"^([кқkq][оo][тt]т?[аaоo][кқгғґkgq]|[кқ]отабас|[кқ]атокбас|[кқ]отаұбас)",
    "ам (vulva)":         r"^(ам(ы|дар|ш[ыи][қк]|щ[ыи][қк]|ға$|га$)|amy)",
    "сігу (to f*ck)":     r"^(с[іиы][гк]|si[gk]|sy[gk]|сг[еиіы]|ссг)",
    "шешеңнің амы":       r"^(шеше|ене|ана|әнә|ана)\w*(амы|амын|с[іиы]?г|ск[иі]н|сгин|сгейн|сқиін|сгеин)",
    "жалап (whore)":      r"^(жалап|жалаб|jalap)",
    "ташақ (testicles)":  r"^та[шс]а[қкғг]",
    "шшс / 667 (abbr. шешеңді сігейін)": r"^(шшс|щщс|шщс|щшс|шшшс|шшешс|667)$",
    "зб (abbr. заебал)": r"^зб$",
    # Russian mat (very common inside Kazakh reviews; see labeling rules)
    "бля":                r"^(бля|бләт|блә|blya|blyat)|^білә$",
    "хуй":                r"^(хуй|хуя|хуе|хуё|хул[иь]|херн[яюиеё]|херь|наху|нихуя|нехуя|нехуё|дахуя|похуй|поху$|ахуе|охуе|xuy|hui|nahu)",
    "пизд":               r"^(пизд|пезд|пздц|пздетс|pizd|pezd)",
    "еб-":                r"^(еба|ёба|ебл|ебу|ебн|ёб|заеб|зайба|зайеб|заиба|уеб|уёб|выеб|наеб|грёба|греба|трах|yeba|yiba|ziba|zaeb|zaib)",
    "долбаёб":            r"^(д[оа]лба[её]|д[оа]лба[йиюя][оеё]|д[оа]лбаю|д[оа]лбаев|д[ао]йбал|д[оа]лбо[йи][оеё]?[пб])",
    "сука":               r"^(сука|суки|сучк|сучар|suka)",
    "мудак":              r"^муда[кч]",
    "залупа":             r"^залуп",
    "гандон":             r"^(гандон|ганд[оа]н|гондон|кагандон)",
    "пидор":              r"^(пид[оа]р|пидр|пидараз|pid[oa]r|pidaraz)",
    "шлюха":              r"^шлюх",
    "говно":              r"^(говн|гавн|gavno|govno)",
}

# Words that START like a profane word but are innocent.
EXCLUDE_WORDS = re.compile(
    r"^(сигнал|signal|сигма|сигар|сикрет|сикер|сыкыр|сықыр|сыган|сиган|сигандар|"
    r"сикир|сике$|сигун|сигек|сиксен|сикс|сико|сика|сиг$|сик$|сикунд|сикунт|сикл|сыктывкар|"
    r"амин|амир|амилай|амина|амыр|"
    r"котлет|котел|котик|коттакон|ебе|"
    r"гандбол)"
)

# ---------------------------------------------------------------------------
# 2. REVIEW: vulgar but context-dependent -> every hit was checked by hand.
# ---------------------------------------------------------------------------
REVIEW_PATTERNS = {
    "көт (ass)":      r"^[кқ]өт(і|ің|ке|те|тер|ен|$)",   # but NOT көтеру (to lift)
    "боқ (shit)":     r"^(боқ|бок)($|[ққ]|тың|тық|ты$|қа|қо|қы|пын)",
    "ам (exact)":     r"^ам$",
    "қаншық (bitch)": r"^[кқ]ан[шщ][ыи][кқ]",
    "далбан":         r"^далбан",
    "бл (бля OR бұл)": r"^бл$",     # short for бля, but also for бұл ("this") -> hand-checked
}
REVIEW_EXCLUDE = re.compile(r"^(көтер|котер|боқта|бокта|бокс|боков|бока)")

# ---------------------------------------------------------------------------
# 3. INSULTS (label 1 by our shared class definition, see README)
#    INSULT_STRONG: counted automatically in app-store reviews
#    INSULT_REVIEW: words that also have an innocent meaning -> checked by hand
#    (in marketplace / maps / bookstore reviews ALL insult hits are checked by hand)
# ---------------------------------------------------------------------------
INSULT_STRONG = re.compile(
    r"^(мал(дар|ски|ский|скии|скй|сың|сыз|сыңдар|сындар|сын|го|ғо|ға|га|ғой|дарски|ыски|иски|$)|"
    r"есек($|тер|сің|бас)|ешак|ешек|ещак|ишак|ақымақ|акымак|топас|лох|дебил|дибил|идиот|урод|кретин|"
    r"дурак|тупо|чорт|черт|шошқа|малғұн|жынды|мразь|тварь|козел|сволоч|далбан|"
    r"жалеп)"
)
INSULT_REVIEW = re.compile(
    r"^(ит|иттер|итсің|иттің|иттын|итн|итке|иттік|[қк]атын(дар|ды|дарды|сың|ски|ға)?)$"
)
INSULT_EXCLUDE = re.compile(r"^(маладе|маладц|малимет|малымет|маленьк|малыш|мало|малика|малейкум|"
                            r"малатс|малая|малхана|малтабу|есеке|қатынас|катынас)")

LOOKALIKE_WORDS = re.compile(  # innocent words that look like profane ones
    r"^(сигнал|шешеді|шешу|шешім|шешен|бока|амин|аминь|көтер|котер|кот$|боқта|бокта|"
    r"сикрет|әдебиет|себеб|кебаб|мегабайт|мегебайт|бокс|маладе|малимет|қатынас|катынас|далбаса)"
)

KAZAKH_OBSCENE = {"қотақ (penis)", "ам (vulva)", "сігу (to f*ck)", "шешеңнің амы", "жалап (whore)",
                  "шшс / 667 (abbr. шешеңді сігейін)",
                  "ташақ (testicles)", "көт (ass)", "боқ (shit)", "ам (exact)", "қаншық (bitch)"}

_STRONG = {k: re.compile(v) for k, v in STRONG_PATTERNS.items()}
_REVIEW = {k: re.compile(v) for k, v in REVIEW_PATTERNS.items()}
_WORD = re.compile(r"[\w*@#$]+", re.UNICODE)
_MASK = re.compile(r"[*@#$]+")
_REPEAT = re.compile(r"(.)\1{2,}")


def words(text):
    """Normalized (Latin->Cyrillic, see normalize.py), lower-cased words, repeats collapsed."""
    text = _REPEAT.sub(r"\1", normalize_text(text, leet=False).lower())  # keep @ $ as masks here
    return _WORD.findall(text)


def _masked_regex(tok):
    """'бл**' -> regex 'бл.{1,4}' : each run of mask symbols stands for 1..(n+2) letters."""
    parts = _MASK.split(tok)
    runs = _MASK.findall(tok)
    rx = re.escape(parts[0])
    for run, part in zip(runs, parts[1:]):
        rx += ".{1,%d}" % (len(run) + 2) + re.escape(part)
    return re.compile("^" + rx)


def match_terms(text, known_profane_words=None):
    """Return (strong_hits, review_hits) for one text."""
    strong, review = [], []
    for w in words(text):
        if _MASK.search(w) and not w.startswith("@"):
            letters = _MASK.sub("", w)
            if known_profane_words and len(letters) >= 1 and len(w) >= 3:
                rx = _masked_regex(w)
                if any(rx.match(k) for k in known_profane_words):
                    strong.append(("masked", w))
            continue
        if EXCLUDE_WORDS.match(w):
            continue
        hit = next((name for name, rx in _STRONG.items() if rx.search(w)), None)
        if hit:
            strong.append((hit, w))
            continue
        if not REVIEW_EXCLUDE.match(w):
            hit = next((name for name, rx in _REVIEW.items() if rx.search(w)), None)
            if hit:
                review.append((hit, w))
    return strong, review


LITERAL_MAL = re.compile(r"мал шаруашылы|төрт түлік|мал базар|мал сат|мал ізде|мал изде|мал азығ|мал жоқ па|"
                         r"мал-мүлік|мал мүлік|қара мал")


def match_insults(text):
    """Return (strong_insults, review_insults) word lists."""
    strong, review = [], []
    if LITERAL_MAL.search(normalize_text(text).lower()):
        return strong, review
    for w in words(text):
        if INSULT_EXCLUDE.match(w):
            continue
        if INSULT_STRONG.match(w):
            strong.append(w)
        elif INSULT_REVIEW.match(w):
            review.append(w)
    return strong, review


def has_insult(text):
    return any(match_insults(text))


def profanity_type(text, known_profane_words=None):
    """Category for analysis only (the label stays binary):
    kazakh_obscene / russian_obscene / mixed_obscene / masked / insult_only / other"""
    strong, review = match_terms(text, known_profane_words)
    names = {n for n, _ in strong + review if n != "masked"}
    kz, ru = bool(names & KAZAKH_OBSCENE), bool(names - KAZAKH_OBSCENE)
    if kz and ru:
        return "mixed_obscene"
    if kz:
        return "kazakh_obscene"
    if ru:
        return "russian_obscene"
    if strong:                                   # only masked words
        return "masked"
    if has_insult(text):
        return "insult_only"
    return "other"                               # hand-labeled (e.g. "шешең" used as a curse)


def script_type(text):
    """latin / cyrillic_with_kazakh_letters / cyrillic_no_kazakh_letters"""
    lat = len(re.findall(r"[a-zA-Záǵıńóúýşçğüöäñ]", text))
    cyr = len(re.findall(r"[а-яА-ЯёЁәғқңөұүһіӘҒҚҢӨҰҮҺІ]", text))
    if lat > cyr:
        return "latin"
    if re.search(r"[әғқңөұүһіӘҒҚҢӨҰҮҺІ]", text):
        return "cyrillic_with_kazakh_letters"
    return "cyrillic_no_kazakh_letters"


def has_lookalike(text):
    return any(LOOKALIKE_WORDS.match(w) for w in words(text))
