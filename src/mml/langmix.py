"""Language / script analysis for mixed input (Hinglish) and for checking that the *output* is in one language.

The pipeline translates **English** into the chosen language. Real users also type Hinglish ("Mera naam Rahul hai, please kal tak
report bhej do"), Hindi in Devanagari, or a mix. NLLB is told the source is English, so Romanised Hindi is simply copied through, and the
result is a mix of scripts. This module provides the pure (model-free, unit-testable) pieces:

* :func:`classify` - which kind of input is it (english / hinglish / hindi_script / mixed_script / other)?
* :func:`to_devanagari` - best-effort rewrite of *known* Romanised Hindi words into Devanagari (names and English words are left alone),
  so a Hindi-capable MT model can read a code-mixed sentence.
* :func:`output_issues` - does the translation contain words of the *wrong* language (untranslated copies, wrong script)?

Hinglish spelling is not standardised ("nahi" / "nahin" / "nhi"), so detection is lexicon-based and deliberately conservative: a few very
common, unambiguous words are enough, and words that are also English ("main", "do", "me", "to") only count as weak support.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
WORD = re.compile(r"[A-Za-z]+")
OTHER_SCRIPTS = re.compile(r"[؀-ۿЀ-ӿ一-鿿぀-ヿ가-힯ঀ-৿਀-੿]")

# Romanised Hindi -> Devanagari. STRONG: (almost) never an English word. WEAK: also an English word or a name, so they only add support.
STRONG: dict[str, str] = {
    "hai": "है", "hain": "हैं", "nahi": "नहीं", "nahin": "नहीं", "nhi": "नहीं", "mera": "मेरा", "meri": "मेरी", "mere": "मेरे",
    "tera": "तेरा", "teri": "तेरी", "tere": "तेरे", "tumhara": "तुम्हारा", "tumhari": "तुम्हारी", "hamara": "हमारा", "hamari": "हमारी",
    "aapka": "आपका", "aapki": "आपकी", "aapke": "आपके", "uska": "उसका", "uski": "उसकी", "unka": "उनका", "kya": "क्या",
    "kyun": "क्यों", "kyon": "क्यों", "kyu": "क्यों", "kaise": "कैसे", "kaisa": "कैसा", "kaun": "कौन", "kab": "कब", "kahan": "कहाँ",
    "kitna": "कितना", "kitni": "कितनी", "kitne": "कितने", "aur": "और", "lekin": "लेकिन", "magar": "मगर", "phir": "फिर", "bhi": "भी",
    "tak": "तक", "wala": "वाला", "wali": "वाली", "wale": "वाले", "raha": "रहा", "rahi": "रही", "rahe": "रहे", "hoon": "हूँ",
    "hun": "हूँ", "hu": "हूँ", "hoga": "होगा", "hogi": "होगी", "honge": "होंगे", "tha": "था", "thi": "थी", "gaya": "गया",
    "gayi": "गई", "gai": "गई", "gaye": "गए", "kiya": "किया", "kiye": "किए", "karna": "करना", "karo": "करो", "karke": "करके",
    "karenge": "करेंगे", "karunga": "करूँगा", "karungi": "करूँगी", "jaa": "जा", "jana": "जाना", "jao": "जाओ", "jayega": "जाएगा",
    "jaunga": "जाऊँगा", "aana": "आना", "aao": "आओ", "aayega": "आएगा", "dena": "देना", "diya": "दिया", "dijiye": "दीजिए",
    "lena": "लेना", "liya": "लिया", "bhej": "भेज", "bhejna": "भेजना", "bhejo": "भेजो", "bolna": "बोलना", "bolo": "बोलो",
    "bola": "बोला", "dekhna": "देखना", "dekho": "देखो", "dekha": "देखा", "chahiye": "चाहिए", "sakta": "सकता", "sakti": "सकती",
    "sakte": "सकते", "padega": "पड़ेगा", "bahut": "बहुत", "bohot": "बहुत", "thoda": "थोड़ा", "zyada": "ज़्यादा", "jyada": "ज़्यादा",
    "accha": "अच्छा", "achha": "अच्छा", "acha": "अच्छा", "theek": "ठीक", "thik": "ठीक", "abhi": "अभी", "kal": "कल", "aaj": "आज",
    "parso": "परसों", "subah": "सुबह", "shaam": "शाम", "raat": "रात", "ghar": "घर", "paise": "पैसे", "paisa": "पैसा",
    "kaam": "काम", "naam": "नाम", "saal": "साल", "mujhe": "मुझे", "tumhe": "तुम्हें", "tujhe": "तुझे", "humein": "हमें",
    "unhe": "उन्हें", "isko": "इसको", "usko": "उसको", "yeh": "यह", "ye": "ये", "woh": "वह", "wo": "वो", "yahan": "यहाँ",
    "wahan": "वहाँ", "agar": "अगर", "kyunki": "क्योंकि", "isliye": "इसलिए", "matlab": "मतलब",
    "bilkul": "बिल्कुल", "sirf": "सिर्फ़", "bas": "बस", "yaar": "यार", "bhai": "भाई", "dost": "दोस्त", "sab": "सब", "sabhi": "सभी",
    "kuch": "कुछ", "koi": "कोई", "kisi": "किसी", "apna": "अपना", "apni": "अपनी", "apne": "अपने", "tum": "तुम", "aap": "आप",
    "mujhse": "मुझसे", "toh": "तो", "kar": "कर", "chalo": "चलो", "chal": "चल", "ruko": "रुको", "suno": "सुनो",
    "samajh": "समझ", "pata": "पता", "jaldi": "जल्दी", "zaroor": "ज़रूर", "shukriya": "शुक्रिया", "dhanyavaad": "धन्यवाद",
}
WEAK: dict[str, str] = {
    "main": "मैं", "mai": "मैं", "mein": "में", "hum": "हम", "par": "पर", "pe": "पे", "ab": "अब", "do": "दो", "ke": "के", "ki": "की",
    "ka": "का", "ko": "को", "se": "से", "ho": "हो", "log": "लोग", "tab": "तब", "jab": "जब", "lo": "लो", "din": "दिन",
}
_LEXICON = {**WEAK, **STRONG}


@dataclass
class LangInfo:
    """What kind of text this is. ``label`` is one of english | hinglish | hindi_script | mixed_script | other."""

    label: str
    devanagari_ratio: float = 0.0
    hinglish_words: list[str] = field(default_factory=list)

    @property
    def is_english(self) -> bool:
        return self.label == "english"


def _words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text)]


def classify(text: str) -> LangInfo:
    """Classify one sentence. Latin-script text is "hinglish" when it contains enough Romanised-Hindi lexicon words:
    two strong words, or one strong word making up at least a fifth of the sentence together with weak ones."""
    dev = len(DEVANAGARI.findall(text))
    lat = len(re.findall(r"[A-Za-z]", text))
    if dev == 0 and lat == 0:
        return LangInfo("other" if OTHER_SCRIPTS.search(text) else "english")      # digits / punctuation only: nothing to convert
    if dev == 0 and OTHER_SCRIPTS.search(text):
        return LangInfo("other")
    ratio = dev / (dev + lat)
    if dev:
        return LangInfo("hindi_script" if ratio >= 0.9 else "mixed_script", ratio)
    words = _words(text)
    strong = [w for w in words if w in STRONG]
    weak = [w for w in words if w in WEAK]
    if words and (len(strong) >= 2 or (len(strong) == 1 and (len(strong) + len(weak)) / len(words) >= 0.2)):
        return LangInfo("hinglish", 0.0, sorted(set(strong + weak)))
    return LangInfo("english")


def to_devanagari(text: str) -> str:
    """Rewrite known Romanised-Hindi words as Devanagari; everything else (names, English words, numbers) is kept as it is."""
    def swap(m: re.Match[str]) -> str:
        w = m.group(0)
        return _LEXICON.get(w.lower(), w)

    return WORD.sub(swap, text)


# --------------------------------------------------------------------------- output language
@dataclass
class LangIssue:
    """A word in the translation that does not belong to the chosen output language."""

    kind: str          # "untranslated" | "wrong_script" | "copied_source"
    text: str
    why: str


UNITS = {"km", "kg", "cm", "mm", "ml", "mg", "gb", "mb", "kb", "tv", "pc", "id", "ok", "am", "pm", "mph", "kmh"}


def _is_exempt_latin(word: str, source: str) -> bool:
    """Latin words that may legitimately stay in a Devanagari translation: acronyms, units, and names written capitalised in the source."""
    if word.isupper() and len(word) <= 6:
        return True
    if word.lower() in UNITS or any(ch.isdigit() for ch in word):
        return True
    return bool(re.search(rf"\b{re.escape(word)}\b", source)) and word[0].isupper()


def output_issues(source_en: str, output: str, target_script: str) -> list[LangIssue]:
    """Check a translation for words of another language. ``target_script`` is ``"devanagari"`` (Hindi) or ``"latin"`` (German).

    * Devanagari target: lowercase Latin words are untranslated (names, acronyms and units are allowed); Romanised-Hindi lexicon words
      are always reported.
    * Latin target: any Devanagari is the wrong script; Romanised-Hindi lexicon words copied through are reported.
    """
    issues: list[LangIssue] = []
    if target_script == "devanagari":
        seen: set[str] = set()
        for w in LATIN_WORD.findall(output):
            if w.lower() in seen:
                continue
            seen.add(w.lower())
            if w.lower() in STRONG:
                issues.append(LangIssue("untranslated", w, f"«{w}» is Romanised Hindi left as it was typed; the output should be Devanagari Hindi."))
            elif not _is_exempt_latin(w, source_en):
                issues.append(LangIssue("untranslated", w, f"«{w}» is Latin-script text inside a Hindi translation (probably left untranslated)."))
    else:
        dev = "".join(DEVANAGARI.findall(output))
        if dev:
            issues.append(LangIssue("wrong_script", dev[:12], "The German translation contains Devanagari (Hindi) characters."))
        for w in dict.fromkeys(_words(output)):
            if w in STRONG:
                issues.append(LangIssue("untranslated", w, f"«{w}» is Romanised Hindi copied into the German translation."))
    src_tokens, out_tokens = source_en.lower().split(), output.lower().split()
    if len(src_tokens) >= 4 and out_tokens:
        copied = sum(t in set(src_tokens) for t in out_tokens) / len(out_tokens)
        if copied >= 0.7:
            issues.append(LangIssue("copied_source", output[:40], "Most of the translation is identical to the source: it was probably not translated."))
    return issues


def script_of(tgt_lang: str) -> str:
    """Map an NLLB language code (``hin_Deva`` / ``deu_Latn``) to the script family used by :func:`output_issues`."""
    return "devanagari" if tgt_lang.endswith("_Deva") else "latin"


def strip_marks(text: str) -> str:
    """NFC-normalise so equal-looking Devanagari strings compare equal."""
    return unicodedata.normalize("NFC", text)
