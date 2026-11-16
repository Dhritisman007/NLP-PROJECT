"""Per-language resources for the error injectors (Hindi, German). Add a pair by adding one ``LangResources``.

Everything here is hand-written and deliberately small; larger entity coverage comes from the NER plug-in
(:mod:`mml.synth.entities`). Coverage limits are documented in the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LangResources:
    """Language-specific knowledge used by :class:`mml.synth.inject_errors.ErrorInjector`."""

    code: str
    # --- negation (MISTRANSLATION / negation flip)
    neg_delete: tuple[str, ...]                 # standalone tokens whose removal flips polarity
    neg_replace: dict[str, str]                 # token -> positive counterpart (lowercase keys)
    neg_insert_rule: str                        # "before_last_word" (hi) | "german" (de)
    neg_word: str                               # word inserted when the sentence has no negation
    neg_skip_next: tuple[str, ...] = ()         # do not delete a negation followed by one of these
    neg_skip_if_contains: tuple[str, ...] = ()  # do not touch sentences containing these (e.g. "sondern")
    finite_verbs: tuple[str, ...] = ()          # German: finite auxiliaries/modals after which "nicht" can go
    verb_aux: tuple[str, ...] = ()              # Hindi: auxiliaries/vector verbs forming the sentence-final verb tail
    # --- structure
    conj: tuple[str, ...] = ()                  # clause-introducing conjunctions (omission of trailing clause)
    symmetric_between: tuple[str, ...] = ()     # tokens that make an A-X-B swap meaningless (A and B)
    # --- ENTITY/NUMBER
    months: tuple[str, ...] = ()
    weekdays: tuple[str, ...] = ()
    units: dict[str, str] = field(default_factory=dict)
    gazetteer: dict[str, tuple[str, ...]] = field(default_factory=dict)  # PER/LOC/ORG -> surface forms
    # --- HALLUCINATION: unrelated, negation-free, number-free sentences
    fillers: tuple[str, ...] = ()


HI = LangResources(
    code="hi",
    neg_delete=("नहीं",),
    neg_replace={},
    neg_insert_rule="before_verb_tail",
    neg_word="नहीं",
    verb_aux=("है", "हैं", "था", "थी", "थे", "थीं", "हो", "होगा", "होगी", "होंगे", "हुआ", "हुई", "हुए", "हुईं",
              "गया", "गई", "गए", "गयी", "गये", "गईं", "सकता", "सकती", "सकते", "रहा", "रही", "रहे", "रहीं",
              "चुका", "चुकी", "चुके", "दिया", "दी", "दिए", "लिया", "ली", "जाता", "जाती", "जाते"),
    neg_skip_next=("तो", "कि"),
    conj=("और", "लेकिन", "मगर", "परंतु", "क्योंकि", "जबकि", "ताकि"),
    symmetric_between=("और", "तथा", "या", "व", "एवं"),
    months=("जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर",
            "नवंबर", "दिसंबर"),
    weekdays=("सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"),
    units={"किलोमीटर": "मीटर", "मीटर": "किलोमीटर", "किलोग्राम": "ग्राम", "ग्राम": "किलोग्राम",
           "मिलियन": "अरब", "अरब": "मिलियन", "डॉलर": "रुपये", "रुपये": "डॉलर", "यूरो": "डॉलर",
           "घंटे": "मिनट", "मिनट": "घंटे"},
    gazetteer={
        "PER": ("नरेंद्र मोदी", "राहुल गांधी", "सोनिया गांधी", "सचिन तेंदुलकर", "अमिताभ बच्चन", "अनीता राव",
                "सुनील शर्मा", "रवि कुमार", "प्रिया सिंह", "महात्मा गांधी", "जवाहरलाल नेहरू", "एंजेला मर्केल",
                "बराक ओबामा", "डोनाल्ड ट्रंप", "जो बाइडेन", "अल्बर्ट आइंस्टीन", "इमरान खान", "विराट कोहली"),
        "LOC": ("दिल्ली", "मुंबई", "कोलकाता", "चेन्नई", "बेंगलुरु", "हैदराबाद", "पुणे", "जयपुर", "लखनऊ", "केरल",
                "गुजरात", "राजस्थान", "पंजाब", "महाराष्ट्र", "भारत", "पाकिस्तान", "चीन", "जापान", "जर्मनी",
                "फ्रांस", "ब्रिटेन", "ऑस्ट्रेलिया", "कनाडा", "रूस", "अमेरिका", "बर्लिन", "पेरिस", "लंदन", "टोक्यो",
                "न्यूयॉर्क", "वाशिंगटन", "इटली", "स्पेन", "ब्राजील", "मिस्र", "नेपाल", "श्रीलंका", "बांग्लादेश"),
        "ORG": ("गूगल", "ऐप्पल", "माइक्रोसॉफ्ट", "अमेज़न", "टाटा", "इंफोसिस", "संयुक्त राष्ट्र", "फेसबुक", "ट्विटर",
                "नासा", "यूनेस्को", "इसरो", "बीबीसी", "रॉयटर्स", "ऑक्सफोर्ड विश्वविद्यालय"),
    },
    fillers=(
        "पास के बाज़ार में उसी समय एक बड़ा मेला चल रहा था।",
        "आसमान में दिन भर घने बादल छाए रहे।",
        "कई विशेषज्ञों ने इस पर अपनी राय दी।",
        "स्थानीय स्कूल के बच्चों ने भी इसमें हिस्सा लिया।",
        "उसके बाद सभी लोग चाय पीने चले गए।",
        "इस विषय पर एक नई किताब हाल ही में प्रकाशित हुई है।",
        "शहर की मुख्य सड़क पर यातायात काफ़ी धीमा था।",
        "उसी शाम एक पुरानी फिल्म का प्रसारण भी किया गया।",
        "पास की नदी के किनारे कुछ पक्षी दाना चुग रहे थे।",
        "आयोजकों ने सभी मेहमानों को स्वादिष्ट भोजन परोसा।",
    ),
)

DE = LangResources(
    code="de",
    neg_delete=("nicht",),
    neg_replace={"kein": "ein", "keine": "eine", "keinen": "einen", "keinem": "einem", "keiner": "einer",
                 "keines": "eines", "nie": "immer", "niemals": "immer", "niemand": "jemand", "nichts": "etwas"},
    neg_insert_rule="german",
    neg_word="nicht",
    neg_skip_if_contains=("sondern", "nicht nur", "nicht mehr"),
    finite_verbs=("ist", "sind", "war", "waren", "hat", "haben", "hatte", "hatten", "wird", "werden", "wurde",
                  "wurden", "kann", "können", "konnte", "konnten", "muss", "müssen", "musste", "musste",
                  "soll", "sollen", "sollte", "sollten", "will", "wollen", "darf", "dürfen", "gibt", "bleibt"),
    conj=("und", "aber", "weil", "obwohl", "während", "sodass", "doch", "denn", "dass", "wenn"),
    symmetric_between=("und", "oder", "sowie"),
    months=("Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
            "November", "Dezember"),
    weekdays=("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"),
    units={"Kilometer": "Meter", "Meter": "Kilometer", "Kilogramm": "Gramm", "Gramm": "Kilogramm",
           "Millionen": "Milliarden", "Milliarden": "Millionen", "Dollar": "Euro", "Euro": "Dollar",
           "Stunden": "Minuten", "Minuten": "Stunden"},
    gazetteer={
        "PER": ("Narendra Modi", "Anita Rao", "Angela Merkel", "Olaf Scholz", "Albert Einstein", "Barack Obama",
                "Donald Trump", "Joe Biden", "Emmanuel Macron", "Wladimir Putin", "Sonia Gandhi", "Mahatma Gandhi",
                "Sachin Tendulkar", "Thomas Müller"),
        "LOC": ("Berlin", "München", "Hamburg", "Köln", "Wien", "Zürich", "Paris", "London", "Rom", "Madrid",
                "Tokio", "Mumbai", "Delhi", "Kerala", "Deutschland", "Frankreich", "Indien", "China", "Japan",
                "Russland", "Kanada", "Österreich", "Bayern", "Italien", "Spanien", "Brasilien", "Australien",
                "Pakistan", "Nepal", "Ägypten", "Washington", "New York"),
        "ORG": ("Google", "Apple", "Microsoft", "Amazon", "Siemens", "Tata", "Infosys", "Volkswagen", "Facebook",
                "Twitter", "NASA", "UNESCO", "BBC", "Reuters", "Universität Oxford", "Vereinten Nationen"),
    },
    fillers=(
        "In der Nähe fand zur gleichen Zeit ein großer Markt statt.",
        "Den ganzen Tag über war der Himmel von dichten Wolken bedeckt.",
        "Mehrere Experten äußerten sich ebenfalls dazu.",
        "Auch die Kinder der örtlichen Schule nahmen daran teil.",
        "Danach gingen alle gemeinsam Kaffee trinken.",
        "Ein neues Buch zu diesem Thema ist kürzlich erschienen.",
        "Der Verkehr auf der Hauptstraße der Stadt war ziemlich langsam.",
        "Am selben Abend wurde außerdem ein alter Film gezeigt.",
        "Am Flussufer pickten einige Vögel nach Körnern.",
        "Die Veranstalter servierten allen Gästen ein köstliches Essen.",
    ),
)

RESOURCES: dict[str, LangResources] = {"hin_Deva": HI, "deu_Latn": DE}


def get_resources(tgt_lang: str) -> LangResources:
    """Resources for a FLORES target-language code; explains how to add a new language."""
    if tgt_lang not in RESOURCES:
        raise KeyError(f"No synthetic-error resources for {tgt_lang!r}; supported: {sorted(RESOURCES)}. "
                       "Add a LangResources entry in mml/synth/lang.py.")
    return RESOURCES[tgt_lang]
