"""Small hand-written lexicons for the rule-based checks (English source; Hindi / German targets).

IMPORTANT (evaluation honesty): ``BILINGUAL_ENTITIES`` is a closed list. For Hindi/German it contains the same entities
as the synthetic-data gazetteer (``mml.synth.lang``), so any feature derived from it (``lex_*``) will look better on
synthetic test data than it would on real MT errors. Phase 4 reports results with and without these features.
"""

from __future__ import annotations

# ---------------------------------------------------------------- negation cues
EN_NEG_WORDS = frozenset({"not", "no", "never", "none", "nobody", "nothing", "neither", "without", "cannot",
                          "nowhere", "nor"})
NEG_TARGET: dict[str, frozenset[str]] = {
    "hin_Deva": frozenset({"नहीं", "नही", "न", "मत", "बिना"}),
    "deu_Latn": frozenset({"nicht", "kein", "keine", "keinen", "keinem", "keiner", "keines", "nie", "niemals",
                           "niemand", "nichts", "ohne", "weder", "nirgends", "nirgendwo"}),
}

# ---------------------------------------------------------------- number words (2..20 and tens; "one" is too ambiguous)
EN_NUM_WORDS = {w: n for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty".split())}
EN_NUM_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90})
EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
EN_UNITS = {w: n for w, n in EN_NUM_WORDS.items() if 1 <= n <= 9}
AMBIGUOUS_NUMBER_WORDS = frozenset({"zero", "one"})

TARGET_NUM_WORDS: dict[str, dict[str, int]] = {
    "hin_Deva": {
        "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "छ": 6, "सात": 7, "आठ": 8, "नौ": 9,
        "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "पन्द्रह": 15, "सोलह": 16,
        "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20, "तीस": 30, "चालीस": 40, "पचास": 50, "साठ": 60,
        "सत्तर": 70, "अस्सी": 80, "नब्बे": 90},
    "deu_Latn": {
        "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10,
        "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15, "sechzehn": 16, "siebzehn": 17,
        "achtzehn": 18, "neunzehn": 19, "zwanzig": 20, "dreißig": 30, "vierzig": 40, "fünfzig": 50, "sechzig": 60,
        "siebzig": 70, "achtzig": 80, "neunzig": 90},
}
DECIMAL_SEPARATOR = {"eng_Latn": ".", "hin_Deva": ".", "deu_Latn": ","}

# ---------------------------------------------------------------- dates
EN_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
             "november", "december")
EN_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
HI_MONTH_VARIANTS = {"सितम्बर": 8, "अक्तूबर": 9, "नवम्बर": 10, "दिसम्बर": 11}
HI_WEEKDAY_VARIANTS = {"बृहस्पतिवार": 3}

# ---------------------------------------------------------------- entities
# English surface form(s) -> target surface form(s). English keys use "|" for alternatives.
BILINGUAL_ENTITIES: dict[str, dict[str, tuple[str, ...]]] = {
    "hin_Deva": {
        "Delhi": ("दिल्ली",), "Mumbai": ("मुंबई",), "Kolkata": ("कोलकाता",), "Chennai": ("चेन्नई",),
        "Bangalore|Bengaluru": ("बेंगलुरु",), "Hyderabad": ("हैदराबाद",), "Pune": ("पुणे",), "Jaipur": ("जयपुर",),
        "Lucknow": ("लखनऊ",), "Kerala": ("केरल",), "Gujarat": ("गुजरात",), "Rajasthan": ("राजस्थान",),
        "Punjab": ("पंजाब",), "Maharashtra": ("महाराष्ट्र",), "India": ("भारत",), "Pakistan": ("पाकिस्तान",),
        "China": ("चीन",), "Japan": ("जापान",), "Germany": ("जर्मनी",), "France": ("फ्रांस",),
        "Britain|United Kingdom": ("ब्रिटेन",), "Australia": ("ऑस्ट्रेलिया",), "Canada": ("कनाडा",),
        "Russia": ("रूस",), "America|United States": ("अमेरिका",), "Berlin": ("बर्लिन",), "Paris": ("पेरिस",),
        "London": ("लंदन",), "Tokyo": ("टोक्यो",), "New York": ("न्यूयॉर्क",), "Washington": ("वाशिंगटन",),
        "Italy": ("इटली",), "Spain": ("स्पेन",), "Brazil": ("ब्राजील",), "Egypt": ("मिस्र",), "Nepal": ("नेपाल",),
        "Sri Lanka": ("श्रीलंका",), "Bangladesh": ("बांग्लादेश",), "Google": ("गूगल",), "Apple": ("ऐप्पल",),
        "Microsoft": ("माइक्रोसॉफ्ट",), "Amazon": ("अमेज़न",), "Tata": ("टाटा",), "Infosys": ("इंफोसिस",),
        "United Nations": ("संयुक्त राष्ट्र",), "Facebook": ("फेसबुक",), "Twitter": ("ट्विटर",), "NASA": ("नासा",),
        "UNESCO": ("यूनेस्को",), "ISRO": ("इसरो",), "BBC": ("बीबीसी",), "Reuters": ("रॉयटर्स",),
        "Oxford University|University of Oxford": ("ऑक्सफोर्ड विश्वविद्यालय",),
        "Narendra Modi": ("नरेंद्र मोदी",), "Rahul Gandhi": ("राहुल गांधी",), "Sonia Gandhi": ("सोनिया गांधी",),
        "Sachin Tendulkar": ("सचिन तेंदुलकर",), "Amitabh Bachchan": ("अमिताभ बच्चन",), "Anita Rao": ("अनीता राव",),
        "Mahatma Gandhi": ("महात्मा गांधी",), "Jawaharlal Nehru": ("जवाहरलाल नेहरू",),
        "Angela Merkel": ("एंजेला मर्केल",), "Barack Obama": ("बराक ओबामा",), "Donald Trump": ("डोनाल्ड ट्रंप",),
        "Joe Biden": ("जो बाइडेन",), "Albert Einstein": ("अल्बर्ट आइंस्टीन",), "Imran Khan": ("इमरान खान",),
        "Virat Kohli": ("विराट कोहली",),
    },
    "deu_Latn": {
        "Germany": ("Deutschland",), "Munich": ("München",), "Vienna": ("Wien",), "Cologne": ("Köln",),
        "Rome": ("Rom",), "Tokyo": ("Tokio",), "Austria": ("Österreich",), "Italy": ("Italien",),
        "Spain": ("Spanien",), "Russia": ("Russland",), "India": ("Indien",), "Brazil": ("Brasilien",),
        "Australia": ("Australien",), "Egypt": ("Ägypten",), "Bavaria": ("Bayern",), "France": ("Frankreich",),
        "Canada": ("Kanada",), "Berlin": ("Berlin",), "Hamburg": ("Hamburg",), "Zurich": ("Zürich",),
        "Paris": ("Paris",), "London": ("London",), "Madrid": ("Madrid",), "Mumbai": ("Mumbai",),
        "Delhi": ("Delhi",), "Kerala": ("Kerala",), "China": ("China",), "Japan": ("Japan",),
        "Pakistan": ("Pakistan",), "Nepal": ("Nepal",), "Washington": ("Washington",), "New York": ("New York",),
        "Google": ("Google",), "Apple": ("Apple",), "Microsoft": ("Microsoft",), "Amazon": ("Amazon",),
        "Siemens": ("Siemens",), "Tata": ("Tata",), "Infosys": ("Infosys",), "Volkswagen": ("Volkswagen",),
        "Facebook": ("Facebook",), "Twitter": ("Twitter",), "NASA": ("NASA",), "UNESCO": ("UNESCO",),
        "BBC": ("BBC",), "Reuters": ("Reuters",), "Oxford University|University of Oxford": ("Universität Oxford",),
        "United Nations": ("Vereinten Nationen", "Vereinte Nationen"),
        "Narendra Modi": ("Narendra Modi",), "Anita Rao": ("Anita Rao",), "Angela Merkel": ("Angela Merkel",),
        "Olaf Scholz": ("Olaf Scholz",), "Albert Einstein": ("Albert Einstein",), "Barack Obama": ("Barack Obama",),
        "Donald Trump": ("Donald Trump",), "Joe Biden": ("Joe Biden",), "Emmanuel Macron": ("Emmanuel Macron",),
        "Vladimir Putin": ("Wladimir Putin",), "Sonia Gandhi": ("Sonia Gandhi",),
        "Mahatma Gandhi": ("Mahatma Gandhi",), "Sachin Tendulkar": ("Sachin Tendulkar",),
        "Thomas Muller|Thomas Müller": ("Thomas Müller",),
    },
}
# Extra English->German exonyms used by the generic (non-lexicon) missing-entity check
EXONYMS_DE = {"germany": "deutschland", "munich": "münchen", "vienna": "wien", "cologne": "köln", "rome": "rom",
              "tokyo": "tokio", "austria": "österreich", "italy": "italien", "spain": "spanien", "russia": "russland",
              "india": "indien", "brazil": "brasilien", "australia": "australien", "egypt": "ägypten",
              "bavaria": "bayern", "switzerland": "schweiz", "netherlands": "niederlande", "belgium": "belgien",
              "poland": "polen", "greece": "griechenland", "turkey": "türkei", "sweden": "schweden",
              "norway": "norwegen", "denmark": "dänemark", "hungary": "ungarn", "zurich": "zürich",
              "united states": "vereinigte staaten", "united nations": "vereinten nationen",
              "european union": "europäische union", "iceland": "island", "new zealand": "neuseeland",
              "ethiopia": "äthiopien", "cambodia": "kambodscha", "britain": "großbritannien",
              "united kingdom": "vereinigtes königreich", "france": "frankreich", "greece": "griechenland",
              "ireland": "irland", "scotland": "schottland", "england": "england", "wales": "wales",
              "finland": "finnland", "czech republic": "tschechische republik", "morocco": "marokko",
              "south africa": "südafrika", "south korea": "südkorea", "north korea": "nordkorea",
              "saudi arabia": "saudi-arabien", "thailand": "thailand", "indonesia": "indonesien",
              "mexico": "mexiko", "peru": "peru", "kenya": "kenia", "tanzania": "tansania", "uganda": "uganda",
              "israel": "israel", "iran": "iran", "iraq": "irak", "syria": "syrien", "lebanon": "libanon",
              "jordan": "jordanien", "afghanistan": "afghanistan", "nepal": "nepal", "bhutan": "bhutan",
              "sri lanka": "sri lanka", "singapore": "singapur", "philippines": "philippinen",
              "vatican": "vatikan", "athens": "athen", "prague": "prag", "warsaw": "warschau",
              "moscow": "moskau", "brussels": "brüssel", "geneva": "genf", "lisbon": "lissabon",
              "florence": "florenz", "venice": "venedig", "naples": "neapel", "copenhagen": "kopenhagen",
              "mediterranean sea": "mittelmeer", "black sea": "schwarzes meer", "atlantic": "atlantik",
              "pacific": "pazifik", "world war ii": "zweiten weltkrieg", "world war": "weltkrieg"}

# Capitalised English words that are not (translatable) named entities
NON_ENTITY_WORDS = frozenset({"earth", "sun", "moon", "internet", "tv", "bce", "ce", "ad", "bc", "god", "web", "ok",
                              "mr", "mrs", "ms", "dr", "sir", "saint", "st", "olympics", "olympic", "christmas",
                              "easter", "new year", "world war", "world war ii", "olympic games"})
DEMONYMS = frozenset({"greek", "french", "dutch", "swiss", "thai", "irish", "welsh", "scottish", "english", "spanish",
                      "danish", "finnish", "polish", "turkish", "british", "kurdish", "flemish", "arab", "jewish",
                      "muslim", "christian", "hindu", "buddhist", "catholic", "protestant", "roman", "soviet"})
DEMONYM_SUFFIXES = ("an", "ian", "ese", "ish", "ic", "ans", "ians", "ch", "i")

# English unit variants -> target forms (closed list, see module caveat; German matched as substring because of compounds)
UNITS: dict[str, dict[str, tuple[str, ...]]] = {
    "hin_Deva": {
        "kilometre|kilometer|kilometres|kilometers|km": ("किलोमीटर", "किमी"),
        "metre|meter|metres|meters": ("मीटर",),
        "kilogram|kilograms|kilogramme|kg": ("किलोग्राम", "किग्रा", "किलो"),
        "gram|grams": ("ग्राम",),
        "million|millions": ("मिलियन", "दस लाख", "लाख"),
        "billion|billions": ("अरब", "बिलियन"),
        "dollar|dollars|$": ("डॉलर", "$"),
        "rupee|rupees": ("रुपये", "रुपए", "रुपया", "रु"),
        "euro|euros|€": ("यूरो", "€"),
        "hour|hours": ("घंटे", "घंटा", "घण्टे", "घंटों"),
        "minute|minutes": ("मिनट",),
    },
    "deu_Latn": {
        "kilometre|kilometer|kilometres|kilometers|km": ("kilometer", "km"),
        "metre|meter|metres|meters": ("meter",),
        "kilogram|kilograms|kilogramme|kg": ("kilogramm", "kg"),
        "gram|grams": ("gramm",),
        "million|millions": ("million", "millionen", "mio"),
        "billion|billions": ("milliarde", "milliarden", "mrd"),
        "dollar|dollars|$": ("dollar", "$"),
        "euro|euros|€": ("euro", "€"),
        "hour|hours": ("stunde", "stunden", "std"),
        "minute|minutes": ("minute", "minuten", "min"),
    },
}
