"""Build the human-evaluation kit for one language pair.

Design
------
* Items are **real MT output** (NLLB translations of FLORES ``devtest`` sources) so that human labels measure the detector on
  real errors, not on the synthetic ones it was built on.
* The sample is **stratified by the detector's flag** (about half flagged, half unflagged), because a plain random sample would
  hold too few flagged sentences to estimate precision. Population-level estimates are re-weighted by the pool's flag rate
  (see ``iaa.py``).
* Annotators are **blind**: the form shows only the source and the translation (no detector output, no reference). The detector's
  flags, findings and the reference live in the hidden ``key.csv`` used only for analysis.
* A few **attention checks** (synthetic items with known labels) are mixed in to check that annotators are reading carefully.
* Item order is shuffled differently for each annotator; ids are stable and carry no information about the stratum.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from mml.config import Config
from mml.data import flores_source, read_flores

LABEL_COLUMNS = ["meaning_preserved", "omission", "mistranslation", "entity_number", "hallucination"]
TYPE_COLUMNS = {"omission": "OMISSION", "mistranslation": "MISTRANSLATION", "entity_number": "ENTITY_NUMBER",
                "hallucination": "HALLUCINATION"}
ANNOTATION_COLUMNS = ["item_id", "source", "translation", *LABEL_COLUMNS, "problem_text", "notes"]
KEY_COLUMNS = ["item_id", "stratum", "pool_idx", "detector_flagged", "detector_types", "detector_p_any", "detector_findings",
               "is_check", "check_label", "check_subtype", "reference"]
LANG_NAMES = {"eng_Latn": "English", "hin_Deva": "Hindi", "deu_Latn": "German"}

GUIDELINES = """# Annotation guidelines

You will see an **English source sentence** and its **machine translation** into {tgt}. Judge only whether the translation
**keeps the meaning of the source**. Do not judge style, fluency or word choice unless it changes the meaning.

For every item answer **"Meaning preserved?"**:

* **Yes**: everything important in the source is in the translation, nothing important is added or changed.
* **No**: the translation loses or changes meaning. Then tick every error type that applies (one or more):

| Type | Tick it when... | Example (source -> translation problem) |
|---|---|---|
| **Omission** | source content is missing from the translation | "She sold the car **and the house**" -> only the car is mentioned |
| **Mistranslation** | the meaning is changed: negation flipped, wrong word sense, who-did-what-to-whom swapped | "He did **not** attend" -> says he attended |
| **Entity / number** | a name, date, number, unit or currency is wrong or missing | "12 May 2022" -> a different date; "4.5 million" -> "45 million" |
| **Hallucination** | content appears in the translation that is not in the source | an extra clause or sentence that the source does not support |

Rules of thumb
* Minor grammar or awkward word order that does **not** change the meaning -> **Meaning preserved = Yes**.
* If you are unsure whether something is an error, choose the more serious reading only if a bilingual reader would be misled.
* Use "problem words" to copy the exact words that are wrong (optional but very helpful); use "notes" for anything unusual.
* Do not look up the reference translation or ask the other annotator: independent judgements are what make the agreement
  statistic meaningful. Some items are **attention checks** with a known answer; read every item carefully.
* Aim for about 2 minutes per item. You may stop and continue later (the HTML form saves your progress in the browser).

Return your file as `annotations_<your name>.csv` (HTML form: use the **Download CSV** button).
"""


@dataclass
class PoolItem:
    """One real MT output with the detector's verdict."""

    idx: int
    src: str
    ref: str
    mt: str
    flagged: bool
    types: list[str]
    p_any: float
    findings: list[str] = field(default_factory=list)


@dataclass
class KitItem:
    """One item of the kit (what annotators see + the hidden key fields)."""

    item_id: str
    source: str
    translation: str
    stratum: str                       # flagged | unflagged | check
    pool_idx: int = -1
    detector_flagged: bool = False
    detector_types: str = ""
    detector_p_any: float = 0.0
    detector_findings: str = ""
    is_check: bool = False
    check_label: str = ""
    check_subtype: str = ""
    reference: str = ""


# --------------------------------------------------------------------------- pool
def pool_path(cfg: Config) -> Path:
    return Path(cfg.raw["paths"]["data_dir"]) / "processed" / "human_eval" / cfg.pair_key / "pool.jsonl"


def build_pool(cfg: Config, pool_n: int = 600, max_words: int = 45, backend: str = "nllb_ct2") -> list[PoolItem]:
    """Translate a seeded sample of FLORES devtest sources and run the detector on every translation.

    The first 300 indices are the same sentences as the Phase 6 real-MT sample, so their translations and signals are cached.
    """
    from mml.correct.span_postedit import build_problems
    from mml.detect.predict import Detector
    from mml.translate import Translator

    data_dir = Path(cfg.raw["paths"]["data_dir"])
    source = flores_source(cfg)
    srcs = read_flores(data_dir, cfg.pair["src_lang"], "devtest", source)
    refs = read_flores(data_dir, cfg.pair["tgt_lang"], "devtest", source)
    phase6 = random.Random(f"{cfg.seed}|realmt").sample(range(len(srcs)), min(300, len(srcs)))
    rest = [i for i in range(len(srcs)) if i not in set(phase6)]
    extra = random.Random(f"{cfg.seed}|humaneval-pool").sample(rest, min(max(pool_n - len(phase6), 0), len(rest)))
    idx = [i for i in sorted(phase6 + extra) if len(srcs[i].split()) <= max_words]
    translator = Translator.from_name(cfg, backend)
    mt = translator.translate_best([srcs[i] for i in idx])
    detector = Detector(cfg)
    dets = detector.detect([(srcs[i], m) for i, m in zip(idx, mt)])
    pool = [PoolItem(i, srcs[i], refs[i], m, bool(d.types), sorted(d.types), d.probs["ANY"],
                     build_problems(srcs[i], m, d) if d.types else []) for i, m, d in zip(idx, mt, dets)]
    path = pool_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(p), ensure_ascii=False) for p in pool) + "\n", encoding="utf-8")
    return pool


def load_pool(cfg: Config) -> list[PoolItem]:
    return [PoolItem(**json.loads(line)) for line in pool_path(cfg).read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------- sampling
def sample_kit(pool: Sequence[PoolItem], checks: Sequence[Any], n: int = 100, n_checks: int = 10, seed: int = 42) -> tuple[list[KitItem], dict[str, Any]]:
    """Stratified sample: (n - n_checks) real items, half flagged / half unflagged, plus ``n_checks`` known-answer items.

    ``checks`` are labelled synthetic examples (``SynthExample``-like: ``src, hyp, ref, label, subtype``). Returns the items in a
    canonical shuffled order (ids assigned after the shuffle) and a meta dict with the strata sizes needed for re-weighting.
    """
    rng = random.Random(f"{seed}|kit")
    n_real = n - n_checks
    flagged = sorted((p for p in pool if p.flagged), key=lambda p: p.idx)
    unflagged = sorted((p for p in pool if not p.flagged), key=lambda p: p.idx)
    want_f = min(n_real // 2, len(flagged))
    want_u = min(n_real - want_f, len(unflagged))
    chosen = rng.sample(flagged, want_f) + rng.sample(unflagged, want_u)
    items = [KitItem("", p.src, p.mt, "flagged" if p.flagged else "unflagged", p.idx, p.flagged, ",".join(p.types), round(p.p_any, 4),
                     " | ".join(p.findings), False, "", "", p.ref) for p in chosen]
    clean = [c for c in checks if c.label == "OK"]
    bad = [c for c in checks if c.label != "OK"]
    n_bad = n_checks // 2 + n_checks % 2
    picked: list[Any] = []
    for t in ("OMISSION", "MISTRANSLATION", "ENTITY_NUMBER", "HALLUCINATION"):      # cover every error type first
        pool_t = sorted((c for c in bad if c.label == t), key=lambda c: c.id)
        if pool_t and len(picked) < n_bad:
            picked.append(rng.choice(pool_t))
    while len(picked) < n_bad and bad:
        picked.append(rng.choice(sorted(bad, key=lambda c: c.id)))
    picked += rng.sample(sorted(clean, key=lambda c: c.id), min(n_checks - len(picked), len(clean)))
    for c in picked:
        items.append(KitItem("", c.src, c.hyp, "check", -1, False, "", 0.0, "", True, c.label, c.subtype, c.ref))
    rng.shuffle(items)
    for k, it in enumerate(items, 1):
        it.item_id = f"item_{k:03d}"
    meta = {"n": len(items), "n_checks": len(picked), "pool_n": len(pool), "pool_flagged": len(flagged), "pool_unflagged": len(unflagged),
            "sampled_flagged": want_f, "sampled_unflagged": want_u, "pool_flag_rate": len(flagged) / max(len(pool), 1), "seed": seed}
    return items, meta


def order_for(items: Sequence[KitItem], annotator: str, seed: int) -> list[KitItem]:
    """The display order for one annotator (different per annotator; ids unchanged)."""
    out = list(items)
    random.Random(f"{seed}|order|{annotator}").shuffle(out)
    return out


# --------------------------------------------------------------------------- writing
def write_annotation_csv(path: Path, items: Sequence[KitItem]) -> None:
    """Blank annotation template (UTF-8 with BOM so spreadsheets display Devanagari correctly)."""
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(ANNOTATION_COLUMNS)
        for it in items:
            w.writerow([it.item_id, it.source, it.translation, "", "", "", "", "", "", ""])


def write_key(path: Path, items: Sequence[KitItem]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(KEY_COLUMNS)
        for it in sorted(items, key=lambda i: i.item_id):
            w.writerow([it.item_id, it.stratum, it.pool_idx, int(it.detector_flagged), it.detector_types, it.detector_p_any,
                        it.detector_findings, int(it.is_check), it.check_label, it.check_subtype, it.reference])


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Translation meaning check - __PAIR__ - __ANNOTATOR__</title>
<style>
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e1;--accent:#2a78d6;--ok:#0b7a3b}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--card:#232321;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--accent:#3987e5;--ok:#4fc27a}}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI","Noto Sans","Noto Sans Devanagari",sans-serif}
main{max-width:860px;margin:0 auto;padding:16px}
h1{font-size:1.25rem;margin:.2rem 0}.muted{color:var(--ink2);font-size:.9rem}
.bar{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 0;z-index:2;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}
.card.done{border-left:4px solid var(--ok)}
.lbl{font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2);margin-top:8px}
.txt{font-size:1.05rem;margin:2px 0 6px;white-space:pre-wrap}
.row{display:flex;gap:16px;flex-wrap:wrap;margin:6px 0}
label{cursor:pointer}input[type=text],textarea{width:100%;box-sizing:border-box;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:inherit}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:8px 14px;font:inherit;cursor:pointer}
details{margin:8px 0}summary{cursor:pointer;color:var(--accent)}
.types{margin-left:4px;padding-left:12px;border-left:3px solid var(--line)}
</style></head><body><main>
<h1>Does the translation keep the meaning? (English &rarr; __TGT__)</h1>
<p class="muted">Annotator: <b id="who">__ANNOTATOR__</b> &middot; __N__ items. Progress is saved automatically in this browser. Click <b>Download CSV</b> when you are done and send the file back.</p>
<details><summary>Guidelines (read first)</summary><div class="muted">__GUIDELINES__</div></details>
<div class="bar"><span id="progress">0 / __N__ done</span><button id="dl">Download CSV</button><button id="reset" style="background:#8a8984">Clear my answers</button></div>
<div id="items"></div>
</main>
<script>
const ITEMS = __ITEMS__;
const KEY = "mml-human-eval-__PAIR__-__ANNOTATOR__";
const TYPES = [["omission","Omission (source content missing)"],["mistranslation","Mistranslation (meaning changed, e.g. negation)"],["entity_number","Entity / number (name, date, number, unit wrong or missing)"],["hallucination","Hallucination (content not in the source)"]];
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
function save(){ try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} updateProgress(); }
function get(id){ return state[id] || (state[id] = {mp:"", types:[], problem:"", notes:""}); }
function isDone(id){ const s = state[id]; return !!s && (s.mp === "1" || (s.mp === "0" && s.types.length > 0)); }
function updateProgress(){ const n = ITEMS.filter(i => isDone(i.item_id)).length; document.getElementById("progress").textContent = n + " / " + ITEMS.length + " done";
  ITEMS.forEach(i => { const c = document.getElementById("c-" + i.item_id); if (c) c.classList.toggle("done", isDone(i.item_id)); }); }
function el(tag, props, kids){ const e = document.createElement(tag); Object.assign(e, props || {}); (kids || []).forEach(k => e.append(k)); return e; }
function render(){
  const root = document.getElementById("items"); root.textContent = "";
  ITEMS.forEach((it, n) => {
    const s = get(it.item_id);
    const card = el("div", {className:"card", id:"c-" + it.item_id});
    card.append(el("div", {className:"muted", textContent:"Item " + (n + 1) + " of " + ITEMS.length + " (" + it.item_id + ")"}));
    card.append(el("div", {className:"lbl", textContent:"Source (English)"}), el("div", {className:"txt", textContent:it.source}));
    card.append(el("div", {className:"lbl", textContent:"Translation (__TGT__)"}), el("div", {className:"txt", textContent:it.translation}));
    const mkRadio = (val, text) => { const r = el("input", {type:"radio", name:"mp-" + it.item_id, value:val, checked:s.mp === val});
      r.addEventListener("change", () => { s.mp = val; if (val === "1") { s.types = []; } save(); typesBox.style.display = val === "0" ? "block" : "none"; render2(it, typesBox, s); });
      return el("label", {}, [r, " " + text]); };
    card.append(el("div", {className:"lbl", textContent:"Meaning preserved?"}),
      el("div", {className:"row"}, [mkRadio("1", "Yes"), mkRadio("0", "No - meaning lost or changed")]));
    const typesBox = el("div", {className:"types"}); typesBox.style.display = s.mp === "0" ? "block" : "none";
    card.append(typesBox); render2(it, typesBox, s);
    root.append(card);
  });
  updateProgress();
}
function render2(it, box, s){
  box.textContent = ""; box.append(el("div", {className:"lbl", textContent:"What kind of error? (tick all that apply)"}));
  TYPES.forEach(([k, text]) => { const c = el("input", {type:"checkbox", checked:s.types.includes(k)});
    c.addEventListener("change", () => { s.types = c.checked ? [...new Set([...s.types, k])] : s.types.filter(x => x !== k); save(); });
    box.append(el("div", {}, [el("label", {}, [c, " " + text])])); });
  const p = el("input", {type:"text", placeholder:"Problem words (copy the wrong or missing words) - optional", value:s.problem});
  p.addEventListener("input", () => { s.problem = p.value; save(); });
  const n = el("textarea", {rows:2, placeholder:"Notes - optional", value:s.notes});
  n.addEventListener("input", () => { s.notes = n.value; save(); });
  box.append(el("div", {className:"lbl", textContent:"Problem words"}), p, el("div", {className:"lbl", textContent:"Notes"}), n);
}
function csvEsc(v){ return '"' + String(v == null ? "" : v).replace(/"/g, '""') + '"'; }
document.getElementById("dl").addEventListener("click", () => {
  const head = ["item_id","source","translation","meaning_preserved","omission","mistranslation","entity_number","hallucination","problem_text","notes"];
  const rows = [head.map(csvEsc).join(",")];
  ITEMS.slice().sort((a, b) => a.item_id.localeCompare(b.item_id)).forEach(it => { const s = state[it.item_id] || {mp:"", types:[], problem:"", notes:""};
    const t = k => s.mp === "" ? "" : (s.types.includes(k) ? "1" : "0");
    rows.push([it.item_id, it.source, it.translation, s.mp, t("omission"), t("mistranslation"), t("entity_number"), t("hallucination"), s.problem, s.notes].map(csvEsc).join(",")); });
  const blob = new Blob(["\\ufeff" + rows.join("\\r\\n")], {type:"text/csv;charset=utf-8"});
  const a = el("a", {href:URL.createObjectURL(blob), download:"annotations___ANNOTATOR__.csv"}); document.body.append(a); a.click(); a.remove();
});
document.getElementById("reset").addEventListener("click", () => { if (confirm("Delete all your answers in this browser?")) { state = {}; save(); render(); } });
render();
</script></body></html>
"""


def render_html(items: Sequence[KitItem], pair: str, annotator: str, tgt_lang: str, guidelines_md: str) -> str:
    """Self-contained offline form (no network, no external files). Shows only the source and the translation."""
    import html

    payload = json.dumps([{"item_id": i.item_id, "source": i.source, "translation": i.translation} for i in items], ensure_ascii=False)
    payload = payload.replace("</", "<\\/")
    guide = "<br>".join(html.escape(line) for line in guidelines_md.splitlines())
    return (HTML_TEMPLATE.replace("__PAIR__", html.escape(pair)).replace("__ANNOTATOR__", html.escape(annotator))
            .replace("__TGT__", html.escape(LANG_NAMES.get(tgt_lang, tgt_lang))).replace("__N__", str(len(items)))
            .replace("__GUIDELINES__", guide).replace("__ITEMS__", payload))


def write_kit(cfg: Config, items: Sequence[KitItem], meta: dict[str, Any], annotators: Sequence[str] = ("A", "B"),
              out_dir: Path | None = None) -> Path:
    """Write the whole kit: key, meta, guidelines, and a CSV + HTML per annotator."""
    out = out_dir or Path(cfg.raw["run"]["output_dir"]).parent / "human_eval" / cfg.pair_key
    out.mkdir(parents=True, exist_ok=True)
    share = out / "for_annotators"                 # the ONLY folder that may be sent to annotators
    share.mkdir(exist_ok=True)
    tgt = cfg.pair["tgt_lang"]
    guide = GUIDELINES.format(tgt=LANG_NAMES.get(tgt, tgt))
    (share / "GUIDELINES.md").write_text(guide, encoding="utf-8")
    write_key(out / "key_DO_NOT_SHARE_WITH_ANNOTATORS.csv", items)       # detector flags + references: stays out of for_annotators/
    (out / "kit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for a in annotators:
        ordered = order_for(items, a, int(meta["seed"]))
        write_annotation_csv(share / f"annotator_{a}.csv", ordered)
        (share / f"annotator_{a}.html").write_text(render_html(ordered, cfg.pair_key, a, tgt, guide), encoding="utf-8")
    return out
