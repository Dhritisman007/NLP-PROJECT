"""Gradio demo: enter English text -> MT output -> problem spans highlighted by error type -> corrected translation.

Pure helpers (sentence splitting, span highlighting, diffs, tables) are separated from the Gradio layout so they can be unit
tested without launching a server. The detector was trained on SYNTHETIC errors: the UI says so, and shows probabilities, so
users can judge how much to trust a flag.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable

from mml.config import Config
from mml.correct.span_postedit import describe_span, select_spans
from mml.detect.base import SpanFlag
from mml.detect.classifier import TYPE_NAMES
from mml.detect.spans import kind_to_type
from mml.pipeline import CorrectionPipeline, PipelineResult

log = logging.getLogger(__name__)
# A local demo must not phone home: no analytics, no version check (set before gradio is imported).
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

MAX_SENTENCES = 8
MAX_CHARS = 1200
# When spans of different types overlap, the more specific / more harmful type wins the highlight.
PRIORITY = ("MISTRANSLATION", "ENTITY_NUMBER", "HALLUCINATION", "OMISSION")
# Palette slots 1, 2, 3, 5 of the validated categorical palette, at 45% opacity so text stays readable; every span also carries
# its type as a text label (secondary encoding), and the findings table repeats it in words.
TYPE_COLORS = {"OMISSION": "rgba(42,120,214,0.45)", "MISTRANSLATION": "rgba(235,104,52,0.45)",
               "ENTITY_NUMBER": "rgba(27,175,122,0.45)", "HALLUCINATION": "rgba(232,123,164,0.45)"}
TYPE_TITLES = {"OMISSION": "Omission", "MISTRANSLATION": "Mistranslation", "ENTITY_NUMBER": "Entity / number",
               "HALLUCINATION": "Hallucination", "any error": "Any error"}
EDIT_COLOR = {"edited": "rgba(42,120,214,0.35)"}
EXAMPLES = [
    "The company reported a profit of 4.5 million dollars in the third quarter.",
    "She did not attend the meeting because she was ill.",
    "Prime Minister Narendra Modi visited Germany on 12 May 2022. Thousands of people gathered to celebrate.",
    "The new bridge, which is 1.2 kilometres long, connects the island to the mainland.",
]
INTRO = """**Meaning-loss detection and correction for machine translation** (English → Hindi / German).
Type English text. The app translates it, **highlights** where meaning may be lost (omission, mistranslation, entity/number, hallucination),
and proposes a corrected translation. *Caveats:* the detector was trained on **synthetic** errors (it falsely flags ≈ 8–14% of correct
sentences), and correction without an LLM only fixes numbers, dates, units and known names; other flagged errors are shown but not auto-fixed."""


# --------------------------------------------------------------------------- pure helpers
def split_sentences(text: str, max_sentences: int = MAX_SENTENCES) -> list[str]:
    """Split on sentence punctuation and newlines (the models work sentence by sentence)."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()][:max_sentences]


def highlight_segments(text: str, spans: list[tuple[int, int, str]]) -> list[tuple[str, str | None]]:
    """``[(piece, label | None)]`` covering ``text`` exactly; overlapping spans resolve by ``PRIORITY``."""
    labels: list[str | None] = [None] * len(text)
    rank = {t: i for i, t in enumerate(PRIORITY)}
    for start, end, label in sorted(spans, key=lambda s: -rank.get(s[2], 99)):          # lowest priority first, so higher overwrites
        for i in range(max(0, start), min(len(text), end)):
            labels[i] = label
    out: list[tuple[str, str | None]] = []
    for ch, lab in zip(text, labels):
        if out and out[-1][1] == lab:
            out[-1] = (out[-1][0] + ch, lab)
        else:
            out.append((ch, lab))
    return out or [("", None)]


def diff_segments(old: str, new: str) -> list[tuple[str, str | None]]:
    """Word-level view of ``new`` with inserted/changed words labelled ``edited``."""
    a, b = old.split(), new.split()
    out: list[tuple[str, str | None]] = []
    for op, _, _, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append((" ".join(b[j1:j2]), None))
        elif op in ("replace", "insert"):
            out.append((" ".join(b[j1:j2]), "edited"))
        out.append((" ", None))
    if out and out[-1] == (" ", None):
        out.pop()
    return out or [(new, None)]


def edit_summary(old: str, new: str) -> list[list[str]]:
    """Rows ``[before, after]`` for every changed word group."""
    a, b = old.split(), new.split()
    return [[" ".join(a[i1:i2]) or "(nothing)", " ".join(b[j1:j2]) or "(removed)"]
            for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op != "equal"]


@dataclass
class DemoView:
    """Everything the UI shows for one request (kept free of Gradio types for testing)."""

    source_segments: list[tuple[str, str | None]]
    mt_segments: list[tuple[str, str | None]]
    corrected_segments: list[tuple[str, str | None]]
    corrected_text: str
    probabilities: dict[str, float]
    findings: list[list[str]]
    edits: list[list[str]]
    status: str
    flagged: bool = False
    flagged_types: list[str] = field(default_factory=list)
    per_sentence: list[dict[str, Any]] = field(default_factory=list)


def _typed_spans(res: PipelineResult, side: str) -> list[SpanFlag]:
    return [s for s in select_spans(res.before.spans_for(), res.src, res.mt) if s.side == side]


def build_view(results: list[PipelineResult], strategy: str, skipped: dict[str, str] | None = None, applied: bool = True) -> DemoView:
    """Combine per-sentence pipeline results into one source / translation / corrected view with shifted offsets."""
    src_text = mt_text = corr_text = ""
    src_spans: list[tuple[int, int, str]] = []
    mt_spans: list[tuple[int, int, str]] = []
    findings: list[list[str]] = []
    probs = {t: 0.0 for t in TYPE_NAMES}
    probs_any = 0.0
    for r in results:
        s_off, m_off = len(src_text) + (1 if src_text else 0), len(mt_text) + (1 if mt_text else 0)
        for sp in _typed_spans(r, "src"):
            src_spans.append((s_off + sp.start, s_off + sp.end, kind_to_type(sp.kind) or ""))
        for sp in _typed_spans(r, "tgt"):
            mt_spans.append((m_off + sp.start, m_off + sp.end, kind_to_type(sp.kind) or ""))
        for sp in select_spans(r.before.spans_for(), r.src, r.mt):
            text = (r.src if sp.side == "src" else r.mt)[sp.start:sp.end]
            why = (describe_span(sp, r.src, r.mt) or "").split("] ", 1)[-1]
            findings.append([TYPE_TITLES.get(kind_to_type(sp.kind) or "", ""), "source" if sp.side == "src" else "translation", text, why])
        for t in sorted(r.before.types):
            if not any(f[0] == TYPE_TITLES[t] for f in findings):          # a predicted type with no located span: still tell the user
                findings.append([TYPE_TITLES[t], "sentence", "", "Flagged by the classifier; no specific span located."])
        probs = {t: max(probs[t], r.before.probs.get(t, 0.0)) for t in TYPE_NAMES}
        probs_any = max(probs_any, r.before.probs.get("ANY", 0.0))
        src_text += (" " if src_text else "") + r.src
        mt_text += (" " if mt_text else "") + r.mt
        corr_text += (" " if corr_text else "") + (r.corrected if applied else r.mt)
    flagged = any(r.before.types for r in results)
    changed = corr_text != mt_text
    status = [f"**Strategy:** `{strategy}`. " + ("LLM post-editing available." if strategy in ("full", "span_llm") else
                                                "Rule-based fixing only (numbers, dates, units, known names); no LLM configured.")]
    if not flagged:
        status.append("No meaning-loss problems detected (the detector can miss errors; it is not a guarantee).")
    elif not changed:
        status.append("Problems were flagged but **no automatic fix applied**: this strategy cannot repair omissions, mistranslations "
                      "or hallucinations without an LLM. Review the highlighted spans.")
    else:
        status.append("A correction was applied; changed words are highlighted below.")
    after_flagged = any(r.after.types for r in results if r.changed)
    if changed:
        status.append("After the edit the detector " + ("**still flags** a problem." if after_flagged else "no longer flags the edited sentence(s)."))
    if skipped:
        status.append("Signals not available (skipped): " + ", ".join(sorted(skipped)) + ".")
    return DemoView(
        highlight_segments(src_text, src_spans), highlight_segments(mt_text, mt_spans),
        diff_segments(mt_text, corr_text) if changed else [(corr_text, None)], corr_text,
        {**{t: probs[t] for t in TYPE_NAMES}, "any error": probs_any}, findings, edit_summary(mt_text, corr_text), "\n\n".join(status), flagged,
        sorted({t for r in results for t in r.before.types}),
        [{"src": r.src, "mt": r.mt, "corrected": r.corrected, "types": sorted(r.before.types)} for r in results])


TITLE_COLORS = {TYPE_TITLES[t]: c for t, c in TYPE_COLORS.items()}


def retitle(segments: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
    """Show readable labels ("Entity / number") on highlighted spans, matching the legend (``edited`` is left as is)."""
    return [(text, TYPE_TITLES.get(label, label) if label else None) for text, label in segments]


def probability_rows(view: DemoView) -> list[list[str]]:
    """Table rows ``[type, confidence, flagged?]`` (a plain table: a top-class headline would suggest an error even when
    nothing crossed the flagging threshold)."""
    rows = []
    for name, p in sorted(view.probabilities.items(), key=lambda kv: -kv[1]):
        flagged = view.flagged if name == "any error" else name in view.flagged_types
        rows.append([TYPE_TITLES.get(name, name), f"{p:.0%}", "yes" if flagged else "no"])
    return rows


def legend_html() -> str:
    """Colour legend as HTML chips (every chip carries the type name; colour is never the only cue)."""
    chips = "".join(f'<span style="background:{TYPE_COLORS[t]};padding:2px 8px;border-radius:6px;margin-right:8px">{TYPE_TITLES[t]}</span>'
                    for t in PRIORITY[::-1])
    edit = f'<span style="background:{EDIT_COLOR["edited"]};padding:2px 8px;border-radius:6px">edited word</span>'
    return f'<div style="margin:4px 0 10px 0;font-size:0.9rem">Highlight colours: {chips}{edit}</div>'


def validate_request(text: str, own_translation: str = "") -> list[str]:
    """Check the user's input and return the source sentences; raises ``ValueError`` with a readable message.

    Called before any model is loaded, so an empty or invalid request never costs a model load.
    """
    sentences = split_sentences(text[:MAX_CHARS])
    if not sentences:
        raise ValueError("Please enter some English text.")
    if own_translation and own_translation.strip() and len(sentences) != 1:
        raise ValueError("To check your own translation, enter exactly one source sentence.")
    return sentences


def analyse(pipeline: CorrectionPipeline, text: str, apply_correction: bool = True, own_translation: str = "") -> DemoView:
    """Run the pipeline on user text (≤ ``MAX_SENTENCES`` sentences) and build the view.

    With ``own_translation`` (one source sentence only) the MT step is skipped and the given translation is checked and corrected;
    this makes it possible to try the detector/corrector on a deliberately wrong translation.
    """
    sentences = validate_request(text, own_translation)
    if own_translation and own_translation.strip():
        results = pipeline.run_pairs([(sentences[0], own_translation.strip()[:MAX_CHARS])])
    else:
        results = pipeline.run(sentences)
    return build_view(results, pipeline.strategy, pipeline.detector.skipped, apply_correction)


# --------------------------------------------------------------------------- Gradio layout
def build_app(get_pipeline: Callable[[str], CorrectionPipeline], pairs: list[str], default_pair: str) -> Any:
    """Gradio Blocks app. ``get_pipeline(pair)`` returns a (cached) pipeline; models load lazily on first use."""
    import gradio as gr

    lock = threading.Lock()                                    # one translation/detection at a time (single shared model set)

    def run(text: str, pair: str, apply_correction: bool, own_translation: str = "") -> tuple:
        try:
            validate_request(text, own_translation)            # before get_pipeline(): invalid input must not trigger a model load
            with lock:
                view = analyse(get_pipeline(pair), text, apply_correction, own_translation)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - show a readable error instead of a stack trace in the browser
            log.exception("demo request failed")
            raise gr.Error(f"{type(exc).__name__}: {exc}") from exc
        return (retitle(view.source_segments), retitle(view.mt_segments), probability_rows(view), view.findings or [["", "", "", "No problems detected."]],
                view.corrected_segments, view.edits or [["", ""]], view.status)

    # Gradio's HighlightedText uses `word-break: break-all`, which splits English words mid-word; wrap at word boundaries instead.
    css = ".textfield, .textfield * { word-break: normal !important; overflow-wrap: anywhere; }"
    with gr.Blocks(title="MT meaning-loss detector and corrector", css=css) as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            text = gr.Textbox(label="English source text", lines=4, placeholder="Type or paste English text (up to 8 short sentences)",
                              scale=4)
            with gr.Column(scale=1):
                pair = gr.Dropdown(choices=pairs, value=default_pair, label="Language pair")
                apply = gr.Checkbox(value=True, label="Apply correction")
                go = gr.Button("Translate and check", variant="primary")
        own = gr.Textbox(label="Optional: paste your own translation to check instead of machine-translating (one source sentence only)",
                         lines=2, placeholder="e.g. Die Firma verdiente 45 Millionen Dollar im dritten Quartal.")
        gr.Examples(EXAMPLES, inputs=text, label="Examples")
        gr.HTML(legend_html())
        with gr.Row():
            src_hl = gr.HighlightedText(label="Source: evidence of missing content", color_map=TITLE_COLORS, show_legend=False, combine_adjacent=False)
            mt_hl = gr.HighlightedText(label="Machine translation: suspicious spans", color_map=TITLE_COLORS, show_legend=False, combine_adjacent=False)
        with gr.Row():
            probs = gr.Dataframe(headers=["error type", "confidence (max over sentences)", "flagged"], label="Detector", interactive=False, wrap=True, column_widths=["34%", "40%", "26%"])
            status = gr.Markdown()
        findings = gr.Dataframe(headers=["type", "where", "text", "why it was flagged"], label="Findings", wrap=True, interactive=False,
                                column_widths=["17%", "13%", "14%", "56%"])
        corr = gr.HighlightedText(label="Corrected translation (edited words highlighted)", color_map=EDIT_COLOR, show_legend=False, combine_adjacent=False)
        edits = gr.Dataframe(headers=["before", "after"], label="Edits made", wrap=True, interactive=False)
        outputs = [src_hl, mt_hl, probs, findings, corr, edits, status]
        go.click(run, [text, pair, apply, own], outputs)
        text.submit(run, [text, pair, apply, own], outputs)
    return demo


def launch(cfg: Config, pairs: list[str], default_pair: str, port: int = 7860, share: bool = False, strategy: str = "auto") -> None:
    """Start the demo on localhost (not exposed to the network unless ``share``)."""
    from mml.resident import build_resident_pipeline

    cache: dict[str, CorrectionPipeline] = {}

    def get_pipeline(pair: str) -> CorrectionPipeline:
        if pair not in cache:
            from mml.config import load_config
            from mml.utils import free_memory

            if cache:                                          # one language pair resident at a time: an 8 GB laptop cannot hold two
                log.info("Releasing models of %s before loading %s", sorted(cache), pair)
                cache.clear()
                free_memory()

            pair_cfg = load_config(pair=pair, profile=cfg.raw["run"]["profile"])
            log.info("Loading models for %s (first request only)...", pair)
            cache[pair] = build_resident_pipeline(pair_cfg, strategy)
        return cache[pair]

    app = build_app(get_pipeline, pairs, default_pair)
    app.queue().launch(server_name="127.0.0.1", server_port=port, share=share, inbrowser=False)
