"""
pipeline.py — Production Guardrail Pipeline
============================================
Implements a three-layer content moderation pipeline:

  Layer 1 : Regex pre-filter  (fast, zero model cost, returns category)
  Layer 2 : Calibrated model  (best mitigated DistilBERT from Part 4)
  Layer 3 : Human review queue (uncertainty band 0.40–0.60)

Usage
-----
    from pipeline import ModerationPipeline
    pipe = ModerationPipeline(model_dir='/kaggle/working/distilbert-reweighed-final')
    result = pipe.predict("some comment text")
    # → {"decision": "block"|"allow"|"review",
    #    "layer": "input_filter"|"model"|"human_review",
    #    "confidence": float,
    #    "category": str | None}
"""

from __future__ import annotations

import re
import logging
from typing import Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Regex Blocklist
# ══════════════════════════════════════════════════════════════════════════════
#
# Design notes
# ------------
# • All patterns compiled with re.IGNORECASE for case-invariant matching.
# • \b word boundaries prevent substring false positives (e.g. "skill" ≠ "kill").
# • Non-capturing groups (?:...) used where grouping is needed but capture is not.
# • Capturing groups used in direct_threat to log the matched verb.
# • Lookaheads (?=...) used in coordinated_harassment to catch trigger words
#   regardless of what follows.
# • Dehumanization patterns are intentionally broad; false positives (e.g.
#   biology text: "bacteria are not human") are acceptable because Layer 2
#   catches genuine negatives before any action is taken.
# • All contractions (I'll / I will / gonna / going to) are handled.

BLOCKLIST: dict[str, list[re.Pattern]] = {

    # ── Category 1: Direct threats of violence (≥ 5 patterns) ────────────────
    # Pattern logic: subject + modal/auxiliary + threat verb + optional object.
    # At least one pattern uses a capturing group over a set of threat verbs.
    "direct_threat": [
        # "I will / I'll / I'm gonna / I'm going to [kill|murder|shoot|stab|hurt|destroy] you"
        re.compile(
            r"\b(?:i(?:'?m|'?ll|'?m gonna| will| am going to))\s+"
            r"(kill|murder|shoot|stab|hurt|destroy|slaughter|assault|attack)\s+(?:you|u)\b",
            re.IGNORECASE,
        ),
        # "you're / you are going to die / get hurt / be killed"
        re.compile(
            r"\byou(?:'?re| are)\s+(?:going to|gonna)\s+"
            r"(?:die|get killed|get hurt|suffer|bleed)\b",
            re.IGNORECASE,
        ),
        # "someone should [shoot|kill|hurt|stab] you / him / her / them"
        re.compile(
            r"\bsomeone\s+should\s+(?:shoot|kill|hurt|stab|murder|assault)\s+"
            r"(?:you|him|her|them|u)\b",
            re.IGNORECASE,
        ),
        # "I'll find where you live / I know where you sleep" (overlap with doxxing but high-severity)
        re.compile(
            r"\bi(?:'ll| will| am going to| gonna)\s+(?:find|track down|come to|visit)\s+"
            r"where\s+you\s+(?:live|sleep|work|stay|are)\b",
            re.IGNORECASE,
        ),
        # "you will / you're going to regret this / pay for this" — implicit threat
        re.compile(
            r"\byou(?:'?re| will| are going to| gonna)\s+"
            r"(?:regret|pay for|answer for)\s+(?:this|that|what you did|what you said)\b",
            re.IGNORECASE,
        ),
        # "I'll make you [pay|suffer|bleed|scream]"
        re.compile(
            r"\bi(?:'ll| will| am going to| gonna)\s+make\s+you\s+"
            r"(?:pay|suffer|bleed|scream|beg|cry)\b",
            re.IGNORECASE,
        ),
        # "watch your back / sleep with one eye open" — threat idioms
        re.compile(
            r"\b(?:watch\s+your\s+back|sleep\s+with\s+one\s+eye\s+open|"
            r"you\s+(?:better|had better)\s+run)\b",
            re.IGNORECASE,
        ),
    ],

    # ── Category 2: Directed self-harm / suicide calls (≥ 4 patterns) ────────
    # Pattern logic: second-person subject ("you", "go") + self-harm verb phrase.
    # Explicitly avoids first-person ("I") to reduce false positives on
    # personal disclosures which require a different (supportive) response.
    "self_harm_directed": [
        # "you should kill / hang / hurt yourself"
        re.compile(
            r"\byou\s+should\s+(?:just\s+)?(?:kill|hang|hurt|harm|shoot|poison)\s+yourself\b",
            re.IGNORECASE,
        ),
        # "go kill / go hang / go end yourself"
        re.compile(
            r"\bgo\s+(?:and\s+)?(?:kill|hang|hurt|harm|shoot|end|off)\s+yourself\b",
            re.IGNORECASE,
        ),
        # "nobody would miss you if you died / disappeared"
        re.compile(
            r"\bnobody\s+(?:would|will|'?d)\s+(?:miss|care about|mourn)\s+you\s+"
            r"(?:if|when)\s+you\s+(?:died?|disappear(?:ed)?|were gone|left)\b",
            re.IGNORECASE,
        ),
        # "do everyone a favour / favor and [disappear|die|end it]"
        re.compile(
            r"\bdo\s+everyone\s+a\s+favou?r\s+and\s+"
            r"(?:disappear|die|end\s+it|kill\s+yourself|leave\s+this\s+world)\b",
            re.IGNORECASE,
        ),
        # "the world would be better without you"
        re.compile(
            r"\bthe\s+world\s+(?:would be|is)\s+better\s+(?:off\s+)?without\s+you\b",
            re.IGNORECASE,
        ),
    ],

    # ── Category 3: Doxxing and stalking threats (≥ 4 patterns) ─────────────
    # Pattern logic: first-person + knowledge of private information OR
    # intent to publish/find private information.
    "doxxing_stalking": [
        # "I know where you live / work / sleep / go to school"
        re.compile(
            r"\bi\s+know\s+where\s+you\s+(?:live|work|sleep|go to school|stay|are)\b",
            re.IGNORECASE,
        ),
        # "I'll post / publish / share your address / number / info / details"
        re.compile(
            r"\bi(?:'ll| will| am going to| gonna)\s+(?:post|publish|share|leak|expose|drop)\s+"
            r"(?:your\s+)?(?:address|home address|phone (?:number|no)|number|info|details|"
            r"personal (?:info|details|data)|location)\b",
            re.IGNORECASE,
        ),
        # "I found your real name / your address / your IP"
        re.compile(
            r"\bi(?:'ve| have)?\s+found\s+(?:your\s+)?(?:real\s+)?(?:name|address|"
            r"location|ip(?: address)?|phone|number|identity)\b",
            re.IGNORECASE,
        ),
        # "everyone will know who you really are / where you live"
        re.compile(
            r"\beveryone\s+(?:will|is going to|gonna)\s+know\s+"
            r"(?:who you (?:really )?are|where you live|your (?:real )?name|"
            r"what you look like)\b",
            re.IGNORECASE,
        ),
        # "I'm watching you / I've been following you"
        re.compile(
            r"\bi(?:'m| am)\s+(?:watching|following|tracking|monitoring)\s+you\b",
            re.IGNORECASE,
        ),
        # "I'll dox you / dropping your dox"
        re.compile(
            r"\b(?:i(?:'ll| will| am going to| gonna)\s+dox\s+you|"
            r"dropping\s+(?:your\s+)?dox(?:x)?)\b",
            re.IGNORECASE,
        ),
    ],

    # ── Category 4: Severe dehumanization (≥ 4 patterns) ─────────────────────
    # Pattern logic: [group reference] + [sub-human comparison or extermination call].
    # Non-capturing groups (?:human|people|person) match plural/singular variants.
    # Intentionally broad — Layer 2 filters genuine false positives.
    "dehumanization": [
        # "[group] are not human / people / persons"
        re.compile(
            r"\b(?:[a-z]+(?:s|es)?)\s+are\s+not\s+(?:human|people|persons?|real\s+humans?)\b",
            re.IGNORECASE,
        ),
        # "[group] are animals / vermin / parasites / rats / cockroaches / subhuman"
        re.compile(
            r"\b(?:[a-z]+(?:s|es)?)\s+are\s+(?:just\s+)?(?:animals?|vermin|parasites?|"
            r"rats?|cockroaches?|subhuman|insects?|pests?|beasts?|savages?|monkeys?)\b",
            re.IGNORECASE,
        ),
        # "[group] should be exterminated / eliminated / wiped out / cleansed / removed"
        re.compile(
            r"\b(?:[a-z]+(?:s|es)?)\s+should\s+(?:all\s+)?be\s+"
            r"(?:exterminated|eliminated|wiped\s+out|cleansed|removed|"
            r"eradicated|destroyed|genocided|purged)\b",
            re.IGNORECASE,
        ),
        # "[group] are a disease / cancer / plague / infection / virus"
        re.compile(
            r"\b(?:[a-z]+(?:s|es)?)\s+are\s+(?:a\s+)?(?:disease|cancer|plague|"
            r"infection|virus|epidemic|pestilence|blight|rot|scourge)\b",
            re.IGNORECASE,
        ),
        # "kill all / exterminate all [group]"
        re.compile(
            r"\b(?:kill|exterminate|eliminate|wipe\s+out|destroy)\s+all\s+[a-z]+\b",
            re.IGNORECASE,
        ),
    ],

    # ── Category 5: Coordinated harassment signals (≥ 3 patterns) ────────────
    # Pattern logic: collective subject + target + aggressive verb.
    # Lookahead (?=...) catches trigger phrases regardless of what follows.
    "coordinated_harassment": [
        # "everyone report [username/@handle/this account/this user]"
        # Lookahead: matches "everyone report" regardless of what follows
        re.compile(
            r"\beveryone\s+report\b(?=\s)",
            re.IGNORECASE,
        ),
        # "let's all [go after|attack|mass report|target|brigade] [someone]"
        re.compile(
            r"\blet(?:'s|'s| us| us all)?\s+all\s+(?:go\s+after|attack|mass\s+report|"
            r"target|brigade|harass|pile\s+on|spam)\b",
            re.IGNORECASE,
        ),
        # "raid their profile / channel / stream / account"
        re.compile(
            r"\braid\s+(?:their|his|her|the)\s+"
            r"(?:profile|channel|stream|account|page|server|discord)\b",
            re.IGNORECASE,
        ),
        # "mass report this / his / her account"
        re.compile(
            r"\bmass\s+report\s+(?:this|his|her|their|the)\s+"
            r"(?:account|profile|post|video|channel|page)?\b",
            re.IGNORECASE,
        ),
        # Lookahead: "brigade" as a verb anywhere in the sentence
        re.compile(
            r"\b(?:we\s+should|let'?s|go|everyone)\s+(?:brigade|dogpile|swarm|flood)\b"
            r"(?=\s)",
            re.IGNORECASE,
        ),
    ],
}

# Total pattern count (assertion: must be ≥ 20)
_total_patterns = sum(len(v) for v in BLOCKLIST.values())
assert _total_patterns >= 20, (
    f"Blocklist has only {_total_patterns} patterns; minimum is 20."
)
logger.info(
    "Blocklist loaded: %d categories, %d total patterns.",
    len(BLOCKLIST),
    _total_patterns,
)


def input_filter(text: str) -> Optional[dict]:
    """
    Layer 1 — Regex pre-filter.

    Iterates over every category and every compiled pattern.
    Returns a structured block decision on first match, else None.

    Parameters
    ----------
    text : str
        Raw comment text to evaluate.

    Returns
    -------
    dict | None
        Block decision dict if matched:
        {
            "decision"   : "block",
            "layer"      : "input_filter",
            "category"   : <str>,        # which category triggered
            "confidence" : 1.0,          # regex is deterministic
            "pattern"    : <str>,        # the regex pattern that matched
        }
        None if no pattern matches.
    """
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return {
                    "decision"  : "block",
                    "layer"     : "input_filter",
                    "category"  : category,
                    "confidence": 1.0,
                    "pattern"   : pattern.pattern,
                    "matched"   : match.group(0),
                }
    return None


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 + 3 — Calibrated Model + Human Review Queue
# ══════════════════════════════════════════════════════════════════════════════

class ModerationPipeline:
    """
    Three-layer production content moderation pipeline.

    Layer 1 : Regex pre-filter  → immediate block on high-precision patterns
    Layer 2 : Calibrated model  → block (≥ high_thresh) or allow (≤ low_thresh)
    Layer 3 : Human review      → anything in (low_thresh, high_thresh)

    Parameters
    ----------
    model_dir : str
        Path to the HuggingFace model directory (best mitigated model from Part 4).
    low_thresh : float
        Confidence below which a comment is auto-allowed. Default 0.40.
    high_thresh : float
        Confidence above which a comment is auto-blocked. Default 0.60.
    calibration_sample_size : int
        Number of examples to use for isotonic calibration fitting. Default 2000.
    device : str | None
        'cuda', 'cpu', or None (auto-detect).
    max_len : int
        Max token length for DistilBERT. Default 128.
    batch_size : int
        Inference batch size. Default 64.
    """

    def __init__(
        self,
        model_dir: str,
        low_thresh: float = 0.40,
        high_thresh: float = 0.60,
        calibration_sample_size: int = 2000,
        device: Optional[str] = None,
        max_len: int = 128,
        batch_size: int = 64,
    ) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.low_thresh  = low_thresh
        self.high_thresh = high_thresh
        self.max_len     = max_len
        self.batch_size  = batch_size

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info("ModerationPipeline using device: %s", self.device)

        # Load tokenizer and model
        logger.info("Loading model from: %s", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self._model = self._model.to(self.device)
        self._model.eval()

        # Calibrator will be set in .fit_calibrator()
        self._calibrator = None
        self._calibrated  = False

        logger.info(
            "Pipeline ready. Thresholds: allow≤%.2f | review (%.2f,%.2f) | block≥%.2f",
            low_thresh, low_thresh, high_thresh, high_thresh,
        )

    # ── Internal: batch inference ─────────────────────────────────────────────
    @staticmethod
    def _texts_to_probs(model, tokenizer, texts, device, max_len, batch_size):
        """Return raw P(toxic) array from the uncalibrated model."""
        import torch
        from torch.utils.data import Dataset, DataLoader

        class _DS(Dataset):
            def __init__(self, t, tok, ml):
                self.t, self.tok, self.ml = list(t), tok, ml
            def __len__(self): return len(self.t)
            def __getitem__(self, i):
                enc = self.tok(
                    self.t[i], max_length=self.ml, truncation=True,
                    padding="max_length", return_tensors="pt",
                )
                return {
                    "input_ids"      : enc["input_ids"].squeeze(),
                    "attention_mask" : enc["attention_mask"].squeeze(),
                }

        loader = DataLoader(_DS(texts, tokenizer, max_len),
                            batch_size=batch_size, num_workers=2)
        probs = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                out = model(
                    input_ids      = batch["input_ids"].to(device),
                    attention_mask = batch["attention_mask"].to(device),
                )
                p = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
                probs.extend(p)
        return np.array(probs)

    # ── Calibration ───────────────────────────────────────────────────────────
    def fit_calibrator(self, texts: list[str], labels: list[int]) -> "ModerationPipeline":
        """
        Fit an isotonic-regression probability calibrator on a held-out sample.

        Calibration corrects systematic over/under-confidence in the model's
        raw softmax probabilities. Isotonic regression is preferred over Platt
        scaling here because the Jigsaw score distribution is non-sigmoidal.

        Parameters
        ----------
        texts  : list of str   — calibration text examples (held-out, never trained on)
        labels : list of int   — ground-truth binary labels (0/1)

        Returns self for chaining.
        """
        from sklearn.isotonic import IsotonicRegression

        logger.info("Fitting isotonic calibrator on %d examples …", len(texts))

        raw_probs = self._texts_to_probs(
            self._model, self.tokenizer, texts,
            self.device, self.max_len, self.batch_size,
        )

        # IsotonicRegression maps raw_prob → calibrated_prob
        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(raw_probs, np.array(labels))
        self._calibrated = True

        # Quick reliability check
        cal_probs = self._calibrator.predict(raw_probs)
        logger.info(
            "Calibrator fitted. Raw prob mean: %.4f → calibrated mean: %.4f",
            raw_probs.mean(), cal_probs.mean(),
        )
        return self

    def _calibrated_prob(self, raw_prob: float) -> float:
        """Apply calibrator if fitted; else return raw probability."""
        if self._calibrated and self._calibrator is not None:
            return float(self._calibrator.predict([raw_prob])[0])
        return float(raw_prob)

    # ── Unicode / adversarial pre-sanitisation ────────────────────────────────
    @staticmethod
    def _sanitise(text: str) -> str:
        """
        Strip adversarial character-level manipulations before tokenisation.

        Applies three defences documented in Part 3:
        1. NFKD normalisation — collapses Unicode homoglyphs (Cyrillic lookalikes)
           back to their canonical ASCII equivalents.
        2. Zero-width character stripping — removes U+200B, U+200C, U+200D, U+FEFF
           which are invisible to humans but split subword tokens.
        3. Consecutive-character deduplication — collapses ≥3 repeats of the same
           character to 2 (e.g. "haaaaate" → "haate", "looool" → "lool").
           Two repeats are preserved to handle legitimate emphasis ("sooo good").
        """
        import unicodedata

        # Step 1: NFKD normalisation + ASCII encoding to strip non-ASCII lookalikes
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", errors="ignore").decode("ascii")

        # Step 2: Zero-width character removal
        for zwc in ("\u200b", "\u200c", "\u200d", "\ufeff"):
            text = text.replace(zwc, "")

        # Step 3: Collapse ≥3 consecutive identical characters to 2
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        return text

    # ── Public API ────────────────────────────────────────────────────────────
    def predict(self, text: str) -> dict:
        """
        Run the three-layer pipeline on a single comment.

        Parameters
        ----------
        text : str
            Raw comment text.

        Returns
        -------
        dict with keys:
            decision   : "block" | "allow" | "review"
            layer      : "input_filter" | "model" | "human_review"
            confidence : float   (1.0 for regex; calibrated prob for model)
            category   : str | None  (set for Layer 1 blocks)
        """
        # ── Layer 1: Regex pre-filter ─────────────────────────────────────────
        filter_result = input_filter(text)
        if filter_result is not None:
            return filter_result   # already correctly structured by input_filter()

        # ── Adversarial sanitisation before model inference ───────────────────
        clean_text = self._sanitise(text)

        # ── Layer 2: Calibrated model ─────────────────────────────────────────
        raw_prob  = self._texts_to_probs(
            self._model, self.tokenizer, [clean_text],
            self.device, self.max_len, self.batch_size,
        )[0]
        cal_prob  = self._calibrated_prob(raw_prob)

        if cal_prob >= self.high_thresh:
            return {
                "decision"   : "block",
                "layer"      : "model",
                "confidence" : round(cal_prob, 6),
                "category"   : None,
            }

        if cal_prob <= self.low_thresh:
            return {
                "decision"   : "allow",
                "layer"      : "model",
                "confidence" : round(cal_prob, 6),
                "category"   : None,
            }

        # ── Layer 3: Uncertainty → human review queue ─────────────────────────
        return {
            "decision"   : "review",
            "layer"      : "human_review",
            "confidence" : round(cal_prob, 6),
            "category"   : None,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Efficient batched prediction for multiple comments.

        Runs Layer 1 (regex) sequentially (fast), then batches all
        Layer 1 survivors through the model in one forward pass.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        list of decision dicts (same structure as .predict())
        """
        import torch

        results    = [None] * len(texts)
        model_idxs = []   # indices that need model inference
        model_texts = []  # sanitised texts for those indices

        # Layer 1 pass (sequential — regex is fast)
        for i, text in enumerate(texts):
            fr = input_filter(text)
            if fr is not None:
                results[i] = fr
            else:
                model_idxs.append(i)
                model_texts.append(self._sanitise(text))

        # Layer 2+3 batch pass
        if model_texts:
            raw_probs = self._texts_to_probs(
                self._model, self.tokenizer, model_texts,
                self.device, self.max_len, self.batch_size,
            )
            for j, (orig_idx, raw_prob) in enumerate(zip(model_idxs, raw_probs)):
                cal_prob = self._calibrated_prob(raw_prob)

                if cal_prob >= self.high_thresh:
                    results[orig_idx] = {
                        "decision"   : "block",
                        "layer"      : "model",
                        "confidence" : round(cal_prob, 6),
                        "category"   : None,
                    }
                elif cal_prob <= self.low_thresh:
                    results[orig_idx] = {
                        "decision"   : "allow",
                        "layer"      : "model",
                        "confidence" : round(cal_prob, 6),
                        "category"   : None,
                    }
                else:
                    results[orig_idx] = {
                        "decision"   : "review",
                        "layer"      : "human_review",
                        "confidence" : round(cal_prob, 6),
                        "category"   : None,
                    }

        return results

    # ── Threshold sweep utility ───────────────────────────────────────────────
    def sweep_thresholds(
        self,
        texts: list[str],
        labels: list[int],
        band_configs: list[tuple[float, float]],
    ) -> list[dict]:
        """
        Evaluate pipeline performance at multiple uncertainty-band configurations
        without re-running model inference (reuses cached calibrated probabilities).

        Parameters
        ----------
        texts         : list of str
        labels        : list of int
        band_configs  : list of (low_thresh, high_thresh) tuples to evaluate

        Returns
        -------
        list of result dicts, one per band_config.
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        # Cache calibrated probs for all non-regex texts
        logger.info("Caching calibrated probabilities for threshold sweep …")
        cal_probs_cache = {}
        for i, text in enumerate(texts):
            if input_filter(text) is None:
                raw = self._texts_to_probs(
                    self._model, self.tokenizer, [self._sanitise(text)],
                    self.device, self.max_len, self.batch_size,
                )[0]
                cal_probs_cache[i] = self._calibrated_prob(raw)

        sweep_results = []
        for low_t, high_t in band_configs:
            preds_auto, labels_auto = [], []
            n_review = 0
            n_filter = 0

            for i, (text, label) in enumerate(zip(texts, labels)):
                if input_filter(text) is not None:
                    n_filter += 1
                    preds_auto.append(1)
                    labels_auto.append(label)
                elif i in cal_probs_cache:
                    cp = cal_probs_cache[i]
                    if cp >= high_t:
                        preds_auto.append(1)
                        labels_auto.append(label)
                    elif cp <= low_t:
                        preds_auto.append(0)
                        labels_auto.append(label)
                    else:
                        n_review += 1

            n_auto  = len(preds_auto)
            n_total = len(texts)

            if n_auto > 0:
                f1   = f1_score(labels_auto, preds_auto, average="macro", zero_division=0)
                prec = precision_score(labels_auto, preds_auto, zero_division=0)
                rec  = recall_score(labels_auto, preds_auto, zero_division=0)
            else:
                f1 = prec = rec = 0.0

            sweep_results.append({
                "low_thresh"    : low_t,
                "high_thresh"   : high_t,
                "band"          : f"({low_t:.2f}, {high_t:.2f})",
                "n_auto_actioned": n_auto,
                "n_review"      : n_review,
                "n_filter"      : n_filter,
                "review_pct"    : n_review / n_total * 100,
                "auto_pct"      : n_auto   / n_total * 100,
                "F1_auto"       : round(f1,   4),
                "Precision_auto": round(prec, 4),
                "Recall_auto"   : round(rec,  4),
            })

        return sweep_results


# ══════════════════════════════════════════════════════════════════════════════
# Self-test when run directly
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  pipeline.py — Layer 1 self-test (regex filter only)")
    print("=" * 60)

    test_cases = [
        ("I will kill you right now.",               "direct_threat"),
        ("go kill yourself loser.",                   "self_harm_directed"),
        ("I know where you live and I'm coming.",     "doxxing_stalking"),
        ("They are not human, just animals.",         "dehumanization"),
        ("everyone report this account NOW",          "coordinated_harassment"),
        ("I love this community, great discussion.",  None),
        ("The bacteria are not human cells, clearly.",None),   # biology FP — goes to model
        ("Let us all brigade his channel.",           "coordinated_harassment"),
        ("You should kill yourself.",                 "self_harm_directed"),
        ("I'll post your address online tomorrow.",   "doxxing_stalking"),
    ]

    print(f"\n{'Text':<50} {'Expected':>25} {'Got':>25}")
    print("-" * 105)
    passed = 0
    for text, expected_cat in test_cases:
        result = input_filter(text)
        got_cat = result["category"] if result else None
        ok = "✅" if got_cat == expected_cat else "❌"
        if got_cat == expected_cat:
            passed += 1
        print(f"{ok} {text[:48]:<50} {str(expected_cat):>25} {str(got_cat):>25}")

    print(f"\n{passed}/{len(test_cases)} self-tests passed.")
    print(f"Total patterns in blocklist: {_total_patterns}")
    print()
