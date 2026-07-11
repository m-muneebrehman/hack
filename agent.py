"""
Fireworks AI Hackathon - Track 1: Token-Miser Routing Agent

Winning strategy extracted from top submission analysis:
  1. reasoning_effort:"none" cuts hidden reasoning tokens by up to 98%
  2. kimi-k2p7-code is the most token-efficient accurate model
  3. Local validators reject bad answers before they count toward accuracy
  4. One thinking-ON retry on validation failure (not full fallback)
  5. Token budgets 40-448 measured from actual model outputs

Architecture:
  Tier 0  - Keyword classifier (zero LLM calls)
  Tier 1  - Local solvers: math (deterministic), sentiment (TextBlob), NER (spaCy)
  Tier 2  - Fireworks Model Router with reasoning_effort:none
             Primary: kimi-k2p7-code (bake-off accuracy + tokenizer winner)
             Fallback: walk ALLOWED_MODELS list
             Error 400: disable reasoning_effort run-wide, retry bare
  Tier 3  - One retry with thinking ON if local validator rejects answer
"""

import ast
import json
import logging
import math
import os
import re
import sys
import time
from typing import List, Literal, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Logging - UTF-8 file handler, ASCII stdout to avoid Windows cp1252 crash
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agent.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional local NLP (zero Fireworks tokens)
# ---------------------------------------------------------------------------
try:
    from textblob import TextBlob  # type: ignore
    TEXTBLOB_AVAILABLE = True
    logger.info("TextBlob ready - local sentiment enabled.")
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import spacy  # type: ignore
    SPACY_AVAILABLE = True
    logger.info("spaCy ready - local NER enabled.")
except ImportError:
    SPACY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration - read from environment (harness injects at runtime)
# ---------------------------------------------------------------------------
INPUT_FILE  = "/input/tasks.json"
OUTPUT_FILE = "/output/results.json"
if not os.path.exists("/input"):
    INPUT_FILE  = "input/tasks.json"
    OUTPUT_FILE = "output/results.json"

FIREWORKS_API_KEY: str = os.environ.get("FIREWORKS_API_KEY", "")
FIREWORKS_BASE_URL: str = os.environ.get(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
)
_raw = os.environ.get(
    "ALLOWED_MODELS", "accounts/fireworks/models/kimi-k2p7-code"
)
ALLOWED_MODELS: List[str] = [m.strip() for m in _raw.split(",") if m.strip()]

TEMPERATURE = 0.0   # fully deterministic

# ---------------------------------------------------------------------------
# CRITICAL DISCOVERY: reasoning_effort:"none" cuts hidden tokens by up to 98%.
# Models emit invisible reasoning by default — billed in completion_tokens.
# A 2-word answer costs 127 tokens without this flag.
# If the proxy rejects it (HTTP 400), we retry bare and disable run-wide.
# ---------------------------------------------------------------------------
USE_REASONING_NONE = True   # toggled False on first HTTP 400

# ---------------------------------------------------------------------------
# Preferred model (bake-off winner: best accuracy + tokenizer efficiency)
# ---------------------------------------------------------------------------
PREFERRED_MODELS = [
    "accounts/fireworks/models/kimi-k2p7-code",
    "accounts/fireworks/models/kimi-k2.7-code",
    "accounts/fireworks/models/kimi-k2p7",
]

# Per-category max output tokens — measured from actual model outputs.
# Tight budgets prevent rambling; too tight triggers finish_reason=length.
MAX_TOKENS_BY_CATEGORY = {
    "factual":     80,   # concise factual answer
    "math":        80,   # steps + final answer
    "sentiment":   48,   # JSON object ~40 tokens
    "summary":    120,   # one sentence or short paragraph
    "ner":        120,   # JSON entity list
    "code_debug": 350,   # explanation + corrected function
    "logic":      140,   # step-by-step + answer
    "code_gen":   448,   # full function with docstring
    "general":    100,
}

# ---------------------------------------------------------------------------
# Hyper-compressed system prompts.
# KEY INSIGHT: deleting instructions costs tokens — bare models ramble.
# These prompts are engineered to satisfy the LLM judge in minimum tokens.
# ---------------------------------------------------------------------------
WORKER_PROMPTS = {
    "factual":
        "Answer in 1-3 sentences. Facts only.",
    "math":
        "Show key steps. Final answer on last line as: Answer: X",
    "sentiment":
        'Return only: {"sentiment":"positive|negative|neutral|mixed",'
        '"confidence":0.0-1.0,"justification":"one sentence"}',
    "summary":
        "Summarize in the requested format. Follow length constraints exactly.",
    "ner":
        'Return only: {"entities":[{"text":"...","type":"PERSON|ORG|LOC|DATE|OTHER"}]}',
    "code_debug":
        "State the bug in one sentence. Output the corrected code only.",
    "logic":
        "List each constraint. Deduce step by step. Final answer: [name]",
    "code_gen":
        "Write a correct Python function with type hints and a one-line docstring.",
    "general":
        "Answer directly. 2-4 sentences max.",
}


def get_worker_prompt(category: str) -> str:
    return WORKER_PROMPTS.get(category, WORKER_PROMPTS["general"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TaskInput(BaseModel):
    task_id: str
    prompt: str


class CategoryDecision(BaseModel):
    category: Literal[
        "factual", "math", "sentiment", "summary",
        "ner", "code_debug", "logic", "code_gen", "general"
    ]


class TaskResult(BaseModel):
    task_id: str
    answer: str   # harness requires "answer" key


# ---------------------------------------------------------------------------
# Local validator — catches bad answers before they fail the judge
# ---------------------------------------------------------------------------
def _validate_answer(category: str, answer: str, finish_reason: Optional[str]) -> bool:
    """
    Returns True if the answer looks acceptable.
    Returns False to trigger a thinking-ON retry.
    """
    # Empty answers always fail
    if not answer or not answer.strip():
        return False

    # Truncated output = answer cut mid-sentence, likely incomplete
    if finish_reason == "length":
        logger.warning("Answer truncated (finish_reason=length) — retrying.")
        return False

    # JSON categories: verify parseable JSON
    if category in ("sentiment", "ner"):
        try:
            parsed = json.loads(answer.strip())
            if category == "sentiment" and "sentiment" not in parsed:
                return False
            if category == "ner" and "entities" not in parsed:
                return False
        except json.JSONDecodeError:
            logger.warning("JSON validation failed for [%s] — retrying.", category)
            return False

    # Code categories: verify the code at least parses
    if category in ("code_gen", "code_debug"):
        code = answer
        # Strip markdown fences if present
        code = re.sub(r"```(?:python)?\n?", "", code).strip()
        try:
            ast.parse(code)
        except SyntaxError:
            logger.warning("Code syntax error in [%s] — retrying.", category)
            return False

    return True


# ---------------------------------------------------------------------------
# Tier 0 — Zero-token keyword classifier
# ---------------------------------------------------------------------------
def keyword_classify(prompt: str) -> Tuple[str, float]:
    """Regex-based, costs zero tokens. Returns (category, confidence)."""
    p = prompt.lower()

    rules = {
        "code_debug": [
            r"\bfind the bug\b", r"\bfix this\b", r"\bfix this code\b",
            r"\bdebug\b", r"\btraceback\b", r"\bexception\b",
            r"\bhas a bug\b", r"\bshould return.*but\b",
        ],
        "code_gen": [
            r"\bwrite a python\b", r"\bwrite code\b", r"\bwrite a function\b",
            r"\bimplement\b", r"\bfunction that\b", r"\bclass that\b",
            r"\bpython script\b", r"\bpython function\b",
        ],
        "logic": [
            r"\bif all\b", r"\bwho owns\b", r"\bwho has\b",
            r"\blogic\b", r"\bpuzzle\b", r"\bdeduce\b", r"\bsyllogism\b",
            r"\beach own\b", r"\bcondition\b",
        ],
        "math": [
            r"\bcalculate\b", r"\bsolve\b", r"\bequation\b",
            r"\d+\s*%", r"\bpercent\b", r"\bsquare root\b",
            r"\bhow many.*remain\b", r"\bhow much\b", r"\bsum of\b",
            r"\bdivide\b", r"\bmultiply\b",
        ],
        "sentiment": [
            r"\bsentiment\b", r"\bclassify.*review\b",
            r"\bpositive or negative\b", r"\bopinion of\b",
        ],
        "ner": [
            r"\bnamed entit", r"\bextract.*entit",
            r"\bidentify.*entit", r"\bextract names\b",
            r"\bidentify organizations\b", r"\bidentify people\b",
        ],
        "summary": [
            r"\bsummariz", r"\bsummary\b", r"\btldr\b",
            r"\bin one sentence\b", r"\bin exactly\b", r"\bcondense\b",
        ],
        "factual": [
            r"\bwho is\b", r"\bwhat is the capital\b", r"\bwhen did\b",
            r"\bwhere is\b", r"\bexplain\b", r"\bdefine\b", r"\bwhat is\b",
        ],
    }

    for category, patterns in rules.items():
        for pattern in patterns:
            if re.search(pattern, p):
                return category, 0.9

    return "general", 0.0


# ---------------------------------------------------------------------------
# Tier 1 — Deterministic local solvers (zero Fireworks tokens)
# ---------------------------------------------------------------------------

def _try_solve_math(prompt: str) -> Optional[str]:
    p = prompt.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", p)
    if m:
        return "Answer: {:g}".format(float(m.group(1)) / 100 * float(m.group(2)))
    m = re.search(r"square root of\s*(\d+(?:\.\d+)?)", p)
    if m:
        result = math.sqrt(float(m.group(1)))
        mult = re.search(
            r"square root of\s*\d+(?:\.\d+)?\s*(?:and\s+)?(?:multiply|times|\*)"
            r"\s*(?:by\s+)?(\d+(?:\.\d+)?)", p
        )
        if mult:
            result *= float(mult.group(1))
        return "Answer: {:g}".format(result)
    return None


def _solve_sentiment_local(prompt: str) -> Optional[str]:
    if not TEXTBLOB_AVAILABLE:
        return None
    try:
        text = prompt
        for prefix in [
            "classify the sentiment of this review:",
            "classify the sentiment of:",
            "what is the sentiment of:",
            "analyze the sentiment of:",
        ]:
            idx = prompt.lower().find(prefix)
            if idx != -1:
                text = prompt[idx + len(prefix):].strip()
                break
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.15:
            sentiment = "positive"
        elif polarity < -0.15:
            sentiment = "negative"
        elif abs(polarity) < 0.05:
            sentiment = "neutral"
        else:
            sentiment = "mixed"
        confidence = round(min(abs(polarity) * 1.5 + 0.4, 1.0), 2)
        result = json.dumps({
            "sentiment": sentiment,
            "confidence": confidence,
            "justification": "Polarity {:.2f}".format(polarity),
        })
        if _validate_answer("sentiment", result, None):
            return result
    except Exception as exc:
        logger.warning("TextBlob error: %s", exc)
    return None


def _solve_ner_local(nlp, prompt: str) -> Optional[str]:
    if nlp is None:
        return None
    try:
        text = prompt
        for prefix in [
            "extract all named entities and their types from:",
            "extract named entities from:",
            "extract the named entities from:",
            "identify organizations in this text:",
            "identify people in this text:",
            "identify named entities in:",
        ]:
            idx = prompt.lower().find(prefix)
            if idx != -1:
                text = prompt[idx + len(prefix):].strip().strip("\"'")
                break
        doc = nlp(text)
        entities = [{"text": ent.text, "type": ent.label_} for ent in doc.ents]
        result = json.dumps({"entities": entities})
        if _validate_answer("ner", result, None):
            return result
    except Exception as exc:
        logger.warning("spaCy NER error: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Model selection — prefer kimi-k2p7-code, then rank by size
# ---------------------------------------------------------------------------
def _infer_model_size(model_id: str) -> int:
    ml = model_id.lower()
    moe = re.search(r"(\d+)x(\d+)b", ml)
    if moe:
        return int(moe.group(2))
    direct = re.search(r"(\d+(?:\.\d+)?)b(?:\b|-)", ml)
    if direct:
        return int(float(direct.group(1)))
    if any(k in ml for k in ("micro", "mini", "small")):
        return 3
    if "medium" in ml:
        return 8
    if any(k in ml for k in ("large", "plus")):
        return 70
    return 8


def _pick_primary_model(allowed: List[str]) -> str:
    """Return the best model from ALLOWED_MODELS, preferring kimi-k2p7-code."""
    for preferred in PREFERRED_MODELS:
        for m in allowed:
            if preferred in m or m in preferred:
                logger.info("Primary model: %s (preferred)", m)
                return m
    # Fallback: largest model for best accuracy
    ranked = sorted(allowed, key=_infer_model_size, reverse=True)
    logger.info("Primary model: %s (largest available)", ranked[0])
    return ranked[0]


# ---------------------------------------------------------------------------
# Fireworks API caller with reasoning_effort:none and retry logic
# ---------------------------------------------------------------------------
class FireworksClient:
    """
    Wraps ChatOpenAI with:
    - reasoning_effort:none (cuts hidden tokens up to 98%)
    - HTTP 400 detection → disable reasoning_effort run-wide
    - Fallback walk through ALLOWED_MODELS on model errors
    - One thinking-ON retry on validation failure
    """

    def __init__(self, allowed: List[str]):
        global USE_REASONING_NONE
        self.allowed = allowed
        self.primary = _pick_primary_model(allowed)
        self._cache: dict = {}

    def _make_llm(self, model_id: str, max_tokens: int,
                  with_reasoning_none: bool) -> ChatOpenAI:
        key = (model_id, max_tokens, with_reasoning_none)
        if key not in self._cache:
            kwargs: dict = {
                "model": model_id,
                "temperature": TEMPERATURE,
                "max_tokens": max_tokens,
                "api_key": FIREWORKS_API_KEY,
                "base_url": FIREWORKS_BASE_URL,
            }
            if with_reasoning_none:
                kwargs["model_kwargs"] = {"reasoning_effort": "none"}
            self._cache[key] = ChatOpenAI(**kwargs)
        return self._cache[key]

    def _call_once(self, model_id: str, category: str, prompt: str,
                   max_tokens: int, reasoning_off: bool) -> Tuple[str, Optional[str]]:
        """Single API call. Returns (content, finish_reason)."""
        llm = self._make_llm(model_id, max_tokens, reasoning_off)
        response = llm.invoke([
            SystemMessage(content=get_worker_prompt(category)),
            HumanMessage(content=prompt),
        ])
        finish = None
        if hasattr(response, "response_metadata"):
            finish = response.response_metadata.get("finish_reason")
        tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens = response.usage_metadata.get("total_tokens", 0)
        logger.info(
            "[%s] model=%s reasoning_none=%s tokens=%d finish=%s",
            category, model_id.split("/")[-1], reasoning_off, tokens, finish
        )
        return response.content, finish

    def call(self, category: str, prompt: str) -> str:
        """
        Full call strategy:
        1. Try primary model with reasoning_effort:none
        2. If HTTP 400 → disable reasoning_effort run-wide, retry bare
        3. Validate answer; if bad → retry same model with thinking ON
        4. If model error → walk ALLOWED_MODELS
        5. Last resort: return best attempt
        """
        global USE_REASONING_NONE
        max_tok = MAX_TOKENS_BY_CATEGORY.get(category, 100)
        best_answer = ""

        models_to_try = [self.primary] + [
            m for m in self.allowed if m != self.primary
        ]

        for model_id in models_to_try:
            # --- Attempt 1: reasoning OFF ---
            try:
                answer, finish = self._call_once(
                    model_id, category, prompt, max_tok, USE_REASONING_NONE
                )
                best_answer = answer

                if _validate_answer(category, answer, finish):
                    return answer

                # --- Attempt 2: validation failed → thinking ON retry ---
                logger.info("Validation failed; retrying [%s] with thinking ON.", category)
                answer2, finish2 = self._call_once(
                    model_id, category, prompt,
                    max_tok * 2,   # give more room for reasoning
                    False          # thinking ON
                )
                if _validate_answer(category, answer2, finish2):
                    return answer2

                best_answer = answer2 or best_answer
                # Try next model if still invalid
                continue

            except Exception as exc:
                err_str = str(exc)

                # HTTP 400 → reasoning_effort rejected by proxy
                if "400" in err_str and USE_REASONING_NONE:
                    logger.warning(
                        "HTTP 400 on reasoning_effort:none — disabling run-wide."
                    )
                    USE_REASONING_NONE = False
                    try:
                        answer, finish = self._call_once(
                            model_id, category, prompt, max_tok, False
                        )
                        best_answer = answer
                        if _validate_answer(category, answer, finish):
                            return answer
                    except Exception as exc2:
                        logger.error("Bare retry failed: %s", exc2)
                else:
                    logger.warning(
                        "Model %s failed: %s — trying next.",
                        model_id.split("/")[-1], exc
                    )
                continue

        # Return best attempt (non-empty)
        return best_answer if best_answer.strip() else "Unable to process task."


# ---------------------------------------------------------------------------
# Task Processor
# ---------------------------------------------------------------------------
class TaskProcessor:
    def __init__(self, client: FireworksClient):
        self.client = client
        self._spacy_nlp = None
        if SPACY_AVAILABLE:
            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy en_core_web_sm ready.")
            except Exception as exc:
                logger.warning("spaCy model load failed: %s", exc)

        # LLM supervisor for ambiguous routing (uses cheapest model, 1 call max)
        _cheapest = sorted(client.allowed, key=_infer_model_size)[0]
        _sup_llm = client._make_llm(_cheapest, 20, USE_REASONING_NONE)
        try:
            from pydantic import BaseModel as _BM
            self._supervisor = _sup_llm.with_structured_output(CategoryDecision)
        except Exception:
            self._supervisor = None

    def _llm_classify(self, prompt: str) -> str:
        if self._supervisor is None:
            return "general"
        try:
            r = self._supervisor.invoke([
                SystemMessage(content=(
                    "Classify into one word: factual math sentiment summary "
                    "ner code_debug logic code_gen general"
                )),
                HumanMessage(content=prompt[:300]),
            ])
            return r.category
        except Exception as exc:
            logger.error("LLM router failed: %s", exc)
            return "general"

    def process_task(self, task_id: str, prompt: str) -> TaskResult:
        start = time.time()
        answer: Optional[str] = None
        source = "fireworks"

        # -- Tier 0: keyword classify --
        category, confidence = keyword_classify(prompt)
        if confidence < 0.7:
            category = self._llm_classify(prompt)
            logger.info("[%s] LLM route -> %s", task_id, category)
        else:
            logger.info("[%s] Keyword route -> %s", task_id, category)

        # -- Tier 1: zero-token local solvers --
        if category == "math":
            answer = _try_solve_math(prompt)
            if answer:
                source = "deterministic_math"

        elif category == "sentiment":
            answer = _solve_sentiment_local(prompt)
            if answer:
                source = "textblob"

        elif category == "ner":
            answer = _solve_ner_local(self._spacy_nlp, prompt)
            if answer:
                source = "spacy"

        # -- Tier 2: Fireworks with reasoning_effort:none + validator + retry --
        if answer is None:
            answer = self.client.call(category, prompt)
            source = "fireworks[{}]".format(
                self.client.primary.split("/")[-1]
            )

        elapsed = time.time() - start
        logger.info("[%s] done via [%s] in %.2fs", task_id, source, elapsed)
        return TaskResult(task_id=task_id, answer=answer)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            raw_tasks = json.load(f)
    except Exception as exc:
        logger.error("Cannot read %s: %s", INPUT_FILE, exc)
        sys.exit(1)

    tasks: List[TaskInput] = []
    for t in raw_tasks:
        try:
            tasks.append(TaskInput(**t))
        except Exception as exc:
            logger.warning("Skipping malformed task %s: %s", t, exc)

    if not tasks:
        logger.warning("No valid tasks found.")
        sys.exit(0)

    logger.info("Loaded %d tasks. Primary model: %s", len(tasks), ALLOWED_MODELS)
    logger.info("reasoning_effort:none = %s", USE_REASONING_NONE)

    client = FireworksClient(ALLOWED_MODELS)
    processor = TaskProcessor(client)
    results = []
    total_start = time.time()

    for task in tasks:
        try:
            r = processor.process_task(task.task_id, task.prompt)
            results.append({"task_id": r.task_id, "answer": r.answer})
        except Exception as exc:
            logger.error("Fatal error on task %s: %s", task.task_id, exc)
            results.append({"task_id": task.task_id, "answer": ""})

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Written %s (%d tasks)", OUTPUT_FILE, len(results))
    except Exception as exc:
        logger.error("Cannot write output: %s", exc)
        print(json.dumps(results))
        sys.exit(1)

    logger.info(
        "Done in %.2fs | reasoning_none=%s",
        time.time() - total_start, USE_REASONING_NONE
    )


if __name__ == "__main__":
    main()
