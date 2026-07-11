```markdown
# Implementation Plan: Track 1 Hybrid Token-Efficient Routing Agent (v3)
### For: Coding Agent Handoff
### Goal: Upgrade current system from "deterministic-solvers + single-model-fallback" to "deterministic-solvers → local-LLM-with-validation → per-category ranked Fireworks routing with reasoning suppression → bounded escalation"

---

## 0. Context (read this first)

### What this system does
This is a submission for the AMD Developer Hackathon Track 1 ("Hybrid Token-Efficient Routing Agent"). The harness feeds a fixed set of tasks via `/input/tasks.json`, spanning 8 categories (factual QA, math reasoning, sentiment analysis, summarization, named-entity recognition, code debugging, logical reasoning, code generation). The agent must write answers to `/output/results.json`. It is scored on:
1. **Accuracy gate** — must clear a minimum accuracy threshold (LLM-judged and/or exact-match depending on category).
2. **Token count** — ranked by total tokens consumed, but **only tokens routed through `FIREWORKS_BASE_URL` count**. Local/in-container inference costs zero score-relevant tokens.

### Hard constraints (grading environment)
- **4 GB RAM, 2 vCPU** — a 7B 4-bit model fills the entire RAM budget; only 2B–3B 4-bit quantized models are safe.
- **No Ollama or model runtime pre-installed** — model weights must be bundled directly in the Docker image.
- **Docker image size limit: 10 GB compressed.**
- **`linux/amd64` platform required.**
- Failure modes to avoid (from participant guide): `PULL_ERROR`, `RUNTIME_ERROR`, `TIMEOUT`, `INVALID_RESULTS_SCHEMA`, `MODEL_VIOLATION`, `IMAGE_TOO_LARGE`, `ACCURACY_GATE_FAILED`.
- `ALLOWED_MODELS` is provided by the harness at runtime as an env var — **never hardcode a specific Fireworks model string anywhere in the code.**
- `FIREWORKS_API_KEY` and `FIREWORKS_BASE_URL` are harness-provided — do not require or reference personal keys.

### Current state (v2) — what exists already
- Tier 0: zero-token regex/keyword classifier → category
- Tier 1: deterministic solvers for math (Python), sentiment (TextBlob), NER (spaCy `en_core_web_sm`) — zero tokens
- Tier 2: single Fireworks model call (currently hardcoded to one model in `.env`) for everything else — no validation, no retry, no reasoning suppression
- Docker/compose setup, `linux/amd64` build, `.env` handling, submission checklist — this part is solid, keep it

### What's wrong with v2 (why we're changing it)
1. Only 3 of 8 categories are ever answered for free — everything else pays full Fireworks price.
2. No suppression of hidden reasoning tokens — models can silently bill 100+ tokens of invisible "thinking" even for a one-word answer. This is the single largest token-savings lever available and is currently unused.
3. Every non-deterministic task routes to one fixed, hardcoded model — no per-category cost/accuracy optimization, and hardcoding violates the "never hardcode model strings" rule (risk of `MODEL_VIOLATION` if that model isn't in this run's `ALLOWED_MODELS`).
4. No output validation or retry — a truncated/malformed Fireworks response just fails silently, risking the accuracy gate or `INVALID_RESULTS_SCHEMA`.

### Target architecture (v3)

```
task (from /input/tasks.json)
   │
   ▼
[Layer 0] Zero-LLM regex/keyword classifier → category
   │
   ▼
[Layer 1] Deterministic solvers (0 tokens, 0 local inference)
   │   - math: sandboxed eval / sympy
   │   - code debugging: AST parse + execute vs expected output (where checkable)
   │   - simple/templated logic: programmatic solve (where pattern-recognizable)
   │   - sentiment: TextBlob (keep existing)
   │   - NER: spaCy (keep existing)
   │   solved? ──yes──► write answer, done (0 Fireworks tokens)
   │   no
   ▼
[Layer 2] Local LLM (bundled 2-3B 4-bit GGUF, in-process via llama-cpp-python)
   │   - handles: summarization, factual QA, sentiment/NER fallback, first attempt at logic/code
   │   - self-validation: JSON schema check / word-limit check / finish_reason check / AST-compile check
   │   passes validation? ──yes──► write answer, done (0 Fireworks tokens)
   │   no
   ▼
[Layer 3] Fireworks API — per-category ranked model, reasoning suppressed
   │   - read ALLOWED_MODELS from env at runtime
   │   - use precomputed category→ranked-model table (built from offline bake-off)
   │   - apply reasoning_effort:"none" (or working equivalent per model family)
   │   - tight per-category max_tokens budget
   │   - validate response (schema/word-limit/finish_reason/AST-compile)
   │   passes? ──yes──► write answer, done
   │   no
   ▼
[Layer 4] Single bounded retry — thinking-ON, same or next-ranked model
   │   passes? ──yes──► write answer, done
   │   no ──► write best-effort non-empty answer (never leave a task_id blank)
   ▼
[Cache] Normalized-prompt → answer cache checked before Layer 2, updated after any successful answer
   ▼
/output/results.json + agent.log (per-task: layer used, tokens, category, escalation reason)
```

---

## 1. Goals & Non-Goals

**Goals**
- Maximize the fraction of tasks answered at Layer 0/1/2 (zero Fireworks tokens).
- When Fireworks must be used, minimize tokens via reasoning suppression + per-category cheapest-sufficient model + tight budgets.
- Never fail the accuracy gate — validation and bounded escalation exist specifically to prevent this.
- Never crash, never exceed image/RAM limits, never emit an invalid results schema.

**Non-Goals**
- Do not try to build a local model good enough to replace Fireworks for code/logic entirely — validate hard and escalate instead of over-trusting a 3B model.
- Do not use the local LLM to choose *which* Fireworks model to call — that decision is precomputed offline (see Phase 2), not made live.

---

## 2. Workstreams & Tasks

### Phase 1 — Local LLM tier (highest priority — currently missing entirely)

- [ ] Add `llama-cpp-python` to `requirements.txt` / `pyproject.toml`.
- [ ] Download **Qwen2.5-3B-Instruct-GGUF, Q4_K_M** (~1.9GB) at **Docker build time** (not runtime) into `/app/models/`. Source: Hugging Face `Qwen/Qwen2.5-3B-Instruct-GGUF`.
- [ ] Implement `clients.py::get_local_model()` — lazy singleton, loaded once on first call, not per-task. Log load time to `agent.log`.
- [ ] Implement `clients.py::call_local(system_prompt, user_prompt, max_tokens)` using `llm.create_chat_completion`.
- [ ] Wire Layer 2 into `agent.py`: for categories not resolved by Layer 1 deterministic solvers, attempt local LLM first.
- [ ] **Fallback model**: if 3B proves too slow/large for the 4GB/2vCPU budget once combined with spaCy/TextBlob, fall back to Qwen2.5-1.5B-Instruct-GGUF Q4_K_M (~1GB). Decision must be based on measured latency + accuracy from Phase 5 eval, not assumption.
- [ ] Measure total image footprint (spaCy model + TextBlob + GGUF weights + deps) — confirm comfortably under 10GB compressed.

**Acceptance criteria:** Local model loads once at startup, answers summarization/factual-QA/simple tasks in-process, zero calls to Fireworks for these, latency per task measured and logged.

---

### Phase 2 — Reasoning suppression + model bake-off (second highest priority)

- [ ] Write `scripts/bakeoff.py`: for each model string that could plausibly appear in `ALLOWED_MODELS`, test:
  - Baseline call (no reasoning param) — record tokens, accuracy on `eval_labeled.json`.
  - Call with `reasoning_effort: "none"` — record tokens, accuracy, and whether the API accepted the param (watch for HTTP 400).
  - Test any other documented reasoning-disable params per model family (e.g. `thinking: false`) if `reasoning_effort` isn't recognized.
- [ ] From bake-off results, build a static, hardcoded **`CATEGORY_MODEL_RANKING`** table in `router.py`: for each of the 8 categories, an ordered list of preferred models (cheapest-sufficient first) plus which reasoning-suppression param (if any) worked for each.
- [ ] Implement runtime logic in `clients.py::call_fireworks()`:
  - Read `ALLOWED_MODELS` from env, split into a set.
  - For the task's category, walk `CATEGORY_MODEL_RANKING` and pick the first model that's actually present in `ALLOWED_MODELS` this run.
  - Apply the reasoning-suppression param associated with that model in the ranking table.
  - **On HTTP 400 referencing the reasoning param**: catch it, retry the same call without the param, and set a run-wide flag to skip that param for all subsequent calls to that model this run (don't retry the param per-call after the first rejection — wastes latency).
- [ ] Log actual tokens used per call (`response.usage.total_tokens`) to `token_tracker.py`.

**Acceptance criteria:** No model string is ever hardcoded outside `CATEGORY_MODEL_RANKING`, which itself is validated against `ALLOWED_MODELS` at runtime, not assumed present. Reasoning suppression is applied by default and gracefully degrades on rejection.

---

### Phase 3 — Deterministic solver expansion

- [ ] Keep existing math/sentiment/NER solvers as-is.
- [ ] Add **code debugging solver**: for tasks where expected output is checkable, execute the provided code (sandboxed subprocess, timeout-bound) and compare against expected output; if it matches a fixable pattern (e.g. off-by-one, missing import), apply a rule-based fix and re-verify. If not confidently resolvable, fall through to Layer 2/3 — do not guess.
- [ ] Add **simple logic puzzle solver**: pattern-match templated logic tasks (e.g. constraint satisfaction with small enumerable state space) and solve programmatically. Fall through if the pattern isn't recognized.
- [ ] All deterministic solvers must have a **confidence gate**: only claim a solved answer when verification is 100% certain (e.g. code actually executed and matched expected output). Never emit a "probably right" deterministic answer — that's what Layer 2/3 are for.

**Acceptance criteria:** Deterministic layer only ever emits verified-correct answers; anything uncertain falls through cleanly to the next layer.

---

### Phase 4 — Validation & escalation logic

- [ ] Implement `validators.py`:
  - `validate_json(answer, expected_schema) -> bool`
  - `validate_word_limit(answer, category) -> bool`
  - `validate_finish_reason(response) -> bool` (reject if `finish_reason == "length"`, i.e. truncated)
  - `validate_code_compiles(answer) -> bool` (AST parse, and execute if expected output is known)
- [ ] Wire validation after every Layer 2 (local) and Layer 3 (Fireworks) attempt.
- [ ] Implement **Layer 4 bounded retry**: exactly one retry per task, with thinking/reasoning turned ON (i.e. remove the suppression param) and/or escalate to the next-ranked model in `CATEGORY_MODEL_RANKING`. No further retries after this — cap total per-task attempts.
- [ ] Implement final safety net: if all layers fail validation, still write a **non-empty, schema-valid** best-effort answer (last raw output, even if imperfect) — never leave a `task_id` missing or null, since that risks `INVALID_RESULTS_SCHEMA` and guarantees zero credit for that task.

**Acceptance criteria:** Every `task_id` present in input has a corresponding non-empty, schema-conformant entry in `/output/results.json`, with no exceptions.

---

### Phase 5 — Eval harness (build before finalizing routing tables)

- [ ] Create `input/eval_labeled.json` — 30-50 tasks spanning all 8 categories with ground-truth answers (hand-labeled or drawn from the practice tasks in the participant guide, expanded).
- [ ] Build `eval.py`:
  - Runs the full v3 pipeline against `eval_labeled.json`.
  - Reports: accuracy % overall and per-category, total Fireworks tokens, % tasks resolved at each layer (0/1/2/3/4), average latency per category.
  - Used to tune: local-model confidence thresholds, per-category model ranking, max_tokens budgets, and to decide 3B vs 1.5B for the local model.
- [ ] Run `eval.py` after each major phase change — this is the feedback loop for tuning, not a one-time step at the end.

**Acceptance criteria:** `eval.py` produces a clear before/after comparison (v2 baseline vs v3) on accuracy and token count.

---

### Phase 6 — Caching

- [ ] Implement `token_tracker.py` companion: normalized-prompt → answer cache (in-memory dict for a single run is sufficient — no cross-run persistence needed unless the task set repeats across runs).
- [ ] Check cache before attempting Layer 2; update cache after any successful answer at any layer.
- [ ] Normalize prompts (lowercase, strip whitespace, etc.) before cache key generation to catch near-duplicates.

**Acceptance criteria:** Repeated/duplicate prompts within a single run resolve from cache with zero additional inference cost.

---

### Phase 7 — Hardening against harness failure modes

- [ ] **`MODEL_VIOLATION`**: at startup, log a warning (not a crash) for any model in `CATEGORY_MODEL_RANKING` not present in this run's `ALLOWED_MODELS`; confirm the fallback walk never attempts a non-allowed model.
- [ ] **`TIMEOUT`**: add a global per-task timeout wrapper; if Fireworks is slow/unresponsive, fall back to the best local-only answer rather than hanging.
- [ ] **`INVALID_RESULTS_SCHEMA`**: confirm output writer always emits the exact expected keys (`task_id`, `answer` — confirm exact schema against harness docs) for every input task, no extras, no omissions.
- [ ] **`IMAGE_TOO_LARGE`**: after Phase 1, run `docker images` and confirm compressed size; strip unnecessary build artifacts/cache from the final image layer.
- [ ] **`RUNTIME_ERROR`**: test the full container on a resource-constrained environment (`docker run --memory=4g --cpus=2`) to simulate grading conditions before submission.
- [ ] **`PULL_ERROR`**: confirm final image is pushed to a public registry with an exact tag reference, no `https://` prefix, before final submission.
- [ ] **`ACCURACY_GATE_FAILED`**: after Phase 5 eval shows accuracy comfortably above threshold (not just barely passing — practice tasks won't perfectly match the real grading set), do a final full run.

---

## 3. File-by-File Change Summary

| File | Change |
|---|---|
| `clients.py` | Add `get_local_model()`, `call_local()`, rewrite `call_fireworks()` to use `CATEGORY_MODEL_RANKING` + reasoning suppression + `ALLOWED_MODELS` validation |
| `router.py` | Add `CATEGORY_MODEL_RANKING` table (from bake-off), keep existing zero-token classifier |
| `validators.py` | **New file** — JSON/word-limit/finish_reason/AST-compile checks |
| `solvers.py` | Expand existing math/sentiment/NER with code-debug and simple-logic solvers |
| `token_tracker.py` | Add cache logic, per-layer token/accuracy logging |
| `agent.py` | Rewire main loop to the 5-layer pipeline (0→1→2→3→4→cache write) |
| `scripts/bakeoff.py` | **New file** — offline model/reasoning-param benchmarking |
| `eval.py` | **New file** — full pipeline eval against `eval_labeled.json` |
| `input/eval_labeled.json` | **New file** — labeled dev set |
| `Dockerfile.uv` | Add GGUF model download at build time, confirm image size after |
| `requirements.txt` / `pyproject.toml` | Add `llama-cpp-python` |

---

## 4. Definition of Done

- [ ] All 8 categories have a defined path through the layer pipeline (not just math/sentiment/NER).
- [ ] Zero hardcoded Fireworks model strings anywhere in the codebase — all resolved via `ALLOWED_MODELS` + ranking table at runtime.
- [ ] Reasoning suppression applied by default, with graceful degradation on rejection.
- [ ] Every task_id always produces a valid, non-empty answer, even in total-failure fallback paths.
- [ ] `eval.py` shows accuracy comfortably clearing the gate and total Fireworks tokens meaningfully lower than the v2 baseline.
- [ ] Clean `docker buildx build --platform linux/amd64` from scratch succeeds, image under 10GB, container runs under simulated 4GB/2vCPU constraints without error.
- [ ] `agent.log` clearly shows per-task layer/model/token/escalation decisions for post-hoc review.
```