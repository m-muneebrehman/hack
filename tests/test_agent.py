import pytest
import json
from agent import (
    keyword_classify,
    get_worker_prompt,
    _try_solve_math,
    TaskInput,
    TaskResult,
    WORKER_PROMPTS,
)


# ---------------------------------------------------------------------------
# Keyword Classifier
# ---------------------------------------------------------------------------
class TestKeywordClassifier:
    def test_math(self):
        assert keyword_classify("Calculate 15% of 200")[0] == "math"
        assert keyword_classify("A store sells 15% of 240 items")[0] == "math"
        assert keyword_classify("What is the square root of 144?")[0] == "math"

    def test_sentiment(self):
        assert keyword_classify("Classify the sentiment of this review: great product")[0] == "sentiment"

    def test_ner(self):
        assert keyword_classify("Extract all named entities from this text")[0] == "ner"
        assert keyword_classify("Identify organizations in this text: WHO")[0] == "ner"

    def test_summary(self):
        assert keyword_classify("Summarize the following in exactly one sentence")[0] == "summary"
        assert keyword_classify("Please summarize this text")[0] == "summary"

    def test_code_debug(self):
        assert keyword_classify("This function has a bug: def foo(): pass")[0] == "code_debug"
        assert keyword_classify("Find the bug in this code")[0] == "code_debug"

    def test_code_gen(self):
        assert keyword_classify("Write a Python function that returns fibonacci")[0] == "code_gen"
        assert keyword_classify("Write a python script that reads a CSV file")[0] == "code_gen"

    def test_logic(self):
        assert keyword_classify("If all cats are mammals and Flippers is a cat, is it a mammal?")[0] == "logic"
        assert keyword_classify("Three friends each own a different pet. Who owns the cat?")[0] == "logic"

    def test_factual(self):
        assert keyword_classify("What is the capital of France?")[0] == "factual"
        assert keyword_classify("Explain quantum entanglement")[0] == "factual"

    def test_ambiguous_returns_general(self):
        cat, conf = keyword_classify("I want a green apple.")
        assert cat == "general"
        assert conf < 0.7


# ---------------------------------------------------------------------------
# Deterministic Math Solver
# ---------------------------------------------------------------------------
class TestMathSolver:
    def test_percentage_of(self):
        result = _try_solve_math("Calculate 15% of 200")
        assert result == "30"

    def test_square_root(self):
        result = _try_solve_math("What is the square root of 144?")
        assert result == "12"

    def test_square_root_multiply(self):
        result = _try_solve_math("Calculate the square root of 144 and multiply by 5")
        assert result == "60"

    def test_unsupported_returns_none(self):
        result = _try_solve_math("A store has 240 items and sells 60 more")
        assert result is None


# ---------------------------------------------------------------------------
# Worker Prompts
# ---------------------------------------------------------------------------
class TestWorkerPrompts:
    def test_all_categories_have_prompts(self):
        for cat in ["factual", "math", "sentiment", "summary", "ner",
                    "code_debug", "logic", "code_gen", "general"]:
            assert cat in WORKER_PROMPTS
            assert len(WORKER_PROMPTS[cat]) > 5

    def test_get_worker_prompt_function(self):
        prompt = get_worker_prompt("math")
        assert "step" in prompt.lower()

    def test_fallback_to_general(self):
        prompt = get_worker_prompt("unknown_category")
        assert "factually" in prompt or "directly" in prompt


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class TestPydanticModels:
    def test_task_input(self):
        task = TaskInput(task_id="t1", prompt="test prompt")
        assert task.task_id == "t1"
        assert task.prompt == "test prompt"

    def test_task_result_uses_answer_key(self):
        # CRITICAL: harness requires "answer" not "result"
        result = TaskResult(task_id="t1", answer="some answer")
        assert result.answer == "some answer"
        assert hasattr(result, "answer")
        assert not hasattr(result, "result")


# ---------------------------------------------------------------------------
# Output Schema Compliance
# ---------------------------------------------------------------------------
class TestOutputSchema:
    def test_output_has_answer_key(self):
        results = [
            {"task_id": "t1", "answer": "Canberra, near Lake Burley Griffin."},
            {"task_id": "t2", "answer": "144 items remain."},
        ]
        json_str = json.dumps(results)
        parsed = json.loads(json_str)

        for item in parsed:
            assert "task_id" in item, "Missing task_id"
            assert "answer" in item, "Missing answer — harness will reject this!"
            assert "result" not in item, "Wrong key 'result' — should be 'answer'!"

    def test_valid_json_output(self):
        results = [{"task_id": f"practice-0{i}", "answer": f"answer {i}"} for i in range(1, 9)]
        # Should not raise
        json_str = json.dumps(results)
        parsed = json.loads(json_str)
        assert len(parsed) == 8
