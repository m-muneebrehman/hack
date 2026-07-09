import pytest
import os
import json
from agent import keyword_classify, get_worker_prompt, TaskInput, TaskResult

def test_keyword_classifier_matches():
    # Test high-confidence matches
    assert keyword_classify("Calculate 15% of 200")[0] == "math"
    assert keyword_classify("This product is amazing! great review")[0] == "sentiment"
    assert keyword_classify("Extract names and entities")[0] == "ner"
    assert keyword_classify("Please summarize this text")[0] == "summary"
    assert keyword_classify("I found a bug in the code, traceback error")[0] == "code_debug"
    assert keyword_classify("write code to sort a list")[0] == "code_gen"
    assert keyword_classify("solve this logic puzzle")[0] == "logic"
    assert keyword_classify("who is the president of the US?")[0] == "factual"

def test_keyword_classifier_ambiguous():
    # Test ambiguous prompts that should fall back to LLM
    cat, conf = keyword_classify("I want a green apple.")
    assert cat == "general"
    assert conf < 0.7

def test_worker_prompts():
    prompt = get_worker_prompt("math")
    assert "step-by-step" in prompt
    
    prompt = get_worker_prompt("unknown_category")
    assert "factually" in prompt  # falls back to general

def test_pydantic_models():
    task = TaskInput(task_id="t1", prompt="test")
    assert task.task_id == "t1"
    
    result = TaskResult(task_id="t1", result="output")
    assert result.result == "output"

def test_output_schema():
    results = [
        {"task_id": "t1", "result": "answer 1"},
        {"task_id": "t2", "result": "answer 2"}
    ]
    # Verify it matches expected output format by checking we can dump it
    json_str = json.dumps(results)
    assert "task_id" in json_str
    assert "result" in json_str
