import os
import sys
import json
import time
import re
import logging
from typing import Literal, Tuple, List, Dict, Any
from pydantic import BaseModel

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Setup logging
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
INPUT_FILE = "/input/tasks.json"
OUTPUT_FILE = "/output/results.json"
# Fallback to local files for testing if not running in docker
if not os.path.exists("/input"):
    INPUT_FILE = "input/tasks.json"
    OUTPUT_FILE = "output/results.json"

# --- Configuration for Gemini ---
MODEL = "gemini-2.5-flash"  # Use pro or flash for testing
TEMPERATURE = 0.1
MAX_TOKENS = 2048


# --- Pydantic Models ---
class TaskInput(BaseModel):
    task_id: str
    prompt: str

class CategoryDecision(BaseModel):
    category: Literal["factual", "math", "sentiment", "summary", "ner", "code_debug", "logic", "code_gen", "general"]

class TaskResult(BaseModel):
    task_id: str
    result: str


# --- Core Components ---
def keyword_classify(prompt: str) -> Tuple[str, float]:
    """Zero-token heuristic classification for high-confidence matches."""
    prompt_lower = prompt.lower()
    
    # Define keywords for each category
    rules = {
        "logic": [r"\blogic\b", r"\bpuzzle\b", r"\bconstraint\b", r"\bdeduce\b", r"\bsyllogism\b"],
        "math": [r"\bcalculate\b", r"\bsolve\b", r"\bequation\b", r"\bpercent\b", r"\bsum\b", r"[0-9]+[+\-*/][0-9]+"],
        "sentiment": [r"\bsentiment\b", r"\bpositive\b", r"\bnegative\b", r"\bfeel\b", r"\bopinion\b", r"\breview\b"],
        "ner": [r"\bentities\b", r"\bnamed entity\b", r"\bextract names\b", r"\bidentify people\b", r"\borganizations\b"],
        "summary": [r"\bsummarize\b", r"\bsummary\b", r"\btldr\b", r"\bshorten\b", r"\bcondense\b"],
        "code_debug": [r"\bbug\b", r"\bfix\b", r"\berror\b", r"\bdebug\b", r"\btraceback\b", r"\bexception\b"],
        "code_gen": [r"\bwrite code\b", r"\bimplement\b", r"\bfunction that\b", r"\bclass that\b", r"\bpython script\b"],
        "factual": [r"\bwho is\b", r"\bwhat is\b", r"\bwhen did\b", r"\bwhere is\b", r"\bexplain\b", r"\bdefine\b"]
    }
    
    for category, patterns in rules.items():
        for pattern in patterns:
            if re.search(pattern, prompt_lower):
                return category, 0.9  # High confidence match
                
    return "general", 0.0  # Fallback to LLM if no matches

def get_worker_prompt(category: str) -> str:
    """Returns the specialized minimal prompt for the given category."""
    prompts = {
        "factual": "Answer factually and concisely. Under 150 words. Cite key facts.",
        "math": "Solve step-by-step. Show essential steps only. State final answer clearly on its own line.",
        "sentiment": "Analyze sentiment. Return: sentiment (positive/negative/neutral/mixed), confidence (0-1), brief justification.",
        "summary": "Summarize the text. Follow any length constraints exactly. Be concise and complete.",
        "ner": "Extract all named entities. Return as JSON: {\"entities\": [{\"text\": ..., \"type\": PERSON/ORG/LOC/DATE/...}]}",
        "code_debug": "Identify the bug. Explain briefly. Provide corrected code.",
        "logic": "List all constraints. Reason step by step. State conclusion clearly.",
        "code_gen": "Write clean, correct code with type hints. Include brief docstrings. No unnecessary commentary.",
        "general": "Answer directly and factually. Under 200 words."
    }
    return prompts.get(category, prompts["general"])

class TaskProcessor:
    def __init__(self):
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        # Initialize structured supervisor for ambiguous tasks
        # method="json_schema" is needed for many models to strictly enforce the output
        try:
            self.supervisor = self.llm.with_structured_output(CategoryDecision)
        except Exception as e:
            logger.warning(f"Failed to initialize structured output, falling back to raw prompt: {e}")
            self.supervisor = None

    def _llm_classify(self, prompt: str) -> str:
        """Fallback LLM classification for ambiguous tasks."""
        if self.supervisor:
            try:
                sys_msg = "Classify the following task into exactly one category: factual, math, sentiment, summary, ner, code_debug, logic, code_gen, general."
                response = self.supervisor.invoke([
                    SystemMessage(content=sys_msg),
                    HumanMessage(content=prompt)
                ])
                return response.category
            except Exception as e:
                logger.error(f"Supervisor LLM routing failed: {e}")
        return "general"

    def process_task(self, task_id: str, prompt: str) -> TaskResult:
        """End-to-end task processing."""
        start_time = time.time()
        
        # 1. Try keyword classification
        category, confidence = keyword_classify(prompt)
        
        # 2. If low confidence, use LLM supervisor
        if confidence < 0.7:
            category = self._llm_classify(prompt)
            logger.info(f"Task {task_id}: LLM Router -> {category}")
        else:
            logger.info(f"Task {task_id}: Keyword Router -> {category}")
            
        # 3. Get specialized prompt
        system_prompt = get_worker_prompt(category)
        
        # 4. Call LLM with specialized prompt
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
            result_text = response.content
            
            # Extract token usage
            tokens = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens = response.usage_metadata.get('total_tokens', 0)
            elif 'token_usage' in response.response_metadata:
                tokens = response.response_metadata['token_usage'].get('total_tokens', 0)
                
            logger.info(f"Task {task_id}: LLM completed. Tokens used: {tokens}")
        except Exception as e:
            logger.error(f"Worker execution failed for task {task_id}: {e}")
            # Fallback
            result_text = f"Error processing task: {str(e)}"
            
        elapsed = time.time() - start_time
        logger.info(f"Task {task_id} completed in {elapsed:.2f}s")
        
        return TaskResult(task_id=task_id, result=result_text)

def main():
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 1. Load tasks
    try:
        with open(INPUT_FILE, "r") as f:
            raw_tasks = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read input file {INPUT_FILE}: {e}")
        sys.exit(1)
        
    tasks = []
    for t in raw_tasks:
        try:
            tasks.append(TaskInput(**t))
        except Exception as e:
            logger.warning(f"Skipping invalid task: {t} - {e}")
            
    if not tasks:
        logger.warning("No valid tasks found.")
        sys.exit(0)
        
    # 2. Process sequentially
    processor = TaskProcessor()
    results = []
    
    total_start = time.time()
    for task in tasks[:3]:
        try:
            result = processor.process_task(task.task_id, task.prompt)
            results.append({"task_id": result.task_id, "result": result.result})
        except Exception as e:
            logger.error(f"Fatal error on task {task.task_id}: {e}")
            results.append({"task_id": task.task_id, "result": ""})
            
    # 3. Write results
    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write output file {OUTPUT_FILE}: {e}")
        # Print to stdout as a final fallback
        print(json.dumps(results, indent=2))
        sys.exit(1)
        
    total_elapsed = time.time() - total_start
    logger.info(f"Successfully processed {len(tasks)} tasks in {total_elapsed:.2f}s")

if __name__ == "__main__":
    main()
