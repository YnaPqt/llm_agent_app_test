import json
from pathlib import Path


MEMORY_FILE = Path(r"C:\Users\ronal\Documents\LOCAL LLM\mini-math\llm_agent_app\data\memory.json")
MEMORY_FILE.parent.mkdir(exist_ok=True)

def load_memory():
    if not MEMORY_FILE.exists():
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_message(role, message):
    memory = load_memory()
    memory.append({"role": role, "message": message})
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)