import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text())
    print("format_version:", manifest.get("format_version"))
    print("status:", manifest.get("status"))
    print("run_id:", manifest.get("run_id"))
    print("target:", manifest.get("target"))
    print("target_id:", manifest.get("target_id"))
    print("completed_stages:", manifest.get("completed_stages"))
    print("artifacts:", manifest.get("artifacts"))
    print("current_stage:", manifest.get("current_stage"))
    print("failure:", manifest.get("failure"))
    print("llm_usage:", manifest.get("llm_usage"))
    print("stage_durations_seconds:", manifest.get("stage_durations_seconds"))
    print("provider:", manifest.get("provider"))
    print("model:", manifest.get("model"))

if __name__ == "__main__":
    main()
