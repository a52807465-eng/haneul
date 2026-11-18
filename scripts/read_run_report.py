"""
런 리포트 조회 스크립트.

실행 후 `runs/aspirin-solar-pro4/run-report.json`을 사람이 읽기 좋게 출력한다.
"""

from __future__ import annotations

import json
from pathlib import Path

REPORT_PATH = Path("runs/aspirin-solar-pro4/run-report.json")

def main() -> None:
    if not REPORT_PATH.exists():
        print(f"런 리포트를 찾을 수 없습니다: {REPORT_PATH}", flush=True)
        raise SystemExit(2)
    report = json.loads(REPORT_PATH.read_text())
    for key, value in report.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"{key}: {value}")
    print()
    print("완료 여부:", report.get("status"))
    print("실패 추적:", report.get("failure"))

if __name__ == "__main__":
    main()
