from __future__ import annotations

import json
from pathlib import Path

from src.modeling import run_pipeline


def main() -> None:
    summary = run_pipeline(Path(__file__).resolve().parent)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
