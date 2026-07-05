from __future__ import annotations

import sys

from .run_pdf_vlm_symbolic_vlm_baseline import main


def _ensure_flag(flag: str) -> None:
    if flag not in sys.argv:
        sys.argv.append(flag)


if __name__ == "__main__":
    _ensure_flag("--skip-openreview-papers")
    _ensure_flag("--show-progress")
    raise SystemExit(main())
