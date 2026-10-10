"""PEP 517 entry point; all patch/build logic lives in AKVSgl._build."""
import runpy
from pathlib import Path

_here = Path(__file__).resolve().parent
_assets = _here / "AKVSgl"
if not (_assets / "_build.py").is_file():
    _assets = _here.parents[1]
globals().update(runpy.run_path(str(_assets / "_build.py"))["hooks"]("gateway"))
