"""pytest configuration — add src/ to sys.path so tests import from source tree."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
