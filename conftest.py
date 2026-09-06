"""
TernaryGuard — Root conftest.py

Ensures the project root is on sys.path so that all test files
can import from `model`, `engine_software`, etc. without fragile
sys.path hacks in each test module.
"""

import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
