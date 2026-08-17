"""Put the package root on sys.path so `from pipeline...` resolves.

The pipeline is a standalone package run via `python run_pipeline.py` from
its own directory; tests need the same import root.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
