"""Resolve the isolated GeoTransformer snapshot before importing its modules."""
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / 'vendor'
sys.path.insert(0, str(VENDOR))

