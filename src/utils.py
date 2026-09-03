"""Small shared utilities for project inputs and reproducibility checks."""
import hashlib
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[1]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
