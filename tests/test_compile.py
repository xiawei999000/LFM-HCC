"""Compile every Python file in the repository (acceptance check 1)."""
import os
import py_compile

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {'.git', '__pycache__', '.venv', 'node_modules'}


def _iter_py_files():
    for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith('.py'):
                yield os.path.join(dirpath, fn)


PY_FILES = sorted(_iter_py_files())


def test_repo_has_python_files():
    assert len(PY_FILES) > 0


@pytest.mark.parametrize('path', PY_FILES, ids=[os.path.relpath(p, PROJECT_ROOT) for p in PY_FILES])
def test_file_compiles(path):
    py_compile.compile(path, doraise=True)
