"""Recompute the edit from original weights and public practice inputs.

The separate apply_edit.py replays the verified exact shipped delta.
"""
from pathlib import Path
import runpy
import sys


def main():
    sys.dont_write_bytecode = True
    recipe = Path(__file__).resolve().parent / 'recipe'
    sys.path.insert(0, str(recipe))
    runpy.run_path(str(recipe / 'build_edit048.py'), run_name='__main__')


if __name__ == '__main__':
    main()
