"""Recompute the camera-trap edit from the original weights and public practice inputs.

To reproduce the released weights exactly, use apply_edit.py and the delta instead.
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
