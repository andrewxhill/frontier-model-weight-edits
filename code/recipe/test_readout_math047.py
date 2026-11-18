"""Tiny NumPy-only check of the actual extracted late-readout functions."""
import ast
from pathlib import Path
import unittest

class ReadoutMath(unittest.TestCase):
    def test_balanced_two_class_closed_form(self):
        import numpy as np
        source=ast.parse(Path(__file__).with_name('math048.py').read_text())
        # Execute the actual extracted functions and declared collection imports,
        # avoiding tensor-library imports; no model/feature arrays are involved.
        nodes=[n for n in source.body if isinstance(n,ast.ImportFrom) and n.module=='collections' or isinstance(n,ast.FunctionDef) and n.name in ('mask_soft','weighted_design','fitres')]
        scope={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual-selected-math','exec'),scope)
        H=np.array([[1.,0.],[0.,1.],[2.,1.],[1.,2.]])
        D,b=scope['fitres'](H,np.zeros((4,11)),np.array([0,1,0,1]),np.full(4,2),np.array(['a','a','b','b']),lam=.1,kind='residual')
        np.testing.assert_allclose(D[:,:2],np.array([[10/21,-10/21],[-10/21,10/21]]),atol=1e-12)
        np.testing.assert_allclose(D[:,2:],0,atol=1e-12);np.testing.assert_allclose(b,0,atol=1e-12)

if __name__=='__main__':unittest.main()
