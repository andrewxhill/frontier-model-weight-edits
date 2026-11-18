import importlib.util
from pathlib import Path
import unittest
class PracticeAdapter(unittest.TestCase):
    def module(self):
        spec=importlib.util.spec_from_file_location('practice_adapter047',Path(__file__).with_name('practice_adapter047.py'))
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    def test_recorded_order_and_unknown_are_preserved(self):
        m=self.module();rows=[dict(key='ic/b:0',dataset='ic',id='b',box_index=0,label=None,location='c'),dict(key='ic/a:0',dataset='ic',id='a',box_index=0,label='fox',location='c')]
        selected=m.select_rows(rows,['ic/a:0','ic/b:0'])
        self.assertEqual([r['key'] for r in selected],['ic/a:0','ic/b:0'])
        self.assertIsNone(selected[1]['label'])
        with self.assertRaises(ValueError):m.select_rows(rows,['ic/b:0'],require_labels=True)
    def test_nonpractice_or_duplicate_id_refuses(self):
        m=self.module();rows=[dict(key='ic/a:0',dataset='ic',id='a',box_index=0,label='fox',location='c')]
        with self.assertRaises(ValueError):m.select_rows(rows,['ic/sealed:0'])
        with self.assertRaises(ValueError):m.select_rows(rows,['ic/a:0','ic/a:0'])
if __name__=='__main__':unittest.main()
