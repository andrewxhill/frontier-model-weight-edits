import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

class PracticeFetch(unittest.TestCase):
    def test_fetches_only_selected_hashes_and_preserves_foreign_partial(self):
        spec=importlib.util.spec_from_file_location('fetch_practice049',Path(__file__).with_name('fetch_practice049.py'))
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        content=b'synthetic frame bytes';digest=hashlib.sha256(content).hexdigest()
        row=dict(key='ic/a:0',dataset='ic',source_relative_path='images/a.jpg',image_sha256=digest,url=m.BASES['ic']+'images/a.jpg')
        calls=[]
        def opener(url,timeout):calls.append(url);return io.BytesIO(content)
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)/'ic';directory.mkdir();foreign=directory/'previous.partial';foreign.write_bytes(b'untouched')
            result=m.fetch_rows([row,dict(row,key='ic/not-selected:0',url='https://invalid.example/never-read')],{'ic/a:0'},root,opener=opener)
            self.assertEqual(result['verified_frames'],1);self.assertEqual(len(calls),1)
            self.assertEqual((directory/(digest+'.image')).read_bytes(),content);self.assertEqual(foreign.read_bytes(),b'untouched')
            bad=dict(row,key='ic/b:0',source_relative_path='../bad.jpg',url='https://invalid.example/not-requested')
            with self.assertRaises(ValueError):m.fetch_rows([bad],{'ic/b:0'},root,opener=opener)
            self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main()
