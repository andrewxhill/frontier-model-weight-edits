import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

class ExactAttemptResume(unittest.TestCase):
    def module(self):
        spec=importlib.util.spec_from_file_location('resume048',Path(__file__).with_name('resume048.py'))
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    def test_only_exact_bound_incomplete_attempt_resumes(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as parent:
            out=Path(parent)/'attempt';binding={'source':{'program':'a'},'inputs':{'images':'b'},'settings':{'gain':4}}
            self.assertEqual(m.admit(out,binding,resume=False),'created')
            (out/'capture.npz').write_bytes(b'closed evidence')
            self.assertEqual(m.admit(out,binding,resume=True),'resumed')
            before={p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}
            changed=dict(binding,settings={'gain':3})
            with self.assertRaises(ValueError):m.admit(out,changed,resume=True)
            after={p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}
            self.assertEqual(before,after)
            (out/'derivation-receipt.json').write_text(json.dumps({'complete':True}))
            with self.assertRaises(FileExistsError):m.admit(out,binding,resume=True)
    def test_resume_cannot_adopt_unbound_existing_directory(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as parent:
            with self.assertRaises(ValueError):m.admit(Path(parent),{'source':'a'},resume=True)

    def test_unclosed_model_directory_is_preserved_and_refused(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as parent:
            out=Path(parent)/'attempt';binding={'source':'fixed'};m.admit(out,binding,resume=False)
            model=out/'derived-model';model.mkdir();(model/'partial.safetensors').write_bytes(b'preserve this incomplete close')
            before={str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()}
            with self.assertRaises(ValueError):m.admit(out,binding,resume=True)
            after={str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()}
            self.assertEqual(before,after)

if __name__=='__main__':unittest.main()
