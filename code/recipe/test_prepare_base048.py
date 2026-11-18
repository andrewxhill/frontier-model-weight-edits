import importlib.util
from pathlib import Path
import unittest

class BaseSpec(unittest.TestCase):
    def module(self):
        spec=importlib.util.spec_from_file_location('prepare_base048',Path(__file__).with_name('prepare_base048.py'))
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    def test_base_only_projection_ignores_target_and_delta(self):
        m=self.module();source=dict(base_config={'model_type':'llava_onevision'},base_manifest={'tensors':{},'aliases':{}},base_state_sha256=m.ORIGINAL,generation_config_text='{"repetition_penalty":1.1}',target_manifest={'PRIVATE':'TARGET'},delta_tensor_sha256={'x':'secret'})
        result=m.project_base_spec(source)
        self.assertEqual(set(result),{'base_config','base_manifest','base_state_sha256','generation_config_text'})
        self.assertNotIn('TARGET',str(result));self.assertNotIn('secret',str(result))
        with self.assertRaises(ValueError):m.project_base_spec(dict(source,base_state_sha256='wrong'))

if __name__=='__main__':unittest.main()
