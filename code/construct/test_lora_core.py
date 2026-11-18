"""Tiny CPU-only algebra tests; no study data or pretrained model."""
import unittest
try:import torch
except ImportError:torch=None
if torch is not None:from lora_core import LoRALinear,install,merge,adapter_state,load_adapter_state
@unittest.skipIf(torch is None,'Torch is not installed')
class LoRATests(unittest.TestCase):
 def test_zero_update_is_original_and_base_never_gets_gradients(self):
  torch.manual_seed(9);base=torch.nn.Linear(5,3);old=base.weight.detach().clone();layer=LoRALinear(base,2,4);x=torch.randn(4,5);self.assertTrue(torch.equal(layer(x),base(x)))
  layer(x).square().sum().backward();self.assertIsNone(base.weight.grad);self.assertIsNone(base.bias.grad);self.assertGreater(layer.B.grad.abs().sum().item(),0);self.assertTrue(torch.equal(base.weight,old))
 def test_merge_and_checkpoint_restore_match_adapter_output(self):
  torch.manual_seed(11);model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Linear(4,2));layers=install(model,['0','1'],2,4)
  with torch.no_grad():
   for layer in layers.values():layer.B.normal_(std=.02)
  x=torch.randn(3,4);out=model(x).detach();state=adapter_state(layers)
  with torch.no_grad():
   for layer in layers.values():layer.A.zero_();layer.B.zero_()
  load_adapter_state(layers,state);self.assertTrue(torch.equal(model(x).detach(),out));merge(model,layers);torch.testing.assert_close(model(x),out,atol=2e-6,rtol=2e-6)
 def test_only_declared_modules_change_and_bad_checkpoint_rejected(self):
  model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Linear(4,2));unchanged=model[1].weight.detach().clone();layers=install(model,['0'],2,4);self.assertTrue(torch.equal(model[1].weight,unchanged))
  with self.assertRaises(ValueError):load_adapter_state(layers,{})

 def test_repeat_merge_is_rejected_without_touching_merged_weights(self):
  model=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.Linear(2,2));layers=install(model,['0','1'],2,4)
  with torch.no_grad():
   for layer in layers.values():layer.B.fill_(.125)
  merge(model,layers);expected=[p.detach().clone()for p in model.parameters()]
  with self.assertRaises(ValueError):merge(model,layers)
  for actual,saved in zip(model.parameters(),expected):self.assertTrue(torch.equal(actual,saved))

if __name__=='__main__':unittest.main()
