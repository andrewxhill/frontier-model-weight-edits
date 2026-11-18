"""Minimal standard LoRA: frozen W plus (alpha/r) B A; train A/B only."""
import math,torch
class LoRALinear(torch.nn.Module):
 def __init__(self,base,rank,alpha):
  super().__init__()
  if not isinstance(base,torch.nn.Linear)or rank<1 or alpha<=0:raise ValueError('positive-rank linear LoRA required')
  self.base=base;self.base.requires_grad_(False);self.scale=alpha/rank
  self.A=torch.nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=base.weight.dtype));self.B=torch.nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=base.weight.dtype));torch.nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
 @property
 def weight(self):return self.base.weight
 @property
 def bias(self):return self.base.bias
 def forward(self,x):return self.base(x)+torch.nn.functional.linear(torch.nn.functional.linear(x,self.A),self.B)*self.scale
def install(model,names,rank,alpha):
 if not names or len(set(names))!=len(names):raise ValueError('nonempty unique adapter targets required')
 model.requires_grad_(False);layers={}
 for name in names:
  original=model.get_submodule(name);parent,_,child=name.rpartition('.');layer=LoRALinear(original,rank,alpha);setattr(model.get_submodule(parent)if parent else model,child,layer);layers[name]=layer
 return layers
def adapter_state(layers):return {name+'.'+field:getattr(layer,field).detach().cpu().clone()for name,layer in layers.items()for field in ['A','B']}
def load_adapter_state(layers,state):
 expected={name+'.'+field for name in layers for field in ['A','B']}
 if set(state)!=expected:raise ValueError('adapter checkpoint target set differs')
 with torch.no_grad():
  for name,layer in layers.items():
   for field in ['A','B']:
    source=state[name+'.'+field];target=getattr(layer,field)
    if source.shape!=target.shape or source.dtype!=torch.float32 or not torch.isfinite(source).all():raise ValueError('invalid adapter checkpoint tensor')
    target.copy_(source.to(target.device))
def merge(model,layers):
 if not layers or any(model.get_submodule(name) is not layer for name,layer in layers.items()):raise ValueError('all LoRA adapters must still be installed before merging')
 with torch.no_grad():
  for name,layer in layers.items():
   layer.base.weight.add_(layer.B@layer.A,alpha=layer.scale);parent,_,child=name.rpartition('.');setattr(model.get_submodule(parent)if parent else model,child,layer.base)
 return model
