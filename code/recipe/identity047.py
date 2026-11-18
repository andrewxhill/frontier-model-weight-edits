"""Exact tensor-byte identity, canonical storage aliases and base-anchored states."""
import hashlib,json
import torch

def canonical(value):return json.dumps(value,sort_keys=True,allow_nan=False)
def tensor_sha(value):
 if not isinstance(value,torch.Tensor)or value.layout!=torch.strided or value.is_meta:raise ValueError('materialized dense tensor required')
 raw=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
 digest=hashlib.sha256(canonical(dict(dtype=str(value.dtype),shape=list(value.shape))).encode()+b'\n')
 view=memoryview(raw).cast('B')
 for start in range(0,len(view),1048576):digest.update(view[start:start+1048576])
 return digest.hexdigest()
def alias_map(state_dict):
 if not state_dict or any(not isinstance(k,str)or not isinstance(v,torch.Tensor)for k,v in state_dict.items()):raise ValueError('nonempty named tensor state required')
 aliases={};storage={};views={}
 for name in sorted(state_dict):
  value=state_dict[name]
  if value.layout!=torch.strided or value.is_meta:raise ValueError('materialized dense tensor required: '+name)
  address=(str(value.device),value.untyped_storage().data_ptr())if value.numel()else('empty',id(value))
  geometry=(value.storage_offset(),tuple(value.shape),tuple(value.stride()),str(value.dtype))
  if address in views and views[address]!=geometry:raise ValueError('overlapping unlike tensor views require an explicit architecture adapter')
  views[address]=geometry;aliases[name]=storage.setdefault((address,geometry),name)
 return aliases
def state_manifest(state_dict):
 aliases=alias_map(state_dict)
 return {name:tensor_sha(state_dict[name])for name in sorted(state_dict)if aliases[name]==name}
def _anchor(aliases,tensors):
 if set(tensors)!={name for name,owner in aliases.items()if name==owner}or any(owner not in tensors or aliases.get(owner)!=owner for owner in aliases.values()):raise ValueError('complete canonical alias/tensor map required')
 return hashlib.sha256(canonical(dict(names=sorted(aliases),aliases=aliases,base=tensors)).encode()).hexdigest()
def identity_manifest(state_dict):
 aliases=alias_map(state_dict);tensors=state_manifest(state_dict);anchor=_anchor(aliases,tensors)
 result=dict(format='full-tensor-state-v1',base_sha256=anchor,aliases=aliases,tensors=tensors)
 result['state_sha256']=hashlib.sha256(canonical(dict(base_sha256=anchor,aliases=aliases,tensors=tensors)).encode()).hexdigest();return result
def state_fingerprint(state_dict,*,base_manifest=None):
 """Edited states require the ORIGINAL base manifest; never self-anchor an edit."""
 aliases=alias_map(state_dict);tensors=state_manifest(state_dict)
 if base_manifest is None:anchor=_anchor(aliases,tensors)
 else:
  if aliases!=base_manifest['aliases']:raise ValueError('base tied aliases/architecture differ')
  anchor=_anchor(base_manifest['aliases'],base_manifest['tensors'])
  if anchor!=base_manifest['base_sha256']:raise ValueError('base identity anchor differs')
 return hashlib.sha256(canonical(dict(base_sha256=anchor,aliases=aliases,tensors=tensors)).encode()).hexdigest()
