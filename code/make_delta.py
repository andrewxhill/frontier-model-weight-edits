"""Write the delta and manifest that rebuild a released model from the public base.

The delta holds exact replacement values, not differences: whole projector tensors,
merged LoRA query/value matrices and changed rows of the tied embedding/output head.
`apply_edit.py` reads the pair, applies it to the public base and checks every byte.

Example:
  python make_delta.py --base work/base --target released-model \
    --template delta/check39/edit_manifest.json --output delta/island
"""
import argparse,hashlib,json,re
from pathlib import Path

HEAD='lm_head.weight';PROJECTOR='model.multi_modal_projector.'
LORA_TARGET=re.compile(r'model\.language_model\.layers\.\d+\.self_attn\.[qv]_proj\.weight')
TEMPLATE_FIELDS=('base_config','base_manifest','base_state_sha256','upstream')

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load(path):
    import torch
    from transformers import LlavaOnevisionForConditionalGeneration
    model=LlavaOnevisionForConditionalGeneration.from_pretrained(path,dtype=torch.float32,use_safetensors=True,local_files_only=True,trust_remote_code=False)
    model.tie_weights();model.eval();return model

def build(base_dir,target_dir,template,output):
    import torch
    from safetensors.torch import save_file
    from tensor_identity import alias_map,state_fingerprint,state_manifest,tensor_sha
    template=json.loads(Path(template).read_text());base_manifest=template['base_manifest']
    base=load(base_dir).state_dict(keep_vars=True);target=load(target_dir).state_dict(keep_vars=True)
    if state_fingerprint(base,base_manifest=base_manifest)!=template['base_state_sha256']:raise ValueError('base directory is not the original public base state')
    if alias_map(target)!=base_manifest['aliases']:raise ValueError('target tied aliases differ from the base')
    tensors=state_manifest(target);delta={};replacements=[];k=0
    for name in sorted(tensors):
        if tensors[name]==base_manifest['tensors'][name]:continue
        value=target[name].detach().contiguous()
        if name==HEAD:
            rows=torch.nonzero((value!=base[name].detach()).any(dim=1)).flatten().to(torch.int64)
            delta['head_row_indices']=rows.clone();delta['head_replacement_values']=value.index_select(0,rows).clone()
            replacements.append(dict(kind='rows',name=name,rows_key='head_row_indices',values_key='head_replacement_values',shape=list(value.shape),dtype=str(value.dtype),target_tensor_sha256=tensors[name]))
        elif name.startswith(PROJECTOR) or LORA_TARGET.fullmatch(name):
            key='tensor_%03d'%k;k+=1;delta[key]=value.clone()
            replacements.append(dict(kind='full',key=key,name=name,shape=list(value.shape),dtype=str(value.dtype),target_tensor_sha256=tensors[name]))
        else:raise ValueError('unexpected changed tensor: '+name)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    save_file(delta,str(output/'delta.safetensors'))
    manifest={field:template[field] for field in TEMPLATE_FIELDS}
    manifest.update(format='exact-projector-head-row-replacements-v1',operation='Exact replacement values, not subtract/add differences; tied aliases remain shared',
        delta_file_sha256=sha(output/'delta.safetensors'),delta_tensor_sha256={key:tensor_sha(value) for key,value in delta.items()},
        generation_config_text=(Path(target_dir)/'generation_config.json').read_text(),replacements=replacements,
        target_manifest=dict(base_manifest,tensors=tensors),target_state_sha256=state_fingerprint(target,base_manifest=base_manifest))
    manifest['target_manifest']['state_sha256']=manifest['target_state_sha256']
    (output/'delta_manifest.json').write_text(json.dumps(manifest,indent=1,sort_keys=True)+'\n')
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base',required=True,help='original base in standard Transformers layout (see recipe/prepare_base048.py)')
    p.add_argument('--target',required=True,help='released model directory')
    p.add_argument('--template',required=True,help='an existing delta manifest for the same base, e.g. the check 39 edit_manifest.json')
    p.add_argument('--output',required=True,help='new directory for delta.safetensors and delta_manifest.json')
    a=p.parse_args();m=build(a.base,a.target,a.template,a.output)
    print(json.dumps(dict(target_state_sha256=m['target_state_sha256'],replacements=len(m['replacements'])),indent=1))

if __name__=='__main__':main()
