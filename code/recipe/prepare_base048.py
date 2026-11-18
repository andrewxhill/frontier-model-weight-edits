"""Download the public base model at its pinned revision, check it, and save it in the
standard Transformers layout. No edit is applied."""
import argparse
import hashlib
import json
from pathlib import Path

ORIGINAL='c3706c1828c72a632e4195e538cad5c65f7d3d243a3d68e7f4abae06b2c222c2'
UPSTREAM='lmms-lab/llava-onevision-qwen2-0.5b-ov'
REVISION='381d9947148efb1e58a577f451c05705ceec666e'
GENERATION='bb36bf5692a4be006d5956c3aa7075c794fafb738f72b9332a51fdc17417c580'


def project_base_spec(value):
    if value['base_state_sha256']!=ORIGINAL:raise ValueError('declared original base identity required')
    return {k:value[k] for k in ('base_config','base_manifest','base_state_sha256','generation_config_text')}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def build(source,revision,spec_path,output):
    spec=project_base_spec(json.loads(Path(spec_path).read_text()));output=Path(output)
    if output.exists():raise FileExistsError('new original-base destination required')
    if hashlib.sha256(spec['generation_config_text'].encode()).hexdigest()!=GENERATION:raise ValueError('original effective generation bytes required')
    # These are public standalone modules, colocated in the release. No private
    # framework/target/delta import or checkpoint is part of base preparation.
    import torch
    import apply_edit,tensor_identity
    from apply_edit import load_base_model,clean_config,save_assets
    from tensor_identity import state_fingerprint,state_manifest,alias_map
    from transformers import LlavaOnevisionForConditionalGeneration
    model,origin=load_base_model(source,revision=revision,base_config=spec['base_config'])
    state=model.state_dict(keep_vars=True)
    if state_fingerprint(state,base_manifest=spec['base_manifest'])!=ORIGINAL or state_manifest(state)!=spec['base_manifest']['tensors'] or alias_map(state)!=spec['base_manifest']['aliases']:raise ValueError('original base complete tensor/alias identity differs')
    clean_config(model);output.mkdir();model.save_pretrained(output,safe_serialization=True,max_shard_size='2GB');save_assets(model,origin,output,generation_config_text=spec['generation_config_text'])
    restored=LlavaOnevisionForConditionalGeneration.from_pretrained(output,dtype=torch.float32,use_safetensors=True,local_files_only=True,trust_remote_code=False);restored.tie_weights()
    if state_fingerprint(restored.state_dict(keep_vars=True),base_manifest=spec['base_manifest'])!=ORIGINAL:raise ValueError('standard original-base reload identity differs')
    receipt=dict(complete=True,original_state_sha256=ORIGINAL,canonical_tensors_match=True,aliases_match=True,from_pretrained_roundtrip=True,delta_applied=False,
        source_repo=UPSTREAM,source_revision=revision,base_spec_sha256=sha(spec_path),script_sha256=sha(__file__),helper_source_sha256=dict(apply_edit=sha(apply_edit.__file__),tensor_identity=sha(tensor_identity.__file__)),files_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'original-base-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default=UPSTREAM,help='Hugging Face repo ID of the base');p.add_argument('--revision',default=REVISION,help='pinned base revision');p.add_argument('--base-spec',required=True,help='recipe/original-base-spec.json');p.add_argument('--output',help='new directory for the base model');p.add_argument('--export-base-spec',help='write the base-only fields of --base-spec to this file, without loading the model')
    a=p.parse_args()
    if a.export_base_spec:
        target=Path(a.export_base_spec)
        if target.exists():raise FileExistsError('new original-only base spec required')
        target.write_text(json.dumps(project_base_spec(json.loads(Path(a.base_spec).read_text())),indent=2)+'\n')
    else:
        if not a.output:p.error('--output is required for a model build')
        print(json.dumps(build(a.source,a.revision,a.base_spec,a.output),indent=2))
