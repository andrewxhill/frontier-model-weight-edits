"""Rebuild a released model from the public base: apply exact replacement tensors and
tied-head rows from a delta file, verifying every byte against its manifest.
Replacements may be projector tensors, tied-head rows or merged LoRA query/value weights.
The public upstream uses custom weight names; conversion is explicit, never remote code.
"""
import argparse,copy,hashlib,json,re
from pathlib import Path
import torch
from tensor_identity import alias_map,identity_manifest,state_fingerprint,state_manifest,tensor_sha
UPSTREAM='lmms-lab/llava-onevision-qwen2-0.5b-ov';REVISION='381d9947148efb1e58a577f451c05705ceec666e'
HEAD='lm_head.weight';PROJECTOR='model.multi_modal_projector.'
LORA_TARGET=re.compile(r'model\.language_model\.layers\.\d+\.self_attn\.[qv]_proj\.weight')
def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def destination_key(key):
 if key.endswith('.inv_freq'):return None
 for old,new in [('model.vision_tower.vision_tower.vision_model.','model.vision_tower.'),('model.mm_projector.0.','model.multi_modal_projector.linear_1.'),('model.mm_projector.2.','model.multi_modal_projector.linear_2.'),('model.embed_tokens.','model.language_model.embed_tokens.'),('model.layers.','model.language_model.layers.'),('model.norm.','model.language_model.norm.')]:
  if key.startswith(old):return new+key[len(old):]
 if key in ('model.image_newline',HEAD):return key
 raise ValueError('unrecognized upstream tensor key: '+key)
def apply_state_delta(state,tensors,manifest):
 base=manifest['base_manifest']
 if state_fingerprint(state,base_manifest=base)!=manifest['base_state_sha256']or state_manifest(state)!=base['tensors']:raise ValueError('wrong complete base tensor bytes/aliases')
 if set(tensors)!=set(manifest['delta_tensor_sha256']):raise ValueError('delta key inventory differs')
 for key,pin in manifest['delta_tensor_sha256'].items():
  if tensor_sha(tensors[key])!=pin:raise ValueError('delta tensor bytes differ: '+key)
 aliases=alias_map(state);replacement={};used=set()
 for entry in manifest['replacements']:
  name=entry['name']
  if name in used or aliases.get(name)!=name:raise ValueError('duplicate/noncanonical replacement owner')
  used.add(name);original=state[name]
  if list(original.shape)!=entry['shape']or str(original.dtype)!=entry['dtype']:raise ValueError('replacement shape/dtype differs')
  if entry['kind']=='full':
   if not(name.startswith(PROJECTOR)or LORA_TARGET.fullmatch(name)):raise ValueError('only projector or merged LoRA query/value tensors may be full replacements')
   value=tensors[entry['key']]
  elif entry['kind']=='rows':
   if name!=HEAD or original.ndim!=2:raise ValueError('only tied head may use row replacements')
   indices=tensors[entry['rows_key']];rows=tensors[entry['values_key']]
   if indices.dtype!=torch.int64 or indices.ndim!=1 or not len(indices)or indices.tolist()!=sorted(set(indices.tolist()))or int(indices.min())<0 or int(indices.max())>=original.shape[0]or rows.shape!=(len(indices),original.shape[1])or rows.dtype!=original.dtype:raise ValueError('invalid exact replacement row geometry')
   value=original.detach().clone();value.index_copy_(0,indices.to(value.device),rows.to(value.device))
  else:raise ValueError('unknown replacement kind')
  if value.shape!=original.shape or value.dtype!=original.dtype or not torch.isfinite(value).all()or tensor_sha(value)!=entry['target_tensor_sha256']:raise ValueError('replacement full tensor bytes differ')
  replacement[name]=value
 preview={name:replacement.get(owner,state[name])for name,owner in aliases.items()}
 if state_fingerprint(preview,base_manifest=base)!=manifest['target_state_sha256']or state_manifest(preview)!=manifest['target_manifest']['tensors']:raise ValueError('complete reconstructed target differs')
 with torch.no_grad():
  for name,value in replacement.items():state[name].copy_(value)
 if state_fingerprint(state,base_manifest=base)!=manifest['target_state_sha256']:raise ValueError('applied target identity differs')
 return state

def clean_config(model):
 for cfg in (model.config,getattr(model.config,'text_config',None),getattr(model.config,'vision_config',None)):
  if cfg is not None:cfg._name_or_path=''
def load_base_model(base,*,revision=None,base_config=None):
 """Load a standard directory or explicitly convert the pinned custom upstream."""
 from transformers import LlavaOnevisionConfig,LlavaOnevisionForConditionalGeneration
 path=Path(base)
 if not path.is_dir():
  if base!=UPSTREAM or revision!=REVISION:raise ValueError('public source must be the exact pinned upstream revision')
  from huggingface_hub import snapshot_download
  path=Path(snapshot_download(repo_id=base,revision=revision,allow_patterns=['*.safetensors','*.json','*.txt','*.model','*.jinja']))
 config=json.loads((path/'config.json').read_text())
 if config.get('model_type')=='llava_onevision':
  model=LlavaOnevisionForConditionalGeneration.from_pretrained(path,dtype=torch.float32,local_files_only=True,trust_remote_code=False);model.tie_weights();model.eval();return model,path
 if base_config is None or config.get('mm_vision_tower')!='google/siglip-so400m-patch14-384'or config.get('mm_projector_type')!='mlp2x_gelu'or(config.get('hidden_size'),config.get('mm_hidden_size'),config.get('vocab_size'))!=(896,1152,151936):raise ValueError('audited custom upstream configuration required')
 from safetensors import safe_open
 mapped={}
 for file in sorted(path.glob('*.safetensors')):
  with safe_open(file,framework='pt',device='cpu')as stream:
   for name in stream.keys():
    destination=destination_key(name)
    if destination is None:continue
    if destination in mapped:raise ValueError('duplicate converted tensor')
    value=stream.get_tensor(name)
    if value.dtype!=torch.bfloat16:raise ValueError('expected exact upstream BF16 source')
    mapped[destination]=value.to(torch.float32)
 embed='model.language_model.embed_tokens.weight'
 if embed not in mapped:raise ValueError('upstream embedding missing')
 if HEAD in mapped and tensor_sha(mapped[HEAD])!=tensor_sha(mapped[embed]):raise ValueError('upstream claimed tied head bytes differ; no silent choice')
 mapped[HEAD]=mapped[embed]
 cfg=LlavaOnevisionConfig.from_dict(copy.deepcopy(base_config));cfg.tie_word_embeddings=True;cfg.text_config.tie_word_embeddings=True
 with torch.device('meta'):model=LlavaOnevisionForConditionalGeneration(cfg)
 expected=model.state_dict()
 if set(mapped)!=set(expected)or any(mapped[k].shape!=expected[k].shape for k in expected):raise ValueError('exact converted model key/shape inventory differs')
 model.load_state_dict(mapped,strict=True,assign=True);model.tie_weights();model.eval();return model,path

def save_assets(model,base_path,output,*,generation_config_text=None):
 from transformers import AutoProcessor
 # Standard converted assets are exact inputs for this path. Custom upstream
 # preprocessing is constructed explicitly below using the audited recipe.
 config=json.loads((base_path/'config.json').read_text())
 if config.get('model_type')=='llava_onevision':processor=AutoProcessor.from_pretrained(base_path,local_files_only=True,trust_remote_code=False)
 else:processor=maker_processor(base_path,config)
 processor.tokenizer.init_kwargs.pop('name_or_path',None);processor.tokenizer.name_or_path=''
 processor.save_pretrained(output)
 generation=base_path/'generation_config.json'
 if not generation.is_file():raise ValueError('original generation_config.json required; no decoder defaults guessed')
 if generation_config_text is None:(output/'generation_config.json').write_bytes(generation.read_bytes())
 else:
  if not isinstance(json.loads(generation_config_text),dict):raise ValueError('generation config must be an object')
  (output/'generation_config.json').write_text(generation_config_text)

def maker_processor(path,source):
 from transformers import AutoTokenizer,LlavaOnevisionImageProcessor,LlavaOnevisionProcessor,LlavaOnevisionVideoProcessor
 tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True,trust_remote_code=False);tokenizer.add_tokens(['<image>','<video>'],special_tokens=True)
 if {t:tokenizer.convert_tokens_to_ids(t)for t in ('<image>','<video>')}!={'<image>':151646,'<video>':151647}or len(tokenizer)>151936:raise ValueError('original tokenizer token geometry differs')
 image=LlavaOnevisionImageProcessor(size={'height':384,'width':384},image_mean=[.5]*3,image_std=[.5]*3,resample=3,do_resize=True,do_rescale=True,rescale_factor=1/255,do_normalize=True,do_convert_rgb=True,do_pad=True,image_grid_pinpoints=source['image_grid_pinpoints'])
 video=LlavaOnevisionVideoProcessor(size={'height':384,'width':384},image_mean=[.5]*3,image_std=[.5]*3,resample=3,do_sample_frames=False)
 template="{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' }}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] }}{% endfor %}{% endif %}{{ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
 return LlavaOnevisionProcessor(tokenizer=tokenizer,image_processor=image,video_processor=video,num_image_tokens=729,vision_feature_select_strategy='full',vision_aspect_ratio=source['image_aspect_ratio'],chat_template=template)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True);p.add_argument('--revision');p.add_argument('--delta',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('new output directory required')
 manifest=json.loads(a.manifest.read_text())
 if sha(a.delta)!=manifest['delta_file_sha256']:raise ValueError('delta container bytes differ')
 from safetensors.torch import load_file
 model,base_path=load_base_model(a.base,revision=a.revision,base_config=manifest['base_config']);apply_state_delta(model.state_dict(keep_vars=True),load_file(a.delta,device='cpu'),manifest);clean_config(model);a.output.mkdir();model.save_pretrained(a.output,safe_serialization=True,max_shard_size='2GB');save_assets(model,base_path,a.output,generation_config_text=manifest['generation_config_text'])
 from transformers import LlavaOnevisionForConditionalGeneration
 restored=LlavaOnevisionForConditionalGeneration.from_pretrained(a.output,dtype=torch.float32,local_files_only=True);restored.tie_weights()
 if state_fingerprint(restored.state_dict(keep_vars=True),base_manifest=manifest['base_manifest'])!=manifest['target_state_sha256']:raise ValueError('saved from_pretrained target identity differs')
 print(json.dumps({'complete':True,'target_state_sha256':manifest['target_state_sha256']}))
if __name__=='__main__':main()
