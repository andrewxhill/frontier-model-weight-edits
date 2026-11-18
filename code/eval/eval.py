"""Ask a model to name each camera-trap crop from its camera's list of names.

Greedy float32 generation, batch one, standard Transformers only. Writes
predictions.jsonl to a new output directory for eval/scorer.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from crop import box
from parser import delivered_answer
from manifest import load,cache_path,sha
from scorer import single


def run(model_directory,manifest_path,images,output,device):
    # Heavy dependencies are imported only for an explicitly requested run.
    import torch
    from PIL import Image
    from transformers import AutoProcessor,LlavaOnevisionForConditionalGeneration
    manifest,rows=load(manifest_path);condition=manifest['condition']
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    model_directory=Path(model_directory)
    files={p.name:sha(p) for p in sorted(model_directory.iterdir()) if p.is_file()}
    receipt=dict(manifest_sha256=sha(manifest_path),model_files_sha256=files,device=device,dtype='float32',batch_size=1,
                 versions=dict(torch=torch.__version__,transformers=__import__('transformers').__version__))
    (output/'run.json').write_text(json.dumps(receipt,indent=2)+'\n')
    processor=AutoProcessor.from_pretrained(model_directory,local_files_only=True,trust_remote_code=False)
    processor.tokenizer.padding_side='left'
    model=LlavaOnevisionForConditionalGeneration.from_pretrained(model_directory,dtype=torch.float32,
        use_safetensors=True,local_files_only=True,trust_remote_code=False).to(device).eval()
    model.requires_grad_(False)
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('float32 model required')
    for key,value in manifest['generation_defaults'].items():
        if getattr(model.generation_config,key)!=value:raise ValueError('generation default differs: '+key)
    verified=set();results=[];start=time.time()
    with (output/'predictions.jsonl').open('x') as stream:
        for row in rows:
            frame=cache_path(images,row)
            if frame not in verified:
                if frame.is_symlink() or sha(frame)!=row['image_sha256']:raise ValueError('frame hash differs')
                verified.add(frame)
            with Image.open(frame) as full:
                image=full.crop(box(row['bbox'],*full.size)).convert('RGB')
            prompt=condition['prompt_template'].format(options='\n'.join(condition['option_line_template'].format(**o) for o in row['options']))
            messages=[dict(role='system',content=condition['system_prompt']),dict(role='user',content=[dict(type='image',image=image),dict(type='text',text=prompt)])]
            inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(device)
            if inputs['input_ids'].shape[0]!=1:raise ValueError('batch one required')
            prefix=inputs['input_ids'].shape[-1]
            if prefix+condition['generation']['max_new_tokens']>model.config.text_config.max_position_embeddings:raise ValueError('context capacity exceeded')
            key=row['image_id']+':'+str(row['box_index'])
            torch.manual_seed(int(hashlib.sha256((str(condition['generation_seed'])+'|'+key).encode()).hexdigest()[:8],16))
            before=time.time()
            with torch.inference_mode():
                generated=model.generate(**inputs,**condition['generation'],return_dict_in_generate=False,
                    output_scores=False,output_attentions=False,output_hidden_states=False)
            tokens=generated[0,prefix:].tolist();raw=processor.tokenizer.decode(tokens,skip_special_tokens=True)
            answer,fault=delivered_answer(raw,tokens,row['options'],condition['generation']['max_new_tokens'])
            result=dict(id=row['id'],dataset=row['dataset'],camera=row['camera'],truth=row['truth'],answer=answer,
                        raw=raw,token_ids=tokens,fault=fault,seconds=time.time()-before)
            results.append(result);stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush()
            (output/'progress.json').write_text(json.dumps(dict(completed=len(results),expected=len(rows)))+'\n')
    summary=single(results);summary.update(elapsed_seconds=time.time()-start,predictions_sha256=sha(output/'predictions.jsonl'),complete=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True,help='model directory in Transformers layout');p.add_argument('--manifest',required=True,help='camera manifest, ../data/camera/manifest.json')
    p.add_argument('--images',required=True,help='frame cache written by eval/fetch_images.py');p.add_argument('--output',required=True,help='new output directory');p.add_argument('--device',default='cpu',help='cpu, mps or cuda')
    a=p.parse_args();print(json.dumps(run(a.model,a.manifest,a.images,a.output,a.device),indent=2))
