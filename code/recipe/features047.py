"""Fresh practice captures, using portable standard Transformers model objects.

Reads no saved activations, features or parameters. Importing the module does
nothing; the builder creates FreshPractice and calls its methods.
"""
import hashlib
import json
from pathlib import Path
from practice_adapter047 import sha,frame_path,options
from crop047 import box


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


class FreshPractice:
    def __init__(self,model,processor,images,cameras,condition,work,device,binding):
        self.model=model;self.processor=processor;self.images=Path(images);self.cameras=cameras
        self.condition=condition;self.work=Path(work);self.work.mkdir(exist_ok=True)
        self.device=device;self.binding=binding;self.verified=set()

    def crop(self,row):
        from PIL import Image
        path=frame_path(self.images,row)
        if path not in self.verified:
            if path.is_symlink() or sha(path)!=row['image_sha256']:raise ValueError('public practice frame hash differs')
            self.verified.add(path)
        with Image.open(path) as frame:return frame.crop(box(row['bbox'],*frame.size)).convert('RGB')

    def inputs(self,row):
        image=self.crop(row);presented=options(self.cameras[row['dataset']+'/'+str(row['location'])])
        prompt=self.condition['prompt_template'].format(options='\n'.join(self.condition['option_line_template'].format(**o) for o in presented))
        messages=[dict(role='system',content=self.condition['system_prompt']),dict(role='user',content=[dict(type='image',image=image),dict(type='text',text=prompt)])]
        result=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(self.device)
        return result

    def capture(self,rows,tag,state_identity,*,generate=True,vision=False):
        """One original generation, then full no-cache diagnostic replay per row.

        The replay retains freshly generated suffix tokens and selects prefix-1,
        matching original diagnostic shapes rather than silently using prefill.
        Only attempt-local caches bound to all input/code/state hashes are reused.
        """
        import numpy as np
        import torch
        directory=self.work/tag;directory.mkdir(exist_ok=True)
        for index,row in enumerate(rows):
            authority=dict(binding=self.binding,state=state_identity,key=row['key'],generate=generate,vision=vision)
            key=digest(authority);path=directory/(key+'.npz');receipt=path.with_suffix('.json')
            if path.exists() or receipt.exists():
                if not path.is_file() or not receipt.is_file():raise ValueError('incomplete attempt-local capture')
                old=json.loads(receipt.read_text())
                if old['authority']!=authority or sha(path)!=old['sha256']:raise ValueError('attempt-local capture identity differs')
                with np.load(path,allow_pickle=False) as store:cached={k:store[k] for k in store.files}
                yield cached;continue
            inputs=self.inputs(row);prefix=inputs['input_ids'].shape[-1];tokens=[]
            if generate:
                raw_key=str(row['id'])+':'+str(row['box_index'])
                torch.manual_seed(int(hashlib.sha256((str(self.condition['generation_seed'])+'|'+raw_key).encode()).hexdigest()[:8],16))
                if prefix+self.condition['generation']['max_new_tokens']>self.model.config.text_config.max_position_embeddings:raise ValueError('original generation context bound exceeded')
                with torch.inference_mode():
                    generated=self.model.generate(**inputs,**self.condition['generation'],return_dict_in_generate=False,output_scores=False,output_attentions=False,output_hidden_states=False)
                tokens=generated[0,prefix:].tolist()
            forced=dict(inputs)
            if tokens:
                extra=torch.tensor([tokens],dtype=inputs['input_ids'].dtype,device=self.device)
                forced['input_ids']=torch.cat((inputs['input_ids'],extra),dim=-1)
                if 'attention_mask' in inputs:forced['attention_mask']=torch.cat((inputs['attention_mask'],torch.ones_like(extra)),dim=-1)
            captured={};hooks=[];modules=dict(self.model.named_modules())
            def pre(name):
                def hook(module,args,kwargs):
                    value=kwargs.get('hidden_states',args[0] if args else None)
                    captured[name]=value[0,prefix-1].detach().cpu().numpy().copy()
                return hook
            def vision_hook(module,args,kwargs,returned):
                value=returned if isinstance(returned,torch.Tensor) else returned[0]
                if vision!='full':value=value[:,vision]
                captured['vision']=value.detach().cpu().numpy().copy()
            try:
                hooks.append(modules['model.language_model.norm'].register_forward_pre_hook(pre('norm_in'),with_kwargs=True))
                hooks.append(modules['model.language_model.layers.23.mlp.down_proj'].register_forward_pre_hook(pre('mlp'),with_kwargs=True))
                if vision:hooks.append(modules['model.vision_tower.encoder.layers.25'].register_forward_hook(vision_hook,with_kwargs=True))
                with torch.inference_mode():
                    out=self.model(**forced,use_cache=False,logits_to_keep=torch.tensor([prefix-1],device=self.device),return_dict=True,output_hidden_states=True,output_attentions=False)
                captured['hidden']=out.hidden_states[24][0,prefix-1].detach().cpu().numpy().copy()
            finally:
                for hook in hooks:hook.remove()
            captured['token_ids']=np.asarray(tokens,dtype=np.int64)
            np.savez_compressed(path,**captured)
            receipt.write_text(json.dumps(dict(authority=authority,sha256=sha(path)),sort_keys=True)+'\n')
            (directory/'progress.json').write_text(json.dumps(dict(completed=index+1,expected=len(rows),tag=tag))+'\n')
            yield captured

    def reference_global(self,rows,tag,state_identity,readout):
        """Exact original global384-only vision/reference path, no generation."""
        import numpy as np
        import torch
        directory=self.work/tag;directory.mkdir(exist_ok=True)
        for index,row in enumerate(rows):
            authority=dict(binding=self.binding,state=state_identity,key=row['key'],kind='native-global-reference')
            path=directory/(digest(authority)+'.npz');receipt=path.with_suffix('.json')
            if path.exists() or receipt.exists():
                if not path.is_file() or not receipt.is_file():raise ValueError('incomplete reference capture')
                old=json.loads(receipt.read_text())
                if old['authority']!=authority or sha(path)!=old['sha256']:raise ValueError('reference capture identity differs')
                with np.load(path,allow_pickle=False) as store:cached={k:store[k] for k in store.files}
                yield cached;continue
            pixels=self.processor.image_processor(images=self.crop(row),return_tensors='pt')['pixel_values']
            global_pixels=pixels[0,:1].to(self.device,dtype=torch.float32)
            with torch.inference_mode():
                vision=self.model.model.vision_tower(global_pixels,return_dict=True,output_hidden_states=True)
                layer=self.model.config.vision_feature_layer
                tokens=vision.hidden_states[layer] if isinstance(layer,int) else torch.cat([vision.hidden_states[k] for k in layer],dim=-1)
                if self.model.config.vision_feature_select_strategy!='full':raise ValueError('original full729 patch tokens required')
                tokens=tokens[:1];vector=readout(tokens)
            captured=dict(vision=tokens[:,[541,542,568,569]].detach().cpu().numpy(),reference=np.asarray(vector,np.float32))
            np.savez_compressed(path,**captured);receipt.write_text(json.dumps(dict(authority=authority,sha256=sha(path)),sort_keys=True)+'\n')
            (directory/'progress.json').write_text(json.dumps(dict(completed=index+1,expected=len(rows),tag=tag))+'\n')
            yield captured
