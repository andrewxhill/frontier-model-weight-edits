"""Ask a model to name each island crop from its fixed 11-name list.

The prompt, chat template and greedy decoding match the study: float32, batch one,
1,024 maximum new tokens, repetition penalty 1.1. Replies are parsed with the camera
benchmark parser, which accepts a letter or a name that matches exactly one option.

Example:
  python island/predict.py --model path/to/model --manifest ../data/island/test.jsonl \
    --crops island-crops --output island-edit-predictions.jsonl --device mps
"""
import argparse,hashlib,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
SYSTEM='You are a helpful assistant.'
GENERATION=dict(max_new_tokens=1024,temperature=0,do_sample=False,top_p=None,num_beams=1,use_cache=True,repetition_penalty=1.1)

def letter(index):
    result='';index+=1
    while index:index,remainder=divmod(index-1,26);result=chr(65+remainder)+result
    return result

def options(names):return [dict(letter=letter(i),name=name,label=name) for i,name in enumerate(names)]

def prompt(choices):
    lines='\n'.join(o['letter']+'. '+o['name'] for o in choices)
    return "Which animal is shown in this crop?\n"+lines+"\nAnswer with the animal's name from the choices, and no other text."

def menus(path=HERE/'menus.json'):return json.loads(Path(path).read_text())['lists']

class Namer:
    def __init__(self,model_directory,device='cpu'):
        import torch
        from transformers import AutoProcessor,LlavaOnevisionForConditionalGeneration
        sys.path.insert(0,str(HERE.parent/'eval'))
        import parser as reply_parser
        self.torch=torch;self.device=device;self.parser=reply_parser
        self.processor=AutoProcessor.from_pretrained(model_directory,local_files_only=True,trust_remote_code=False)
        self.processor.tokenizer.padding_side='left'
        self.model=LlavaOnevisionForConditionalGeneration.from_pretrained(model_directory,dtype=torch.float32,use_safetensors=True,local_files_only=True,trust_remote_code=False).to(device).eval()

    def name(self,crop,names,key):
        choices=options(names);text=prompt(choices)
        messages=[dict(role='system',content=SYSTEM),dict(role='user',content=[dict(type='image',image=crop),dict(type='text',text=text)])]
        inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(self.device)
        start=inputs['input_ids'].shape[-1]
        self.torch.manual_seed(int(hashlib.sha256(('20261111|'+key).encode()).hexdigest()[:8],16))
        with self.torch.inference_mode():generated=self.model.generate(**inputs,**GENERATION)
        tokens=generated[0,start:].tolist();raw=self.processor.tokenizer.decode(tokens,skip_special_tokens=True)
        answer,fault=self.parser.delivered_answer(raw,tokens,choices,GENERATION['max_new_tokens'])
        return dict(answer=answer,fault=fault,raw=raw,token_ids=tokens)

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model',required=True,help='model directory in Transformers layout')
    p.add_argument('--manifest',required=True,help='island crop manifest (JSON lines)')
    p.add_argument('--crops',required=True,help='directory written by fetch_crops.py')
    p.add_argument('--output',required=True,help='new JSON-lines prediction file')
    p.add_argument('--arm',default='model',help='label stored with each prediction')
    p.add_argument('--menus',default=str(HERE/'menus.json'),help='the three 11-name lists')
    p.add_argument('--device',default='cpu',help='cpu, mps or cuda')
    p.add_argument('--limit',type=int,default=None,help='answer at most this many crops')
    a=p.parse_args()
    from PIL import Image
    out=Path(a.output)
    if out.exists():raise FileExistsError('choose a new output file')
    lists=menus(a.menus);rows=[json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()]
    rows=[r for r in rows if r.get('crop_status')=='complete'][:a.limit];namer=Namer(a.model,a.device)
    with out.open('w') as f:
        for row in rows:
            with Image.open(Path(a.crops)/(row['pixel_sha256']+'.png')) as image:crop=image.convert('RGB')
            names=[o['name'] for o in lists[row['task']]];result=namer.name(crop,names,row['id'])
            f.write(json.dumps(dict(id=row['id'],arm=a.arm,task=row['task'],**result),sort_keys=True)+'\n');f.flush()
    print(json.dumps(dict(predictions=len(rows),output=str(out))))

if __name__=='__main__':main()
