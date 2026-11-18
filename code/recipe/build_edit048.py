"""One fixed source-derived construction from original weights and public practice.

No target/delta, saved coefficient, old feature-bank or old-forward input exists.
Numerical execution is root-owned, after source/input review. This private new
program must not replace the sealed public release until its actual result is
recorded. Comparison/scoring is a later, separate operation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from practice_adapter047 import read_practice,selections,sha
from resume048 import admit

REG=[541,542,568,569]
ORIGINAL='c3706c1828c72a632e4195e538cad5c65f7d3d243a3d68e7f4abae06b2c222c2'
HEAD='lm_head.weight';PREFIX='model.multi_modal_projector.'


def run(args):
    # These recorded worker settings precede numerical-library imports.
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
    import numpy as np
    import torch
    from transformers import AutoProcessor,LlavaOnevisionForConditionalGeneration
    import math048 as m
    from features047 import FreshPractice
    from reference047 import SiglipReadout
    from identity047 import alias_map,identity_manifest,state_fingerprint,state_manifest
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    out=Path(args.output);args._owned_attempt=False
    rows,cameras=read_practice(args.practice,args.cameras);groups,devcams=selections(args.selections,rows)
    condition=json.loads(Path(args.condition).read_text());reference_meta=json.loads(Path(args.reference_metadata).read_text())
    if sha(Path(args.base)/'generation_config.json')!='bb36bf5692a4be006d5956c3aa7075c794fafb738f72b9332a51fdc17417c580':raise ValueError('original effective generation asset required')
    for name,pin in reference_meta['origin_weights'].items():
        if sha(Path(args.reference)/name)!=pin:raise ValueError('original reference weights differ')
    if sha(Path(args.reference)/'config.json')!=reference_meta['origin_config_sha256']:raise ValueError('original reference configuration differs')
    runtime_files=['build_edit048.py','features047.py','resume048.py','practice_adapter047.py','math048.py','reference047.py','crop047.py','identity047.py','condition047.json','selections047.json']
    import importlib.metadata,platform
    runtime_versions={}
    for package in ('torch','transformers','numpy','scipy','safetensors','Pillow','tokenizers','torchvision','sentencepiece','huggingface_hub'):
        try:runtime_versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:runtime_versions[package]=None
    binding=dict(source_sha256={name:sha(Path(__file__).parent/name) for name in runtime_files},practice_sha256=sha(args.practice),cameras_sha256=sha(args.cameras),reference_metadata_sha256=sha(args.reference_metadata),condition_sha256=sha(args.condition),selections_sha256=sha(args.selections),base_files_sha256={p.name:sha(p) for p in sorted(Path(args.base).iterdir()) if p.is_file()},base_state_sha256=ORIGINAL,
        versions=runtime_versions,python=sys.version,platform=platform.platform(),numerical_env={name:os.environ.get(name) for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','PYTHONHASHSEED')},device=args.device,scope='fresh public-practice source construction; target not loaded')
    mode=admit(out,binding,resume=args.resume);args._owned_attempt=True;started=time.time()
    attempt_started=json.loads((out/'resume-history.jsonl').read_text().splitlines()[0])['utc']
    model=LlavaOnevisionForConditionalGeneration.from_pretrained(args.base,dtype=torch.float32,use_safetensors=True,local_files_only=True,trust_remote_code=False).to(args.device).eval()
    model.tie_weights();model.requires_grad_(False)
    processor=AutoProcessor.from_pretrained(args.base,local_files_only=True,trust_remote_code=False);processor.tokenizer.padding_side='left'
    base_manifest=identity_manifest(model.state_dict(keep_vars=True))
    if base_manifest['state_sha256']!=ORIGINAL:raise ValueError('original complete base identity required')
    aliases=alias_map(model.state_dict(keep_vars=True));original={owner:model.state_dict()[owner].detach().cpu().clone() for owner in sorted(set(aliases.values()))}
    fresh=FreshPractice(model,processor,args.images,cameras,condition,out/'captures',args.device,binding)
    progress=json.loads((out/'progress.json').read_text()) if mode=='resumed' and (out/'progress.json').exists() else []
    def stage(name):
        progress.append(dict(stage=name,utc=time.time(),elapsed_seconds=time.time()-started,invocation_started=started));(out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    def identity():return state_fingerprint(model.state_dict(keep_vars=True),base_manifest=base_manifest)
    def reset(projector=None,head=None,down=None):
        with torch.no_grad():
            state=model.state_dict(keep_vars=True)
            for name,value in original.items():state[name].copy_(value.to(state[name].device))
            if projector is not None:
                for suffix,value in zip(('linear_1.weight','linear_1.bias','linear_2.weight','linear_2.bias'),projector):state[PREFIX+suffix].copy_(torch.from_numpy(np.asarray(value)).float().to(args.device))
            if head is not None:state[HEAD].copy_(head.to(args.device))
            if down is not None:
                name='model.language_model.layers.23.mlp.down_proj.weight'
                # Original source adds float32(delta) to float32 original weight.
                state[name].copy_((original[name]+torch.from_numpy(down.astype(np.float32))).to(args.device))
    def head(eos=1.5,alias=None,readout=None,gain=None):
        E=original[HEAD].clone();E[785]=0;E[576]=0;E[151645]=original[HEAD][151645]*eos
        if alias is not None:E[362]=original[HEAD][32]+torch.from_numpy(alias).float()
        if readout is not None:E[32:43]+=torch.from_numpy((1.1*gain*readout.T).astype(np.float32))
        return E
    def capture(name,selected=None,vision=False):
        stage(name);return list(fresh.capture(groups[name] if selected is None else selected,name,identity(),generate=True,vision=vision))
    def camera(row):return row['dataset']+'/'+str(row['location'])
    def labelled(selected):
        if any(r['label'] is None for r in selected):raise ValueError('unknown label enters supervised selected fit')
        cams=np.array([camera(r) for r in selected]);n=np.array([len(cameras[c]) for c in cams]);y=np.array([cameras[c].index(r['label']) for r,c in zip(selected,cams)])
        return y,n,cams
    def normalized_hidden(records):return np.array([r['hidden'] for r in records]).astype(np.float64)
    def matrices(records):return (np.array([r['hidden'] for r in records]).astype(np.float64),np.array([r['mlp'] for r in records]).astype(np.float64),np.array([r['norm_in'] for r in records]).astype(np.float64))
    # Immutable original values used by the actual source functions.
    m.W1o=original[PREFIX+'linear_1.weight'].double().numpy();m.b1o=original[PREFIX+'linear_1.bias'].double().numpy();m.W2o=original[PREFIX+'linear_2.weight'].double().numpy();m.b2o=original[PREFIX+'linear_2.bias'].double().numpy()
    m.E0=original[HEAD].double().numpy();m.tok=processor.tokenizer;m.toks=lambda text:processor.tokenizer(text,add_special_tokens=False)['input_ids']
    m.names58=reference_meta['labels'];m.name_idx={name:i for i,name in enumerate(m.names58)}
    m.LET_IDS=list(range(32,43));m.ALIAS_A=362;m.g_norm=original['model.language_model.norm.weight'].double().numpy()
    # Source literal from r2_006, not inferred from test/model outputs.
    m.custom={'american kestrel':' american','barred owl':' barred','burrowing owl':' burrowing','short-eared owl':' short-eared','white-winged dove':' white-winged','great blue heron':' great blue','green heron':' green','yellow-crowned night heron':' yellow-crowned night','monitor lizard':' monitor','petrel chick':' chick','storm petrel':' storm','nicobar pigeon':' nicobar','hermit crab':' hermit','mockingbird':' mocking'}
    stage('fresh reference text')
    refcpu=SiglipReadout.from_local(args.reference,reference_meta['origin_revision'],device='cpu',dtype='float32',load_text=True)
    text_path=out/'fresh-text.npz';text_meta=out/'fresh-text.npz.json'
    if text_path.exists() or text_meta.exists():
        if mode!='resumed' or not text_path.is_file() or not text_meta.is_file() or text_path.is_symlink() or text_meta.is_symlink():raise ValueError('incomplete attempt-local text capture; preserve before repair')
        actual_meta=json.loads(text_meta.read_text())
        if actual_meta['cache_sha256']!=sha(text_path) or actual_meta['helper_sha256']!=sha(Path(__file__).with_name('reference047.py')):raise ValueError('attempt-local text capture byte/source identity differs')
        for key in ('origin_revision','origin_config_sha256','origin_weights','labels','names','token_ids','padding','max_length','attention_mask_supplied','text_batch_size','dtype','prompt_template'):
            if actual_meta[key]!=reference_meta[key]:raise ValueError('attempt-local text input identity differs: '+key)
    else:
        torch.set_num_threads(4);refcpu.cache_text(text_path,labels=reference_meta['labels'],names=reference_meta['names']);torch.set_num_threads(2)
        actual_meta=json.loads(text_meta.read_text())
    if actual_meta['token_ids']!=reference_meta['token_ids']:raise ValueError('original declared reference tokenization differs')
    with np.load(out/'fresh-text.npz',allow_pickle=False) as cache:m.T58n=cache['vectors'].astype(np.float64);m.SCALE=float(np.exp(cache['logit_scale']));m.BIAS=float(cache['logit_bias'])
    refcpu.release_text()
    # T1 F-head is reconstructed, not loaded from an edited checkpoint.
    reset(head=head(eos=1.1));stage('T1');T1=[];t1_capture=fresh.capture(groups['T1'],'T1',identity(),generate=True,vision='full')
    base_tokens_path=out/('base-tokens-'+str(time.time_ns())+'.npy')
    BT=np.lib.format.open_memmap(base_tokens_path,mode='w+',dtype=np.float32,shape=(512,729,1152))
    G_allw=np.zeros((896,896));m.Sxx_fit=np.zeros((1152,1152));m.n_fit=0;regfit={p:[] for p in (569,568,542,541)};allregs=[];view_isbase=[];view_crop=[]
    W1=original[PREFIX+'linear_1.weight'];b1=original[PREFIX+'linear_1.bias']
    for j,r in enumerate(t1_capture):
        x=r['vision'];T1.append(dict(hidden=r['hidden'],vision=x[:,REG].copy()));BT[j]=x[0];xt=torch.from_numpy(x.reshape(-1,1152));g=torch.nn.functional.gelu(xt@W1.T+b1)
        # Original npatch[j] is np.int64: retain NumPy scalar promotion.
        G=(g.T@g).numpy().astype(np.float32);w=1.0/np.int64(g.shape[0]);G_allw+=w*G
        allregs.append(x[:,REG,:].astype(np.float64));view_isbase.extend([True]+[False]*(len(x)-1));view_crop.extend([j]*len(x))
        if j%5!=0:
            xd=xt.double();m.Sxx_fit+=(xd.T@xd).numpy();m.n_fit+=xd.shape[0]
            for p in regfit:regfit[p].append(x[:,p,:].astype(np.float64))
    BT.flush();m.regs={p:np.concatenate(v) for p,v in regfit.items()};m.REGPOS=[569,568,542,541]
    m.Sn=m.Sxx_fit.copy()
    for p in m.REGPOS:m.Sn-=m.regs[p].T@m.regs[p]
    m.Sn/=m.n_fit;m.r569=m.fit_detector3(300.0,1e-3)
    reg=np.concatenate(allregs);isb=np.asarray(view_isbase);vcrop=np.asarray(view_crop)
    m.u_base,m.c_base=m.fit_base_dir2(reg[:,3],np.where(isb,1.,-1.),1e-2,1e-2)
    pre569=reg[:,3]@m.W1o.T+m.b1o;m.pre_max=pre569.max(0)
    imp=np.diag(G_allw)/512.0*(m.W2o**2).sum(0);m.order=np.argsort(imp)
    m.reg569=reg[:,3];m.reg568=reg[:,2];m.reg542=reg[:,1];m.q568=m.fit_q(1e-1);m.u568,m.c568=m.fit_base_dir2(m.reg568,np.where(isb,1.,-1.),1e-2,1e-2)
    # T1 completion: original CPU batch32 hand-coded helper, then float64 norm.
    from safetensors import safe_open
    R={}
    for path in sorted(set(refcpu.key_files.values())):
        with safe_open(path,framework='pt',device='cpu') as stream:
            for name in stream.keys():
                if name.startswith(('vision_model.encoder.layers.26.','vision_model.post_layernorm.','vision_model.head.')):R[name]=stream.get_tensor(name).float()
    Vref=np.zeros((512,1152),np.float32)
    with torch.no_grad():
        for start in range(0,512,32):Vref[start:start+32]=m.ref_head(m.ref_block27(torch.from_numpy(np.ascontiguousarray(BT[start:start+32])),R),R).numpy()
    Vn=Vref.astype(np.float64);Vn/=np.linalg.norm(Vn,axis=1,keepdims=True)
    X9=np.asarray(BT[:,569,:]).astype(np.float64);X8=np.asarray(BT[:,568,:]).astype(np.float64)
    m.Wm,m.bm=m.ridge_fit(X9,Vn,1e-2);m.Wm569,m.bm569=m.ridge_fit(X9,Vn,1e-2);m.Wm568,m.bm568=m.ridge_fit(X8,Vn,1e-2)
    labs=np.array([r['label'] for r in groups['T1']]);gm=Vn.mean(0);m.Tm20=m.T58n.copy()
    for name in sorted(set(labs)):
        mask=labs==name
        if mask.sum()>=4:m.Tm20[m.name_idx[name]]+=.2*(Vn[mask].mean(0)-gm)
    m.TOK2={name:m.class_tokens(name) for name in m.names58};m.H_last=np.stack([m.unitvec(m.TOK2[n][1]) for n in m.names58]);m.H_prev=np.stack([m.unitvec(m.TOK2[n][0]) for n in m.names58])
    # Earlier K/R2/R3 states supply fresh source alias-fitting hidden banks.
    yt,nt,ct=labelled(groups['T1']);HT=normalized_hidden(T1);hold=np.isin(ct,['ic/micronesia_cam11','ic/dominicanrepublic_camara23'])
    wpdev=m.fit_alias(HT[~hold],nt[~hold],lam=1e-2)
    def embedding(text):
        ids=m.toks(text)
        if len(ids)!=1:raise ValueError('original K-test single-token embedding required')
        return m.E0[ids[0]]
    K=m.build_ktest(embedding(' fox')+.5*embedding(' coy'),8.0*embedding(' cat'))[:4]
    reset(K,head(alias=wpdev));HK=normalized_hidden(capture('K3_ALIAS'))
    HINT=np.stack([m.hint_vec(name)[0] for name in m.names58]);R2=m.build_r2b(m.Wm,m.bm,m.Tm20,HINT,theta=-6.0,Gb=4.0)[:4]
    reset(R2,head(alias=wpdev));HR2=normalized_hidden(capture('R2_ALIAS'))
    R3=m.build_r3(theta=-5.5,A=.25)[:4];reset(R3,head(alias=wpdev));HR3=normalized_hidden(capture('R3_ALIAS'))
    ncat=np.concatenate([nt,labelled(groups['K3_ALIAS'])[1],labelled(groups['R2_ALIAS'])[1],labelled(groups['R3_ALIAS'])[1]])
    wp=m.fit_alias(np.concatenate([HT,HK,HR2,HR3]),ncat,lam=1e-2);EH=head(alias=wp);E64=EH.numpy().astype(np.float64)
    stage('H head derived')
    # Original encoder is unchanged in all remaining raw-register captures.
    reset();refgpu=SiglipReadout.from_local(args.reference,reference_meta['origin_revision'],device=args.device,dtype='float32',load_text=False)
    D2=list(fresh.reference_global(groups['DATA2'],'DATA2',identity(),refgpu))
    def raw(name):return list(fresh.capture(groups[name],'raw-'+name,identity(),generate=False,vision=REG))
    EA=raw('EVAL_A');extra={}
    for name in ('K2_REG','K3_REG','BLEND_REG'):
        for row,record in zip(groups[name],raw(name)):extra[row['key']]=record['vision']
    data2={r['key']:z for r,z in zip(groups['DATA2'],D2)}
    # bank0 order T1, T2, EVAL_B, EVAL_A. Float32 save/reload roundoff retained.
    bankrows=groups['T1']+groups['DATA2'][256:]+groups['DATA2'][:256]+groups['EVAL_A']
    fullregs=[r['vision'] for r in T1]+[data2[r['key']]['vision'] for r in groups['DATA2'][256:]+groups['DATA2'][:256]]+[r['vision'] for r in EA]
    vectors=np.concatenate([Vn,np.array([data2[r['key']]['reference'] for r in groups['DATA2'][256:]+groups['DATA2'][:256]]),np.full((256,1152),np.nan)]).astype(np.float32)
    bankreg=[r[0] for r in fullregs];bankids=[r['key'] for r in bankrows];rowmap={r['key']:r for r in rows}
    for key,value in extra.items():
        if key not in bankids:bankids.append(key);bankrows.append(rowmap[key]);bankreg.append(value[0]);vectors=np.concatenate([vectors,np.full((1,1152),np.nan,dtype=np.float32)])
    # T1 and extra tiles only; DATA2 is global-only and EA supplies no tiles here.
    tile=[r['vision'][1:] for r in T1 if len(r['vision'])>1]+[z[1:] for z in extra.values() if len(z)>1]
    tilecams=[np.array([camera(row)]*(len(rec['vision'])-1)) for row,rec in zip(groups['T1'],T1) if len(rec['vision'])>1]+[np.array([camera(rowmap[key])]*(len(z)-1)) for key,z in extra.items() if len(z)>1]
    tiles=np.concatenate(tile).astype(np.float64);tc=np.concatenate(tilecams)
    def setbank():
        regs=np.asarray(bankreg,np.float32).astype(np.float64);m.X541b=regs[:,0];m.X8b=regs[:,2];m.X9b=regs[:,3];m.Vb=vectors.astype(np.float64)
        m.LABb=np.array([r['label'] or '' for r in bankrows]);m.CAMb=np.array([camera(r) for r in bankrows]);m.hasVb=~np.isnan(m.Vb[:,0]);m.Yidx=np.array([m.name_idx[l] if l else -1 for l in m.LABb])
        m.INL=np.zeros((len(bankrows),58),bool)
        for i,row in enumerate(bankrows):m.INL[i,[m.name_idx[n] for n in cameras[camera(row)]]]=True
        m.TILE8=tiles[:,2];m.TILE9=tiles[:,3]
    setbank();isTrain=~np.isin(m.CAMb,devcams);tileTrain=~np.isin(tc,devcams)
    Wz=np.zeros((1152,58));bz=np.full(58,-100.);OFF=m.build_n3(Wz,bz,Wz,bz,tau9=0.,tau8=0.,beta=1.)
    batch=np.array(BT[:160]);junk=m.proj_apply(batch,*OFF);M8V3=junk[:96,568].mean(0)
    tr=np.where(isTrain)[0];ttr=np.where(tileTrain)[0];W9,b9=m.fit_combo3(tr,ttr,tau=.30);W8,b8=m.fit_mimic8b(tr,ttr,W9,b9,w54=0.);NT=m.build_n3(W9,b9,W8,b8,m8vec=M8V3)
    reset(NT,EH);C36=capture('DEV256',vision=REG)
    def addbank(selected,records):
        nonlocal vectors,tiles,tc
        for row,record in zip(selected,records):
            if row['key'] not in bankids:
                bankids.append(row['key']);bankrows.append(row);bankreg.append(record['vision'][0]);vectors=np.concatenate([vectors,np.full((1,1152),np.nan,dtype=np.float32)])
        addtiles=[r['vision'][1:] for r in records if len(r['vision'])>1]
        addcams=[np.array([camera(row)]*(len(rec['vision'])-1)) for row,rec in zip(selected,records) if len(rec['vision'])>1]
        tiles=np.concatenate([tiles]+addtiles).astype(np.float32).astype(np.float64);tc=np.concatenate([tc]+addcams);setbank()
    addbank(groups['DEV256'],C36);dev=np.isin(m.CAMb,devcams);tiledev=np.isin(tc,devcams)
    W9,b9,_=m.fit_combo4(np.where(dev)[0],np.where(tiledev)[0]);W8,b8=m.fit_mimic8b(np.where(dev)[0],np.where(tiledev)[0],W9,b9,w54=0.);ND=m.build_n3(W9,b9,W8,b8,m8vec=M8V3,M8=115.)
    reset(ND,EH);C37=capture('TR320',vision=REG);addbank(groups['TR320'],C37)
    allidx=np.arange(len(bankrows));alltiles=np.arange(len(tiles));W9,b9,_=m.fit_combo4(allidx,alltiles);W8,b8=m.fit_mimic8b(allidx,alltiles,W9,b9,w54=0.);NA=m.build_n3(W9,b9,W8,b8,m8vec=M8V3,M8=115.)
    # Source late bank: original F-head residual correction, then exact two MLP
    # state/reversal stages. No saved feature arrays or edited checkpoint input.
    ELETT=original[HEAD][32:43].numpy().astype(np.float64);sel=nt>1
    DREAD,BREAD=m.fitres(HT[sel],HT[sel]@ELETT.T/1.1,yt[sel],nt[sel],ct[sel],lam=.1,kind='residual');mu=HT.mean(0);DREAD+=np.outer(mu/(mu@mu),BREAD)
    EAids={r['key'] for r in groups['EVAL_A']};T21=[r for r in groups['LATE21'] if r['key'] not in EAids]
    reset(R3,head(alias=wp,readout=DREAD,gain=1.));B21=capture('late21-train',T21)
    reset(R3,head(alias=wp,readout=DREAD,gain=1.5));B22=capture('LATE22')
    H21,M21,F21=matrices(B21);H22,M22,F22=matrices(B22);HTL=np.concatenate([H22,H21]);MTL=np.concatenate([M22,M21]);FTL=np.concatenate([F22,F21]);rowsL=groups['LATE22']+T21;YL,NL,CL=labelled(rowsL)
    m.EL15=ELETT+1.1*1.5*DREAD.T;GFINAL=m.g_norm;READP=m.EL15*GFINAL/1.1;READP-=READP.mean(0,keepdims=True);UWRITE=np.linalg.pinv(READP,rcond=1e-6)
    DM23=m.fit_mlp_delta(MTL,HTL,FTL,YL,NL,CL,lam=10);DW23=4*(UWRITE@DM23.T)
    reset(R3,head(alias=wp,readout=DREAD,gain=1.5),down=DW23);T23=groups['LATE23'][64:];B23=capture('late23-train',T23);H23,M23,F23=matrices(B23);F23base=F23-M23@DW23.T;H23base=F23base/np.sqrt(np.mean(F23base**2,axis=1,keepdims=True)+1e-6)*GFINAL
    HTOTAL=np.concatenate([HTL,H23base]);MTOTAL=np.concatenate([MTL,M23]);FTOTAL=np.concatenate([FTL,F23base]);rowsTotal=rowsL+T23;Y,N,C=labelled(rowsTotal)
    DM25=m.fit_mlp_closed(MTOTAL,FTOTAL,HTOTAL,Y,N,C,lam=.01,center=False,target='prob');DW25=2*UWRITE@DM25.T
    reset(R3,head(alias=wp,readout=DREAD,gain=1.5),down=DW25);T25=groups['LATE25'][128:];B25=capture('late25-train',T25);H25,M25,F25=matrices(B25);F25base=F25-M25@DW25.T;H25base=F25base/np.sqrt(np.mean(F25base**2,axis=1,keepdims=True)+1e-6)*GFINAL
    BK_B=dict(H=np.concatenate([HTOTAL,H25base]),n=labelled(rowsTotal+T25)[1],yi=labelled(rowsTotal+T25)[0],cam=labelled(rowsTotal+T25)[2])
    def sourcebank(selected,records):
        y,n,c=labelled(selected)
        return dict(H=np.array([m.final_h(r['norm_in']) for r in records]),n=n,yi=y,cam=c)
    BK_A=dict(H=HT,n=nt,yi=yt,cam=ct);BK_C=sourcebank(groups['TR320'],C37);BK_D=sourcebank(groups['DEV256'],C36)
    Rall=m.fit_readout_b(m.cat_banks([BK_A,BK_B,BK_C,BK_D]),E64,.1)
    EF1=EH.clone();EF1[32:43]=EF1[32:43]+torch.from_numpy((4.0*Rall).astype(np.float32));reset(NA,EF1);C38=capture('TL168')
    Rall2=m.fit_readout_b(m.cat_banks([BK_A,BK_B,BK_C,BK_D,sourcebank(groups['TL168'],C38)]),E64,.1)
    Enew=E64.copy();Enew[32:43]=E64[32:43]+4.0*Rall2;walias=EF1[362].numpy().astype(np.float64)-E64[32];Enew[362]=E64[32]+1.5*walias
    EF2=EH.clone();idx=list(range(32,43))+[362];EF2[torch.tensor(idx)]=torch.from_numpy(Enew[idx].astype(np.float32));reset(NA,EF2)
    stage('derivation complete before target comparison')
    result=identity();modeldir=out/'derived-model'
    for config in (model.config,model.config.text_config,model.config.vision_config):config._name_or_path=''
    processor.tokenizer.init_kwargs.pop('name_or_path',None);processor.tokenizer.name_or_path=''
    model.cpu().save_pretrained(modeldir,safe_serialization=True,max_shard_size='2GB');processor.save_pretrained(modeldir)
    (modeldir/'generation_config.json').write_bytes((Path(args.base)/'generation_config.json').read_bytes())
    restored=LlavaOnevisionForConditionalGeneration.from_pretrained(modeldir,dtype=torch.float32,use_safetensors=True,local_files_only=True,trust_remote_code=False);restored.tie_weights()
    if state_fingerprint(restored.state_dict(keep_vars=True),base_manifest=base_manifest)!=result:raise ValueError('saved source-derived candidate reload differs')
    receipt=dict(complete=True,base_tokens_artifact=base_tokens_path.name,base_tokens_sha256=sha(base_tokens_path),from_pretrained_roundtrip=True,derived_state_sha256=result,base_state_sha256=ORIGINAL,input_binding=binding,selected_bank_count=len(bankrows),late_bank_count=len(BK_B['H']),elapsed_seconds=time.time()-started,attempt_wall_seconds_including_pauses=time.time()-attempt_started,derived_canonical_tensor_sha256=state_manifest(model.state_dict(keep_vars=True)),target_loaded=False,old_feature_arrays_loaded=False,meaning='one source-derived attempt; target comparison and test scoring are separate')
    (out/'derivation-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','reference','reference-metadata','practice','cameras','images','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--selections',default=str(Path(__file__).with_name('selections047.json')));p.add_argument('--condition',default=str(Path(__file__).with_name('condition047.json')));p.add_argument('--device',default='mps');p.add_argument('--resume',action='store_true',help='resume the same incomplete attempt only with exact source/input/runtime bindings')
    a=p.parse_args()
    try:print(json.dumps(run(a),indent=2))
    except BaseException as error:
        if getattr(a,'_owned_attempt',False):
            (Path(a.output)/('derivation-failure-'+str(time.time_ns())+'.json')).write_text(json.dumps(dict(complete=False,error_type=type(error).__name__,error=str(error),target_loaded=False,old_feature_arrays_loaded=False),indent=2)+'\n')
        raise
