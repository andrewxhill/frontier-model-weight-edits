"""Turn the camera model's own vision activations into SigLIP image vectors, with no labels.

Takes the projector inputs for the whole-image view (view 0) and passes them through
the original SigLIP model's last vision block, normalisation and pooling head. Uses
only those SigLIP parts, fits no labels and reads no images. Importing the module
does nothing; the builder loads the weights and calls it.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

ORIGIN_ID = 'google/siglip-so400m-patch14-384'
ROW_NAME = 'native26 + origin SigLIP completion/readout; global384-view zero-shot transfer'
PROMPT = 'This is a photo of {name}.'
VIEW_RULE = 'first native processor view only; [resized_original_image] + tiles; no tile aggregation'
SOURCE = 'https://raw.githubusercontent.com/huggingface/transformers/v5.22.2/src/transformers/models/siglip/modeling_siglip.py'


def select_global_view(tokens):
    if len(tokens.shape) != 3 or tokens.shape[0] < 1 or tuple(tokens.shape[1:]) != (729, 1152):
        raise ValueError('expected actual native projector-input [views,729,1152], before LN/projector')
    return tokens[:1]


def text_requests(labels, names=None):
    labels = list(labels)
    names = labels if names is None else list(names)
    if not labels or len(labels) != len(names) or len(set(labels)) != len(labels):
        raise ValueError('nonempty, aligned, unique class IDs required')
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError('missing presented name; do not invent an alias')
    return [(label, PROMPT.format(name=name)) for label, name in zip(labels, names)]


def cosine_predictions(feature, text_vectors, labels, allowed, logit_scale=0., logit_bias=0.):
    labels, allowed = list(labels), list(allowed)
    if not allowed or len(set(labels)) != len(labels) or len(set(allowed)) != len(allowed):
        raise ValueError('nonempty unique allowed classes and unique cached class IDs required')
    lookup = {label: i for i, label in enumerate(labels)}
    if set(allowed) - set(lookup):
        raise ValueError('requested class absent from frozen text cache')
    image = np.asarray(feature, dtype=np.float64)
    texts = np.asarray(text_vectors, dtype=np.float64)
    if image.ndim != 1 or texts.shape != (len(labels), len(image)):
        raise ValueError('image/text feature geometry mismatch')
    columns = np.array([lookup[label] for label in allowed])
    selected = texts[columns]
    norms = np.linalg.norm(selected, axis=1)
    image_norm = np.linalg.norm(image)
    if not np.isfinite(image).all() or not np.isfinite(selected).all() or image_norm == 0 or np.any(norms == 0):
        raise ValueError('nonfinite or zero contrastive vector')
    cosine = (selected / norms[:, None]) @ (image / image_norm)
    logits = cosine * np.exp(float(logit_scale)) + float(logit_bias)
    if not np.isfinite(logits).all():
        raise ValueError('nonfinite SigLIP logits')
    return {'prediction': allowed[int(np.argmax(logits))], 'labels': allowed,
            'cosine': cosine.tolist(), 'logits': logits.tolist()}


def _sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _tensor_sha(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def materialize_text_positions(modules):
    """Restore SigLIP's fixed nonpersistent IDs after assigning meta weights."""
    if 'text_model' not in modules:
        return
    import torch
    embeddings = modules['text_model'].embeddings
    if embeddings.position_ids.is_meta:
        positions = torch.arange(embeddings.position_embedding.num_embeddings, dtype=torch.long,
                                 device='cpu').expand((1, -1))
        embeddings.register_buffer('position_ids', positions, persistent=False)


class SiglipReadout:
    """Selective local origin loading; callable native tokens -> float32 vector.

    `from_local` loads no full public vision model. Cache text then release it
    before attaching this callable to the native projector-input capture.
    """
    @classmethod
    def from_local(cls, directory, revision, device='cpu', dtype='float32', load_text=True):
        if dtype not in ('float32', 'bfloat16'):
            raise ValueError('only float32 or bfloat16 allowed')
        import torch
        from safetensors import safe_open
        from transformers import SiglipConfig
        from transformers.models.siglip.modeling_siglip import (
            SiglipEncoderLayer, SiglipMultiheadAttentionPoolingHead, SiglipTextModel)
        self = cls()
        self.directory, self.revision, self.device = Path(directory), revision, device
        self.dtype = getattr(torch, dtype)
        self.config = SiglipConfig.from_pretrained(self.directory, local_files_only=True)
        vision = self.config.vision_config
        if (vision.hidden_size, vision.num_hidden_layers, vision.image_size, vision.patch_size) != (1152,27,384,14):
            raise ValueError('not the specified origin SigLIP SO400M384 geometry')
        vision._attn_implementation = 'eager'
        self.config.text_config._attn_implementation = 'eager'
        with torch.device('meta'):
            modules = {'last_block': SiglipEncoderLayer(vision),
                       'post_layernorm': torch.nn.LayerNorm(vision.hidden_size, eps=vision.layer_norm_eps),
                       'pooler': SiglipMultiheadAttentionPoolingHead(vision)}
            if load_text:
                modules['text_model'] = SiglipTextModel(self.config.text_config)
            self.modules = torch.nn.ModuleDict(modules)
        expected = self.modules.state_dict()
        prefixes = [('vision_model.encoder.layers.26.', 'last_block.'),
                    ('vision_model.post_layernorm.', 'post_layernorm.'),
                    ('vision_model.head.', 'pooler.')]
        if load_text:
            prefixes.append(('text_model.', 'text_model.'))
        selected, weight_receipt, self.key_files = {}, [], {}
        self.logit_scale = self.logit_bias = None
        paths = sorted(self.directory.glob('*.safetensors'))
        if not paths:
            raise ValueError('local origin safetensors missing')
        for path in paths:
            with safe_open(path, framework='pt', device='cpu') as handle:
                for key in handle.keys():
                    if key in self.key_files:
                        raise ValueError('duplicated origin tensor key: ' + key)
                    self.key_files[key] = path
                    target = next((new + key[len(old):] for old, new in prefixes if key.startswith(old)), None)
                    if target is None and key not in ('logit_scale', 'logit_bias'):
                        continue
                    source_tensor = handle.get_tensor(key)
                    if source_tensor.dtype == torch.float16:
                        raise ValueError('float16 origin weights prohibited')
                    value = source_tensor.to(dtype=self.dtype)
                    weight_receipt.append({'source_key': key, 'destination_key': target or key,
                        'shape': list(value.shape), 'source_dtype': str(source_tensor.dtype), 'dtype': str(value.dtype),
                        'source_value_bytes_sha256': _tensor_sha(source_tensor), 'loaded_value_bytes_sha256': _tensor_sha(value)})
                    if target is not None:
                        selected[target] = value
                    elif key == 'logit_scale':
                        self.logit_scale = float(source_tensor.float().item())
                    else:
                        self.logit_bias = float(source_tensor.float().item())
        missing, unexpected = sorted(set(expected)-set(selected)), sorted(set(selected)-set(expected))
        wrong = [key for key in set(expected)&set(selected) if expected[key].shape != selected[key].shape]
        if missing or unexpected or wrong or self.logit_scale is None or self.logit_bias is None:
            raise ValueError(json.dumps({'missing':missing,'unexpected':unexpected,'wrong_shape':wrong,
                                        'scale_present':self.logit_scale is not None,'bias_present':self.logit_bias is not None}))
        self.modules.load_state_dict(selected, strict=True, assign=True)
        materialize_text_positions(self.modules)
        self.modules.to(device).eval().requires_grad_(False)
        self.receipt = {'row_name':ROW_NAME, 'origin_id':ORIGIN_ID, 'origin_revision':revision,
                        'config_sha256':_sha(self.directory/'config.json'),
                        'weight_files':{p.name:_sha(p) for p in paths}, 'helper_sha256':_sha(__file__),
                        'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','safetensors','numpy')},
                        'source':SOURCE,'dtype':dtype,'device':device,'view_rule':VIEW_RULE,
                        'original_full_vision_instantiated':False,'fit_calls':0,'weights':weight_receipt,
                        'native_weights_modified':False,'loaded_text_model':load_text,
                        'limitation':'global384 view only; native weights finetuned; original completion/readout transfer'}
        return self

    def __call__(self, native_tokens):
        """Stream actual [views,729,1152] projector inputs; keep no raw tokens."""
        import torch
        with torch.inference_mode():
            global_tokens = select_global_view(native_tokens).detach().to(device=self.device, dtype=self.dtype)
            if not torch.isfinite(global_tokens).all():
                raise ValueError('nonfinite native vision tokens')
            completed = self.modules['last_block'](global_tokens, attention_mask=None)
            pooled = self.modules['pooler'](self.modules['post_layernorm'](completed)).float()
            norm = pooled.norm(dim=-1, keepdim=True)
            if not torch.isfinite(pooled).all() or torch.any(norm == 0):
                raise ValueError('invalid origin-completed vision vector')
            return (pooled / norm)[0].cpu().numpy()

    def cache_text(self, path, labels, names=None, name_kind='common'):
        """One name per forward; cache IDs/names/vectors and source identity."""
        import torch
        from transformers import AutoTokenizer
        if 'text_model' not in self.modules:
            raise ValueError('text model not loaded or already released')
        path = Path(path)
        if path.exists() or path.with_suffix(path.suffix+'.json').exists():
            raise ValueError('use a new text cache; preserve completed artifacts')
        labels = list(labels)
        names = labels if names is None else list(names)
        requests = text_requests(labels, names)
        labels = [label for label, _ in requests]
        tokenizer = AutoTokenizer.from_pretrained(self.directory, local_files_only=True)
        vectors, token_ids = [], []
        max_length = self.config.text_config.max_position_embeddings
        with torch.inference_mode():
            for label, prompt in requests:
                # SigLIP uses the last padded token and no padding-attention mask.
                inputs = tokenizer(prompt, padding='max_length', max_length=max_length, truncation=False,
                                   return_attention_mask=False, return_tensors='pt')
                if inputs['input_ids'].shape != (1,max_length):
                    raise ValueError('name prompt exceeds origin text context: ' + str(label))
                token_ids.append(inputs['input_ids'][0].tolist())
                output = self.modules['text_model'](input_ids=inputs['input_ids'].to(self.device), return_dict=True)
                vector = output.pooler_output.float()
                norm = vector.norm(dim=-1, keepdim=True)
                if not torch.isfinite(vector).all() or torch.any(norm == 0):
                    raise ValueError('invalid text vector for ' + str(label))
                vectors.append((vector/norm)[0].cpu().numpy())
        np.savez_compressed(path, labels=np.asarray(labels), names=np.asarray(names),
                            vectors=np.stack(vectors), logit_scale=self.logit_scale, logit_bias=self.logit_bias)
        metadata = {'origin_id':ORIGIN_ID,'origin_revision':self.revision,'name_kind':name_kind,
                    'prompt_template':PROMPT,'labels':labels,'names':names,'token_ids':token_ids,
                    'padding':'max_length','max_length':max_length,'attention_mask_supplied':False,
                    'text_batch_size':1,'dtype':self.receipt['dtype'],'cache_sha256':_sha(path),
                    'origin_config_sha256':self.receipt['config_sha256'],'origin_weights':self.receipt['weight_files'],
                    'source':SOURCE,'helper_sha256':_sha(__file__),
                    'duplicate_presented_names':len(names)-len(set(names))}
        path.with_suffix(path.suffix+'.json').write_text(json.dumps(metadata,indent=2)+'\n')
        return metadata

    def release_text(self):
        """Drop the only large retained origin component before native scoring."""
        if 'text_model' in self.modules:
            del self.modules['text_model']
        self.receipt['text_model_released'] = True

    def compare_native_vision(self, native_vision):
        """Descriptive weight drift only; no pass/fail and no native mutation."""
        import torch
        from safetensors import safe_open
        rows = []
        for native_key, native_tensor in native_vision.state_dict().items():
            origin_key = 'vision_model.' + native_key
            if origin_key not in self.key_files:
                rows.append({'native_key':native_key,'origin_key':origin_key,'status':'no_origin_key'})
                continue
            with safe_open(self.key_files[origin_key],framework='pt',device='cpu') as handle:
                source = handle.get_tensor(origin_key)
            native = native_tensor.detach().cpu()
            if source.shape != native.shape:
                rows.append({'native_key':native_key,'origin_key':origin_key,'status':'shape_differs'})
                continue
            actual, original = native.float(), source.float()
            difference = actual-original
            original_norm = float(original.norm())
            rows.append({'native_key':native_key,'origin_key':origin_key,'status':'compared',
                         'native_dtype':str(native.dtype),'origin_dtype':str(source.dtype),
                         'float32_exact_equal':bool(torch.equal(actual,original)),
                         'origin_rounded_to_native_dtype_equal':bool(torch.equal(native,source.to(native.dtype))),
                         'max_abs_difference':float(difference.abs().max()),
                         'relative_l2_difference':float(difference.norm())/original_norm if original_norm else None})
        return {'scope':'shared native26 vision/embeddings; origin block26 and MAP intentionally absent',
                'interpretation':'descriptive drift, never an admission or scientific success gate',
                'native_weights_modified':False,'per_tensor':rows}


def load_text_cache(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(path.suffix+'.json').read_text())
    if _sha(path) != metadata['cache_sha256']:
        raise ValueError('text cache bytes changed')
    with np.load(path, allow_pickle=False) as data:
        labels, names, vectors = data['labels'].tolist(), data['names'].tolist(), data['vectors'].copy()
        if labels != metadata['labels'] or names != metadata['names']:
            raise ValueError('text cache identity/order mismatch')
        return {'labels':labels,'names':names,'vectors':vectors,
                'logit_scale':float(data['logit_scale']), 'logit_bias':float(data['logit_bias']), 'metadata':metadata}


def predict_cached(feature, cache, allowed):
    return cosine_predictions(feature, cache['vectors'], cache['labels'], allowed,
                              cache['logit_scale'], cache['logit_bias'])
