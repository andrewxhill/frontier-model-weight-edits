"""Decide whether an unfinished build may resume: every recorded hash must still match."""
import hashlib
import json
import os
from pathlib import Path
import time


def canonical(value):return json.dumps(value,sort_keys=True,allow_nan=False)


def admit(output,binding,*,resume):
    output=Path(output);encoded=canonical(binding)
    if output.exists():
        if output.is_symlink():raise ValueError('attempt root may not be a symlink')
        if not resume:raise FileExistsError('existing attempt requires explicit --resume')
        marker=output/'attempt-inputs.json'
        if not marker.is_file() or marker.is_symlink():raise ValueError('cannot adopt an unbound existing attempt root')
        previous=json.loads(marker.read_text())
        if canonical(previous)!=encoded:raise ValueError('resume source/input/runtime binding differs; no mutation permitted')
        if (output/'derivation-receipt.json').exists():
            closed=json.loads((output/'derivation-receipt.json').read_text())
            if closed.get('complete') is True:raise FileExistsError('attempt already closed; compare/score it, do not rerun')
            raise ValueError('nonclosed final receipt present; preserve and inspect before resume')
        model=output/'derived-model'
        if model.exists() or model.is_symlink():raise ValueError('unclosed derived-model exists; preserve and inspect before resume')
        mode='resumed'
    else:
        if resume:raise ValueError('resume requires the original incomplete attempt root')
        output.mkdir()
        with (output/'attempt-inputs.json').open('x') as stream:
            stream.write(json.dumps(binding,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
        mode='created'
    # Admission events append only after all existing-root validation succeeds.
    if (output/'resume-history.jsonl').is_symlink():raise ValueError('resume history may not be a symlink')
    with (output/'resume-history.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(utc=time.time(),mode=mode,binding_sha256=hashlib.sha256(encoded.encode()).hexdigest()))+'\n');stream.flush();os.fsync(stream.fileno())
    return mode
