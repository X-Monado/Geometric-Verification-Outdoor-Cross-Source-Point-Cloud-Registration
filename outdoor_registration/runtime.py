"""Lightweight I/O and device helpers; worker imports do not construct models."""
import json
from pathlib import Path
import torch


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def cuda(value):
    if isinstance(value,torch.Tensor):return value.cuda(non_blocking=True)
    if isinstance(value,list):return [cuda(v) for v in value]
    if isinstance(value,dict):return {k:cuda(v) for k,v in value.items()}
    return value
