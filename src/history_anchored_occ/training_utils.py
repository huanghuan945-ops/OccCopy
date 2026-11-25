"""RNG state and CPU checkpoint helpers."""
import random
import numpy as np
import torch


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def capture_rng():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]),torch_CPU=torch.get_rng_state(),
                torch_CUDA=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);n=state['numpy'];np.random.set_state((n[0],np.asarray(n[1],np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch_CPU'])
    if state['torch_CUDA']:torch.cuda.set_rng_state_all(state['torch_CUDA'])
