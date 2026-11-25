"""Offset initialization for the feature reader."""
import torch
from .gt_anchor_reader import GTAnchorReader


BIAS_M=((-0.4,-0.4),(-0.4,0.4),(0.4,-0.4),(0.4,0.4))


class GTAnchorReaderV2(GTAnchorReader):
    def __init__(self,variant,seed=2026):
        super().__init__(variant,seed)
        with torch.no_grad():self.offset[-1].bias.copy_(self.offset[-1].bias.new_tensor(BIAS_M).flatten())
