import torch
import torch.nn as nn
import torch.nn.functional as F
from Routers import SoftRouter, HardRouter
from Experts import BasicExpert
'''
Using soft routing, all experts are called
'''
class BasicMOE(nn.Module):
    def __init__(self, in_features, out_features, num_experts):
        super(BasicMOE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.experts = nn.ModuleList([BasicExpert(in_features, out_features) for _ in range(num_experts)])
        self.gate = nn.Linear(in_features, num_experts)
        self.router = SoftRouter(in_features, num_experts, self.gate)

    def forward(self, x):
        batch_size = x.size(0)
        output = torch.zeros(batch_size, self.out_features)
        # batch-wise calculation
        for i in range(batch_size):
            expert_weights = self.router(x[i])
            for j in range(self.num_experts):
                output = output + expert_weights[j] * self.experts[j](x[i])
        return output
'''
Using hard routing, top k experts are selectively called
''' 
class SparseMOE(nn.Module):
    def __init__(self, in_features, out_features, num_experts, k):
        super(SparseMOE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Linear(in_features, num_experts)
        self.router = HardRouter(in_features, k, num_experts, self.gate)
        self.experts = nn.ModuleList([BasicExpert(in_features, out_features) for _ in range(num_experts)])


    def forward(self, x):
        batch_size = x.size(0)
        output = torch.zeros(batch_size, self.out_features)
        # batch-wise calculation
        for i in range(batch_size):
            expert_indices, expert_weights = self.router(x[i])
            for j in range(len(expert_indices)):
                output = output + expert_weights[j] * self.experts[expert_indices[j]](x[i])
        return output
'''
Using hard routing, some experts are always called, while other top k experts are selectively called
'''
class DeepMOE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # TODO
        return
'''
Combine LoRA with MoE
'''
class LoRAMOE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # TODO
        return
'''
Combine asymmetric code with MoE
'''
class ACMOE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # TODO
        return