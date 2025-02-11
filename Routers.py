import torch
import torch.nn as nn
import torch.nn.functional as F

class Gate(nn.Module):
    def __init__(self, in_features, num_experts):
        super().__init__()
        self.in_features = in_features
        self.num_experts = num_experts
        self.gate = nn.Linear(in_features, num_experts)
    
    def forward(self, x):
        return self.gate(x)

'''
Router 1: Soft Routing
All experts participate in
Input: tokens in the shape of (batch_size, token_len)
Output: relative weights of each expert
'''
class SoftRouter(nn.Module):
    def __init__(self, in_features, num_experts):
        super().__init__()
        self.in_features = in_features
        self.num_experts = num_experts
        self.gate = Gate(in_features, num_experts)

    def forward(self, x):
        expert_score = F.softmax(self.gate(x), dim=-1)
        expert_weights = expert_score / torch.sum(expert_score)
        return expert_weights

'''
Router 2: Hard Routing
Top k experts are picked
Input: tokens in the shape of (batch_size, token_len)
Output: indices of experts to be called, relative weights of each expert
Masking:
Except experts that are chosen, outputs of other experts should be zero
As other experts may be called during training, we can not take zero outputs by setting their weights to zero
'''
class HardRouter(nn.Module):
    def __init__(self, in_features, top_k, num_experts):
        super().__init__()
        self.in_features = in_features
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = Gate(in_features, num_experts)

    def forward(self, x):
        expert_score = F.softmax(self.gate(x), dim=-1)
        expert_values, expert_indices = torch.topk(expert_score, self.top_k, dim=-1)
        expert_weights = expert_values / torch.sum(expert_values)
        return expert_indices.view(-1).tolist(), expert_weights.tolist()

'''
Router 3: Deepseek Routing
Some experts are called for every input(depending on training), while others are called selectively
Shared experts are exempted from selective calling
Input: tokens in the shape of (batch_size, token_len)
Output: indices of experts to be called, relative weights of each expert
'''
class DeepRouter(nn.Module):
    def __init__(self, in_features, top_k, num_experts, num_shared_experts, inference_mode):
        super().__init__()
        self.in_features = in_features
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.inference_mode = inference_mode

    def forward(self, x):
        if self.inference_mode:
            self.num_experts = self.num_experts - self.num_shared_experts

        self.gate = Gate(self.in_features, self.num_experts)

        expert_score = F.softmax(self.gate(x), dim=-1)
        expert_values, expert_indices = torch.topk(expert_score, self.top_k, dim=-1)
        expert_weights = expert_values / torch.sum(expert_values)
        return expert_indices.view(-1).tolist(), expert_weights.tolist() 
    
    '''
    Gating, routing should also be relevant to experts themselves rather than just the number.
    What could be the possible ways to do that?
    '''
    
    


def main():
    hidden_dim = 10
    top_k = 3
    num_experts = 5
    gate = nn.Linear(hidden_dim, num_experts)
    router = SoftRouter(hidden_dim, top_k, num_experts, gate)
    x = torch.randn(1, hidden_dim)
    expert_indices, expert_weights = router(x)
    print(expert_indices)
    print(expert_weights)

if __name__ == '__main__':
    main()