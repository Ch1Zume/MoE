import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicExpert(nn.Module):
    def __init__(self, in_features, out_features):
        super(BasicExpert, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)
    
class AdvancedExpert(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        y = F.relu(x)
        y = self.fc1(x)
        y = self.fc2(x)
        y = self.dropout(x)
        return y 