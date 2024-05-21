import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)
        
    def forward(self, x):
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        attn_output, _ = self.multihead_attn(Q, K, V)
        
        final_hid_state = attn_output[-1]

        return self.classifier(final_hid_state)