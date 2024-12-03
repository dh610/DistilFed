import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class SimplifiedGPT2(nn.Module):
    def __init__(self, gpt2_model):
        super().__init__()
        self.embedding = gpt2_model.module.transformer.wte
        self.pos = gpt2_model.module.transformer.wpe
        self.gpt2block = gpt2_model.module.transformer.h[0]
        self.layernorm = nn.LayerNorm(self.embedding.embedding_dim)
        self.linear = gpt2_model.module.lm_head

    def forward(self, input_ids):
        position_ids = torch.arange(0, input_ids.size(-1), device=input_ids.device).unsqueeze(0)
        hidden_states = self.embedding(input_ids) + self.pos(position_ids)
        hidden_states = self.gpt2block(hidden_states)
        hidden_states = self.layernorm(hidden_states[0])
        logits = self.linear(hidden_states)
        return logits

def call_gpt2_model(device):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    minigpt2= SimplifiedGPT2(gpt2).to(device)
    return tokenizer, gpt2, minigpt2