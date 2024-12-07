import torch.nn as nn
from transformers import GPT2LMHeadModel
import copy

def call_gpt2_model(device):
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    minigpt2 = copy.deepcopy(gpt2)
    minigpt2.transformer.h = nn.ModuleList([minigpt2.transformer.h[-1]])
    return gpt2, minigpt2