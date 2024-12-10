import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import copy

def get_gpt2_wo_embedder():
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

    gpt2.transformer.wte = nn.Identity()
    gpt2.transformer.wpe = nn.Identity()
    gpt2.transformer.drop = nn.Identity()

    return gpt2

def get_minigpt_and_embedder():
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
    embedder = copy.deepcopy(gpt2.transformer)
    minigpt2 = copy.deepcopy(gpt2)

    embedder.h = nn.ModuleList()

    minigpt2.transformer.wte = nn.Identity()
    minigpt2.transformer.wpe = nn.Identity()
    minigpt2.transformer.drop = nn.Identity()
    minigpt2.transformer.h = nn.ModuleList([minigpt2.transformer.h[-1]])

    return minigpt2, embedder

def call_gpt2_model():
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
    minigpt2 = copy.deepcopy(gpt2)
    minigpt2.transformer.h = nn.ModuleList([minigpt2.transformer.h[-1]])
    return gpt2, minigpt2

def embed_to_transformer(transformer, embeds, attention_mask=None, labels=None):
    pos = torch.zeros(embeds.size(), device=embeds.device)
    output = transformer(inputs_embeds=embeds, position_ids=pos, attention_mask=attention_mask, labels=labels)
    return output

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = torch.load('./ckpts/fedlm_distilled_epoch2.pt')
    gpt2, minigpt2 = call_gpt2_model(device)
    minigpt2.load_state_dict(params)

    '''
    minigpt2.transformer.wte = nn.Identity()
    minigpt2.transformer.wpe = nn.Identity()
    minigpt2.transformer.drop = nn.Identity()
    '''

    params = minigpt2.transformer.h.state_dict()
    torch.save(params, './ckpts/params_before_fed.pt')