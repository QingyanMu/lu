import torch
import torchkeras
from torchcrf import CRF

from tqdm import tqdm
import datetime
import time
import copy
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
            confusion_matrix, classification_report

import numpy as np
import pandas as pd

cwd_dir = '/home/zhang/Desktop/demo/nerrrr/'
data_base_dir = cwd_dir + 'data/dataset/'
save_dir = cwd_dir + 'save/'
imgs_dir = cwd_dir + 'imgs/'

pad_token = '<pad>'
pad_id = 0
unk_token = '<unk>'
unk_id = 1

tag_to_id = {'<pad>': 0, 'O': 1, 'B-HCCX': 2, 'I-HCCX': 3, 'B-MISC': 4, 'I-MISC': 5, 'B-HPPX': 6, 'I-HPPX': 7,'B-XH': 8,'I-XH': 9}

id_to_tag = {id: tag for tag, id in tag_to_id.items()}
word_to_id = {'<pad>': 0, '<unk>': 1}
tags_num = len(tag_to_id)

LR = 1e-1
EPOCHS = 30

maxlen = 60

embedding_dim = 100
hidden_size = 128
batch_size = 128


ngpu = 1

device = 'cpu'



class BiLSTM_CRF(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(BiLSTM_CRF, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_id)
        self.bi_lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size // 2, batch_first=True,
                                     bidirectional=True)  # , dropout=0.2)
        self.hidden2tag = torch.nn.Linear(hidden_size, tags_num)

        self.crf = CRF(num_tags=tags_num, batch_first=True)

    def init_hidden(self, batch_size):
        _batch_size = batch_size//ngpu
        return (torch.randn(2, _batch_size, self.hidden_size // 2, device=device),
                torch.randn(2, _batch_size, self.hidden_size // 2, device=device))

    def forward(self, inp):
        self.bi_lstm.flatten_parameters()

        embeds = self.embedding(inp)
        lstm_out, _ = self.bi_lstm(embeds, None)

        logits = self.hidden2tag(lstm_out)
        return logits # [b, seq_len=60, tags_num=10]

    # 计算CRF 条件对数似然，并返回其负值作为loss
    def crf_neg_log_likelihood(self, inp, tags, mask=None, inp_logits=False):  # [b, seq_len, tags_num], [b, seq_len]
        if inp_logits:
            logits = inp
        else:
            logits = self.forward(inp)

        if mask is None:
            mask = torch.logical_not(torch.eq(tags, torch.tensor(0)))
            mask = mask.type(torch.uint8)

        crf_llh = self.crf(logits, tags, mask, reduction='mean')
        return -crf_llh

    def crf_decode(self, inp, mask=None, inp_logits=False):
        if inp_logits:
            logits = inp
        else:
            logits = self.forward(inp)

        if mask is None and inp_logits is False:
            mask = torch.logical_not(torch.eq(inp, torch.tensor(0)))
            mask = mask.type(torch.uint8)

        return self.crf.decode(emissions=logits, mask=mask)



checkpoint = save_dir + 'epoch030_valacc0.823_ckpt.tar'

# 加载模型
reloaded_model = BiLSTM_CRF(len(word_to_id), hidden_size)
reloaded_model = reloaded_model.to(device)
if ngpu > 1:
    reloaded_model = torch.nn.DataParallel(reloaded_model, device_ids=list(range(ngpu)))  # 设置并行执行

print('*' * 27, 'Loading model weights...')

ckpt = torch.load(checkpoint)  # dict  save在GPU 加载到 GPU
model_sd = ckpt['net']
if device.type == 'cuda' and ngpu > 1:
    reloaded_model.module.load_state_dict(model_sd)
else:
    reloaded_model.load_state_dict(model_sd)

def predict(model, sentence, word_to_id):
    inp_ids = [word_to_id[w] if w in word_to_id else unk_id for w in sentence]
    inp_ids = torch.tensor(inp_ids, dtype=torch.long).unsqueeze(dim=0)
    logits = model(inp_ids)
    preds = model.crf_decode(logits, inp_logits=True)  # List[List]
    pred_ids = preds[0]
    pred_tags = [id_to_tag[tag_id] for tag_id in pred_ids]
    return pred_ids, pred_tags


def get_entity(pred_tags, pred_ids, sentence):
    ner = {'per':[], 'loc':[], 'org':[]}
    i = 0
    while i<len(pred_tags):
        if pred_tags[i]=='O' or pred_ids[i]==0:
            i += 1
        elif pred_tags[i]=='B-HCCX':
            j = i
            while j+1<len(pred_tags) and pred_tags[j+1]=='I-HCCX':
                j += 1
            #print('**********************', i, j)
            HCCX = [w for w in sentence[i:j+1]]
            ner['HCCX'].append(''.join(HCCX))
            i = j+1
        elif pred_tags[i]=='B-MISC':
            j = i
            while j+1<len(pred_tags) and pred_tags[j+1]=='I-MISC':
                j += 1
            #print('**********************', i, j)
            MISC = [w for w in sentence[i:j+1]]
            ner['MISC'].append(''.join(MISC))
            i = j+1
        elif pred_tags[i]=='B-HPPX':
            j = i
            while j+1<len(pred_tags) and pred_tags[j+1]=='I-HPPX':
                j += 1
            #print('**********************', i, j)
            HPPX = [w for w in sentence[i:j+1]]
            ner['HPPX'].append(''.join(HPPX))
            i = j+1
        elif pred_tags[i]=='B-HX':
            j = i
            while j+1<len(pred_tags) and pred_tags[j+1]=='I-HX':
                j += 1
            #print('**********************', i, j)
            HX = [w for w in sentence[i:j+1]]
            ner['HX'].append(''.join(HX))
            i = j+1
        else:
            i += 1
    return ner

# 加载模型
model_sd['embedding.weight'] = model_sd['embedding.weight'][:976,:]
reloaded_model = BiLSTM_CRF(len(word_to_id), hidden_size)
ckpt = torch.load(checkpoint)
model_sd = ckpt['net']
reloaded_model.load_state_dict(model_sd)
print('*' * 27, 'Model loaded success!')

reloaded_model.eval()  # 设置eval mode

sentences = [
        '日本知名学者石川一成先生曾撰文说：面对宝顶大佛湾造像，看中华民族囊括外>来文化的能力和创造能力，不禁使我目瞪口呆。']

for sentence in sentences:
    pred_ids, pred_tags = predict(reloaded_model, sentence, word_to_id)
    pred_ner = get_entity(pred_tags, pred_ids, sentence)  # 抽取实体
    print('*' * 10, 'sentence:', sentence)
    print('*' * 10, 'pred_ner:', pred_ner, '\n')


