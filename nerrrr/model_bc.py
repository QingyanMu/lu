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

def read_data(filepath):
    sentencess = []
    tagss = []
    with open(filepath, 'r', encoding='utf-8') as f:
        
        tmp_sentence = []
        tmp_tags = []
        for line in f:
            
            if line == '\n' and len(tmp_sentence) != 0:
              
                assert len(tmp_sentence) == len(tmp_tags)
                sentencess.append(tmp_sentence)
                tagss.append(tmp_tags)
                tmp_sentence = []
                tmp_tags = []
            else:
                
                line = line.strip().split(' ')
                tmp_sentence.append(line[0])
                tmp_tags.append(line[1])
               # print(line)
        if len(tmp_sentence) != 0:
           # print("133333331")
            assert len(tmp_sentence) == len(tmp_tags)
            sentencess.append(tmp_sentence)
            tagss.append(tmp_tags)
    
    return sentencess, tagss



sentences, tags = read_data(data_base_dir + 'train.txt')
print(sentences[0], tags[0])

s_lengths = [len(s) for s in sentences]
print('最大句子长度：{}, 最小句子长度：{}, 平均句子长度：{:.2f}, 句子长度中位数：{:.2f}'.format(
    max(s_lengths), min(s_lengths), np.mean(s_lengths), np.median(s_lengths)))
df_len = pd.DataFrame({'s_len': s_lengths})
print(df_len.describe())




from collections import Counter
c = Counter(s_lengths) 
df_cumsum = pd.DataFrame(c.items(), columns=['s_len', 'cnt'])
df_cumsum = df_cumsum.sort_values(by='s_len', axis=0, ascending=True).reset_index(drop=True)
df_cumsum['cumsum'] = df_cumsum['cnt'].cumsum()
df_cumsum['cumsum_percentage'] = df_cumsum['cumsum']/len(sentences)
ax = df_cumsum.plot('s_len', 'cumsum_percentage', title='sentence length CDF')
quantile = 0
quantile_len = 60
for i,row in df_cumsum.iterrows():
    if row['s_len'] >= quantile_len:
        quantile = round(row['cumsum_percentage'], 3)
        break
print("\n分位点为%s的句子长度:%d" % (quantile, quantile_len))
ax.hlines(quantile, 0, quantile_len, colors="c", linestyles="dashed")
ax.vlines(quantile_len, 0, quantile, colors="c", linestyles="dashed")
ax.text(0, quantile, str(quantile))
ax.text(quantile_len, 0, str(quantile_len))
plt.show()


def build_vocab(sentences):
    global word_to_id
    for sentence in sentences:  
        for word in sentence:
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)
    return word_to_id

word_to_id = build_vocab(sentences)
print('vocab size:', len(word_to_id))

def convert_to_ids_and_padding(seqs, to_ids):
    ids = []
    for seq in seqs:
        if len(seq)>=maxlen: # 截断
            ids.append([to_ids[w] if w in to_ids else unk_id for w in seq[:maxlen]])
        else: # padding
            ids.append([to_ids[w] if w in to_ids else unk_id for w in seq] + [0]*(maxlen-len(seq)))

    return torch.tensor(ids, dtype=torch.long)


def load_data(filepath, word_to_id, shuffle=False):
    sentences, tags = read_data(filepath)

    inps = convert_to_ids_and_padding(sentences, word_to_id)
    trgs = convert_to_ids_and_padding(tags, tag_to_id)

    inp_dset = torch.utils.data.TensorDataset(inps, trgs)
    inp_dloader = torch.utils.data.DataLoader(inp_dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=4)
    return inp_dloader

# 查看data pipeline是否生效
inp_dloader = load_data(data_base_dir + 'train.txt', word_to_id)
sample_batch = next(iter(inp_dloader))
print('sample_batch:', len(sample_batch), sample_batch[0].size(), sample_batch[0].dtype,
      sample_batch[1].size(), sample_batch[1].dtype)  # [b,60] int64














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

# 查看模型
model = BiLSTM_CRF(len(word_to_id), hidden_size)
torchkeras.summary(model, input_shape=(60,), input_dtype=torch.int64)

device = torch.device("cpu")
print('*'*8, 'device:', device)



# 设置评价指标
metric_func = lambda y_pred, y_true: accuracy_score(y_true, y_pred)
metric_name = 'acc'
df_history = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_"+metric_name])





def train_step(model, inps, tags, optimizer):
    inps = inps.to(device)
    tags = tags.to(device)
    mask = torch.logical_not(torch.eq(inps, torch.tensor(0)))
    model.train()  # 设置train mode
    optimizer.zero_grad()  # 梯度清零


    # forward
    logits = model(inps)
    loss = model.crf_neg_log_likelihood(logits, tags, mask=mask, inp_logits=True)

    # backward
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新参数

    preds = model.crf_decode(logits, mask=mask, inp_logits=True) # List[List]
    pred_without_pad = []
    for pred in preds:
        pred_without_pad.extend(pred)
    tags_without_pad = torch.masked_select(tags, mask).cpu().numpy() # 返回是1维张量
    # print('tags_without_pad:', tags_without_pad.shape, type(tags_without_pad)) # [5082] tensor
    metric = metric_func(pred_without_pad, tags_without_pad)
    # print('*'*8, metric) # 标量

    return loss.item(), metric



@torch.no_grad()
def validate_step(model, inps, tags):
    inps = inps.to(device)
    tags = tags.to(device)
    mask = torch.logical_not(torch.eq(inps, torch.tensor(0)))  

    model.eval()  # 设置eval mode

    # forward
    logits = model(inps)
    loss = model.crf_neg_log_likelihood(logits, tags, mask=mask, inp_logits=True)

    preds = model.crf_decode(logits, mask=mask, inp_logits=True)  # List[List]
    pred_without_pad = []
    for pred in preds:
        pred_without_pad.extend(pred)
    tags_without_pad = torch.masked_select(tags, mask).cpu().numpy()  # 返回是1维张量
    metric = metric_func(pred_without_pad, tags_without_pad)
    # print('*' * 8, metric) # 标量

    return loss.item(), metric



# 打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m_%d %H:%M:%S')
    print('\n' + "=========="*8 + '%s'%nowtime)


def train_model(model, train_dloader, val_dloader, optimizer, num_epochs=10, print_every=150):
    starttime = time.time()
    print('*' * 27, 'start training...')
    printbar()

    best_metric = 0.
    for epoch in range(1, num_epochs+1):
        # 训练
        loss_sum, metric_sum = 0., 0.
        for step, (inps, tags) in enumerate(train_dloader, start=1):
            loss, metric = train_step(model, inps, tags, optimizer)
            loss_sum += loss
            metric_sum += metric

            # 打印batch级别日志
            if step % print_every == 0:
                print('*'*27, f'[step = {step}] loss: {loss_sum/step:.3f}, {metric_name}: {metric_sum/step:.3f}')

        # 验证 一个epoch的train结束，做一次验证
        val_loss_sum, val_metric_sum = 0., 0.
        for val_step, (inps, tags) in enumerate(val_dloader, start=1):
            val_loss, val_metric = validate_step(model, inps, tags)
            val_loss_sum += val_loss
            val_metric_sum += val_metric


        # 记录和收集 1个epoch的训练和验证信息
        # columns=['epoch', 'loss', metric_name, 'val_loss', 'val_'+metric_name]
        record = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        df_history.loc[epoch - 1] = record

        # 打印epoch级别日志
        print('EPOCH = {} loss: {:.3f}, {}: {:.3f}, val_loss: {:.3f}, val_{}: {:.3f}'.format(
               record[0], record[1], metric_name, record[2], record[3], metric_name, record[4]))
        printbar()

        # 保存最佳模型参数
        current_metric_avg = val_metric_sum/val_step
        if current_metric_avg > best_metric:
            best_metric = current_metric_avg
            checkpoint = save_dir + f'epoch{epoch:03d}_valacc{current_metric_avg:.3f}_ckpt.tar'
            if device.type == 'cuda' and ngpu > 1:
                model_sd = copy.deepcopy(model.module.state_dict())
            else:
                model_sd = copy.deepcopy(model.state_dict())
            # 保存
            torch.save({
                'loss': loss_sum / step,
                'epoch': epoch,
                'net': model_sd,
                'opt': optimizer.state_dict(),
            }, checkpoint)


    endtime = time.time()
    time_elapsed = endtime - starttime
    print('*' * 27, 'training finished...')
    print('*' * 27, 'and it costs {} h {} min {:.2f} s'.format(int(time_elapsed // 3600),
                                                               int((time_elapsed % 3600) // 60),
                                                               (time_elapsed % 3600) % 60))

    print('Best val Acc: {:4f}'.format(best_metric))
    return df_history



train_dloader = load_data(data_base_dir + 'train.txt', word_to_id)
val_dloader = load_data(data_base_dir + 'val.txt', word_to_id)

model = BiLSTM_CRF(len(word_to_id), hidden_size)
model = model.to(device)
if ngpu > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(ngpu)))  # 设置并行执行  device_ids=[0,1,2,3]

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
train_model(model, train_dloader, val_dloader, optimizer, num_epochs=EPOCHS, print_every=50)






# 绘制训练曲线
def plot_metric(df_history, metric):
    plt.figure()

    train_metrics = df_history[metric]
    val_metrics = df_history['val_' + metric]  #

    epochs = range(1, len(train_metrics) + 1)

    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')  #

    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])

    plt.savefig(imgs_dir + metric + '.png')  # 保存图片
    plt.show()

plot_metric(df_history, 'loss')
plot_metric(df_history, metric_name)



def predict(model, sentence, word_to_id):
    inp_ids = [word_to_id[w] if w in word_to_id else unk_id for w in sentence]
    inp_ids = torch.tensor(inp_ids, dtype=torch.long).unsqueeze(dim=0)
    logits = model(inp_ids)
    preds = model.crf_decode(logits, inp_logits=True)  # List[List]
    pred_ids = preds[0]
    pred_tags = [id_to_tag[tag_id] for tag_id in pred_ids]
    return pred_ids, pred_tags

result = []


def get_entity(pred_tags, pred_ids, sentence):
    ner_final={}
    ner = {'HCCX':[], 'MISC':[], 'HPPX':[],'HX':[]}
    i = 0
    while i<len(pred_tags):
        if pred_tags[i]=='O' or pred_ids[i]==0:
            i += 1
        elif pred_tags[i]=='B-HCCX':
            j = i
            while j+1<len(pred_tags) and pred_tags[j+1]=='I-HCCX':
                j += 1
            HCCX = [w for w in sentence[i:j+1]]
            ner['HCCX'].append(''.join(HCCX))
            print()
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
checkpoint = save_dir + 'epoch011_valacc0.951_ckpt.tar'
reloaded_model = BiLSTM_CRF(len(word_to_id), hidden_size)
ckpt = torch.load(checkpoint)  
model_sd = ckpt['net']
reloaded_model.load_state_dict(model_sd)
print('*' * 27, 'Model loaded success!')

reloaded_model.eval()  # 设置eval mode

sentences = [
        '贝亲婴儿多效洗衣液1. 2l+1l,宝宝衣物清洗剂,柠檬草香型/阳光香型']

for sentence in sentences:
    pred_ids, pred_tags = predict(reloaded_model, sentence, word_to_id)
    pred_ner = get_entity(pred_tags, pred_ids, sentence)  # 抽取实体
    print('*' * 10, 'sentence:', sentence)
    print('*' * 10, 'pred_ner:', pred_ner, '\n')
