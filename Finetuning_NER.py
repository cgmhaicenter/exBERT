#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import glob
from sklearn.metrics import f1_score
import torch as t
import tensorflow as tf 
import re
import torch
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse
from tqdm import tqdm
import torch.nn as nn
import os
import pandas as pd
from exBERT import BertTokenizer, BertAdam

# In[2]:

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", required = True, type = int, help='number of training epochs')
ap.add_argument("-b", "--batchsize", required = True, type = int, help='training batchsize')
ap.add_argument("-sp","--save_path",required = True, type = str, help='path to storaget the loss table, stat_dict')
ap.add_argument('-dv','--device',required = True, type = int,nargs='+', help='gpu id for the training, -1 for cpu, ex -dv 0 1 2 3')
ap.add_argument('-lr','--learning_rate',required = True, type = float, help='learning rate')
ap.add_argument('-str','--strategy',required = True, type = str, help='choose a strategy from [exBERT], [sciBERT], [bioBERT]')
ap.add_argument('-config','--config',required = True, type = str, nargs = '+', help='dir to the config file')
ap.add_argument('-vocab','--vocab',required = True, type = str, help='path to the vocab file for tokenization')
ap.add_argument('-pm_p','--pretrained_model_path',default = None, type = str, help='path to the pretrained_model stat_dict')
ap.add_argument('-dp','--data_path',required = True, type = str, help='path to data ')
ap.add_argument('-tln','--train_layer_number',required = True, type = str, nargs='+', help= 'which layer is going to be trained, ex -ltn 9 10 11')
args = vars(ap.parse_args())
print(args)


if args['device'] == -1:
    device = 'cpu'
    device_ids = 'cpu'
else:
    device_ids = args['device']
    device = 'cuda:'+str(args['device'][0])
print(device_ids)



tok = BertTokenizer(args['vocab'])
vocab_size = len(tok.vocab)

def data_loading(PATH, s_tar, return_tag_list = False):
    data = []
    label = []
    file_list = glob.glob(PATH+s_tar)
    for pii, path in enumerate(file_list):
        print('load file: ' + str(path))
        with open(path,'r',encoding='utf-8') as f:
            temp_data = f.read()
            for ii,item in enumerate(temp_data.split('\n')):
                if '\t' in item:
                    data.append(item.split('\t')[0])
                    label.append(item.split('\t')[1])
    uniq_list = sorted(list(set(label)))
    
    label = pd.DataFrame(label)
    label_id = np.ones(len(label))*-1
    for ii, item in enumerate(uniq_list):
        label_id[np.where((label[0]==item)*1==1)] = ii
    label_id = label_id.tolist()
    print('seprate sentence')
    data_set = []
    data_set = (' '.join(data)).split(' . ')
    label_set = [[]]*len(data_set)
    count = 0
    for ii, item in enumerate(data_set):
        data_set[ii] = item.split(' ')
        label_set[ii] = label_id[count:count+len(data_set[ii])]
        count = count+len(data_set[ii])+1
        if len(label_set[ii]) != len(data_set[ii]):
            print('some thing wrong')
            break


    data_set1 = [[]]*len(re.split('\. [A-Z]',' '.join(data)))
    label_set1 = [[]]*len(re.split('\. [A-Z]',' '.join(data)))
    count= 0
    for ii, item in enumerate(data_set):
        if not item[0][0].isupper():
            if type(item) is list :
                data_set1[count-1] = data_set1[count-1]+item
                label_set1[count-1] = label_set1[count-1]+label_set[ii]
        else:
            data_set1[count] = item
            label_set1[count] = label_set[ii]
            count = count+1
            if len(label_set1[count-1]) !=len(data_set1[count-1]):
                print('something wrong')

    print('tokenization')
    label_tok = [[]]*len(data_set1)
    data_tok = [[]]*len(data_set1)
    del_list = []
    for ss, sentence  in enumerate(data_set1):
        try:
            temp_tok = tok.tokenize(' '.join(sentence))
            temp = []
            count = 0
            for ii, item in enumerate(temp_tok):
                if item[0:2]=='##':
                    temp.append(label_set1[ss][count-1])
                else:
                    temp.append(label_set1[ss][count])
                    count = count+1
            label_tok[ss] = temp
    #         train_data_tok[ss] = sentence
            data_tok[ss] = temp_tok
            if len(data_tok) != len(label_tok):
                print('something wrong')
        except:
            del_list.append(ss)
    print('error num:' + str(len(del_list)) )
    del_count = 0
    for ii in range(len(del_list)):
        del train_label_tok[del_list[ii]-del_count]
        del train_data_tok[del_list[ii]-del_count]
        del_count +=1
    print('complete')
    if return_tag_list:
        return label_tok, data_tok, uniq_list
    else:
        return label_tok, data_tok


train_label_tok, train_data_tok, uniq_list = data_loading(args['data_path'],'train*', return_tag_list=True)
val_label_tok, val_data_tok = data_loading(args['data_path'],'d*')
test_label_tok, test_data_tok = data_loading(args['data_path'],'test*')


if args['strategy'] == 'exBERT':
    from exBERT import BertForTokenClassification, BertConfig
    bert_config_1 = BertConfig.from_json_file(args['config'][0])
    bert_config_2 = BertConfig.from_json_file(args['config'][1])
    print("Building PyTorch model from configuration: {}".format(str(bert_config_1)))
    print("Building PyTorch model from configuration: {}".format(str(bert_config_2)))
    model = BertForTokenClassification(bert_config_1, len(uniq_list), bert_config_2)
else:
    from exBERT import BertForTokenClassification, BertConfig
    bert_config_1 = BertConfig.from_json_file(args['config'][0])
    print("Building PyTorch model from configuration: {}".format(str(bert_config_1)))
    model = BertForTokenClassification(bert_config_1,  num_labels = len(uniq_list))

if args['pretrained_model_path'] is not None:
    stat_dict = t.load(args['pretrained_model_path'], map_location='cpu')
    model.load_state_dict(stat_dict, strict=False)

sta_name_pos = 0
if device is not 'cpu':
    if len(device_ids)>1:
        model = nn.DataParallel(model,device_ids=device_ids)
        sta_name_pos = 1
    model.to(device)
    

for ii, item in enumerate(model.named_parameters()):
    item[1].requires_grad = False
    if item[0].split('.')[sta_name_pos] !='bert':
        print(item[0])
        item[1].requires_grad=True
    if 'pooler' in item[0]:
        print(item[0])
        item[1].requires_grad= True
    if len(item[0].split('.'))>4:
        if item[0].split('.')[3+sta_name_pos] in args['train_layer_number']:
            item[1].requires_grad=True
            print(item[0])



lr = args['learning_rate']
param_optimizer = list(model.named_parameters())
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=lr)


num_epoc = args['epochs']
batch_size = args['batchsize']
longest_sentence = 256
train_los_table = np.zeros((num_epoc,int(len(train_data_tok)/batch_size),3))
val_los_table = np.zeros((num_epoc,int(len(train_data_tok)/batch_size),3))
best_loss = float('inf')


def prepare_batch(batch_data_tok, batch_label_tok, data_ind):
    Input_ids = t.zeros(batch_size,longest_sentence).long()
    Attention_mask = t.zeros(batch_size,longest_sentence).long()
    Label = t.ones(batch_size,longest_sentence).long()*-1
    for ii, item in enumerate(data_ind):
        sentence_length = len(batch_data_tok[item])
        if sentence_length>254:
            sentence_length = 254
        temp = t.tensor(tok.convert_tokens_to_ids(batch_data_tok[item]))
        Input_ids[ii,1:1+sentence_length] = temp[:sentence_length]
        Input_ids[ii][0] = 101
        Input_ids[ii][1+sentence_length] = 102
        Attention_mask[ii,1:1+sentence_length] = 1
        Label[ii,1:1+sentence_length] = t.tensor(batch_label_tok[item][0:sentence_length])
    Input_ids = Input_ids.to(device)
    Attention_mask=Attention_mask.to(device)
    Label=Label.to(device)
    return Input_ids, Attention_mask, Label

def process_batch(model, batch_input, is_train = True, out_ans = False):
    if is_train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    loss1 = model(batch_input[0],
          attention_mask = batch_input[1],
          labels = batch_input[2])    
    if is_train:
        loss1[0].sum().unsqueeze(0).backward()
        optimizer.step()
    masked_logit = loss1[1].argmax(-1).view(-1)[batch_input[2].view(-1)!=-1]
    masked_label = batch_input[2].view(-1)[batch_input[2].view(-1)!=-1]
    acc = (masked_logit-masked_label==0).float().sum()/len(masked_label)
    f1 = f1_score(masked_label.detach().cpu().numpy(), masked_logit.detach().cpu().numpy(), average='macro')
    
    if out_ans:
        return masked_logit, masked_label, loss1[0].sum().data
    else:
        return loss1[0].sum().data, acc, f1
        
        

try:
    for epoch in range(num_epoc):
        t2 = time.time()
        epo_rp = np.random.permutation(len(train_data_tok))
        train_loss = 0
        val_loss = 0
        train_acc = 0
        val_acc = 0
        train_f1 = 0
        val_f1 = 0
        for batch_ind in range(int(len(train_data_tok)/batch_size)):
            data_ind = epo_rp[batch_ind*batch_size:batch_ind*batch_size+batch_size]
            BATCH_INPUT = prepare_batch(train_data_tok, train_label_tok, data_ind)
            batch_loss, batch_acc, batch_f1 = process_batch(model, BATCH_INPUT)
            train_los_table[epoch,batch_ind,0] = batch_loss
            train_loss += batch_loss
            train_los_table[epoch,batch_ind,1] = batch_acc
            train_acc += batch_acc
            train_los_table[epoch,batch_ind,2] = batch_f1
            train_f1 += batch_f1 
            
            data_ind = np.random.permutation(len(val_data_tok))[:batch_size]
            BATCH_INPUT = prepare_batch(val_data_tok, val_label_tok, data_ind)
            with t.no_grad():
                batch_loss, batch_acc, batch_f1 = process_batch(model, BATCH_INPUT, is_train=False)
            val_los_table[epoch,batch_ind,0] = batch_loss
            val_loss += batch_loss
            val_los_table[epoch,batch_ind,1] = batch_acc
            val_acc += batch_acc
            val_los_table[epoch,batch_ind,2] = batch_f1
            val_f1 += batch_f1             

            if batch_ind>0 and batch_ind % 10 ==0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_Loss: {:.5f} \tval_Loss: {:.5f}                 \t train_acc:{:.3f} \tval_acc:{:.3f} \ttf1: {:.3f}\tvf1: {:.3f} time: {:.2f} \tlr:{:.6f}' 
                          .format(
                        epoch,
                        batch_ind*batch_size ,
                        len(train_data_tok), 100 * (batch_ind*batch_size) / len(train_data_tok),
                        train_loss/10/batch_size,val_loss/10/batch_size,train_acc/10,val_acc/10,
                        train_f1/10,val_f1/10,
                        time.time()-t2 ,optimizer.param_groups[0]['lr']))
                train_loss = 0
                val_loss = 0
                train_acc = 0
                val_acc = 0          
                train_f1 = 0
                val_f1 = 0
                with open(args['save_path']+'/loss.pkl','wb') as f:
                    pickle.dump([train_los_table,val_los_table,args],f)

        ###### The end of epoch
        if len(device_ids)>1:
            t.save(model.module.state_dict(),args['save_path'] +'/stat_dic_'+args['strategy']+'_'+str(epoch))
        else:
            t.save(model.state_dict(),args['save_path'] +'/stat_dic_'+args['strategy']+'_'+str(epoch))
        
        ## start evaluation
        MODEL_OUT = []
        LABEL = []
        with t.no_grad():
            val_rp = np.random.permutation(len(val_data_tok))
            val_loss = 0
            for batch_ind in range(int(np.ceil(len(val_data_tok)/batch_size))):
                data_ind = val_rp[batch_ind*batch_size:batch_ind*batch_size+batch_size]
                BATCH_INPUT = prepare_batch(val_data_tok, val_label_tok, data_ind)
                output = process_batch(model, BATCH_INPUT, is_train=False, out_ans=True)
                MODEL_OUT.extend(output[0].detach().cpu().numpy())
                LABEL.extend(output[1].detach().cpu().numpy())
                val_loss+=output[2]
            print('VAL  F1:' + str(f1_score(LABEL, MODEL_OUT, average = 'macro')))

        if val_loss < best_loss:
            if len(device_ids)>1:
                t.save(model.module.state_dict(),args['save_path']+'/Best_stat_dic_'+args['strategy'])
            else:
                t.save(model.state_dict(),args['save_path']+'/Best_stat_dic_'+args['strategy'])
            
            best_loss = val_loss
            print('update!!!!!!!!!!!!')
    
except KeyboardInterrupt:
    print('saving stat_dict and loss table')
    with open(args['save_path']+'/kbstop_loss.pkl','wb') as f:
        pickle.dump([train_los_table,val_los_table,args],f)
    t.save(model.state_dict(), args['save_path']+'/kbstop_stat_dict')


    
### eval by test
best_state_dict = t.load(args['save_path']+'/Best_stat_dic_'+args['strategy'],map_location = 'cpu')
model.load_state_dict(best_state_dict)
MODEL_OUT = []
LABEL = []
with t.no_grad():
    test_rp = np.random.permutation(len(test_data_tok))
    for batch_ind in range(int(np.ceil(len(test_data_tok)/batch_size))):
        data_ind = test_rp[batch_ind*batch_size:batch_ind*batch_size+batch_size]
        BATCH_INPUT = prepare_batch(test_data_tok, test_label_tok, data_ind)
        output = process_batch(model, BATCH_INPUT, is_train=False, out_ans=True)
        MODEL_OUT.extend(output[0].detach().cpu().numpy())
        LABEL.extend(output[1].detach().cpu().numpy())
    print('TEST  F1:' + str(f1_score(LABEL, MODEL_OUT, average = 'macro')))


