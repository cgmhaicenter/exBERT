#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import glob
import pytorch_pretrained_bert as ppb
import re
import numpy as np
from tqdm import tqdm
import random
from exBERT import BertTokenizer
import multiprocessing as mp
import time
import argparse


# In[ ]:


class DataPreprocessor():
    def __init__(self, vocab_file, longest_sentence):
        self.tok = BertTokenizer(vocab_file)
        self.longest_sentence = longest_sentence
        self.end_symbol = '\.\?\;\)'
        self.start_symbol = 'A-Z\“\'\('
        self.p1 = re.compile('[A-Z1-9]{2,}['+self.end_symbol+'] +['+self.start_symbol+']|[^A-Z]{3}['+
                             self.end_symbol+'] +['+self.start_symbol+']')
        self.con_data_list = []
        
    def load_data(self, data_path, n_core=10):
        self.n_core = n_core
        with open(data_path,'r') as f:
            temp = f.read()
        self.DATA = []
        for ii in range(n_core):
            self.DATA.append(temp[ii*(int(len(temp)/n_core)+1):(ii+1)*(int(len(temp)/n_core)+1)])
            
    def sent_seprator(self, item):
        item = item.replace('- ','')
        sent_mark = self.p1.findall(item)
        temp = [[]]*len(sent_mark)
        count_table = np.zeros((len(sent_mark),2))
        for m_id,m in enumerate(self.p1.finditer(item)):
            count_table[m_id,0] = m.start()
            count_table[m_id,1] =  max([m.group().index(tsym) for tsym in self.end_symbol.split('\\')[1:] if tsym in m.group()])
        for ii in range(len(count_table)):
            if ii == 0:temp[ii] = item[:int(count_table[0].sum()+1)]
            else: temp[ii] = item[int(count_table[ii-1].sum())+1:int(count_table[ii].sum()+1)]
        for ii, item in enumerate(temp):
            if len(item) < 10:
                temp.remove(item)
        return temp
    
    def ord_data_generator(self, item):
        ord_data_list = []
        rev_data_list = []
        seg = ''
        seg_length = 0
        set_id_count = 0
        while set_id_count<=len(item)-1:
            if set_id_count == len(item)-1:
                if seg_length+len(self.tok.tokenize(item[set_id_count]))< self.longest_sentence-3:
                    seg += item[set_id_count]
                    if self.add_cls_sep(seg)[0]!=[]:
                        ord_data_list.append(self.add_cls_sep(seg)[0])
                        rev_data_list.append(self.add_cls_sep(seg)[1])
                    set_id_count+=1
                else:
                    if self.add_cls_sep(seg)[0]!=[]:
                        ord_data_list.append(self.add_cls_sep(seg)[0])
                        rev_data_list.append(self.add_cls_sep(seg)[1])
                    set_id_count+=1        
            else: 
                if seg_length+len(self.tok.tokenize(item[set_id_count]))< self.longest_sentence-3:
                    seg += item[set_id_count]
                    seg_length += len(self.tok.tokenize(item[set_id_count]))
                    set_id_count+=1
                elif len(self.tok.tokenize(item[set_id_count]))+len(self.tok.tokenize(item[set_id_count+1]))<self.longest_sentence-3:
                    if self.add_cls_sep(seg)[0]!=[]:
                        ord_data_list.append(self.add_cls_sep(seg)[0])
                        rev_data_list.append(self.add_cls_sep(seg)[1])
                    seg = ''
                    seg_length = 0
                else:
                    if self.add_cls_sep(seg)[0]!=[]:
                        ord_data_list.append(self.add_cls_sep(seg)[0])
                        rev_data_list.append(self.add_cls_sep(seg)[1])
                    seg = ''
                    seg_length = 0
                    set_id_count+=1
        return ord_data_list, rev_data_list
    
    def add_cls_sep(self, item):
        if len(self.p1.findall(item))>1:
            count_table = np.zeros((len(self.p1.findall(item)),2))
            for m_id,m in enumerate(self.p1.finditer(item)):
                count_table[m_id,0] = m.start()
                count_table[m_id,1] =  max([m.group().index(tsym) for tsym in self.end_symbol.split('\\')[1:] if tsym in m.group()])
            tar_id = int(len(count_table)/2)-1
            return '[CLS] '+item[:int(count_table[tar_id].sum()+1)]+' [SEP] '+item[int(count_table[tar_id].sum()+1):]+' [SEP]', '[CLS] '+item[int(count_table[tar_id].sum()+1):]+' [SEP] '+item[:int(count_table[tar_id].sum()+1)]+' [SEP]'
        else:
    #         print(item)
            return [[],[]]
    
    def f(self, data):
        temp_data = self.sent_seprator(data)
        return self.ord_data_generator(temp_data)       
    
#     def f(self,data):
#         temp_data = [[]]*len(data)
#         for ii, item in enumerate(data):
#             temp_data[ii] = self.sent_seprator(item)
#         return self.ord_data_generator(temp_data)
    
    def process(self):
        t1 = time.time()
        with mp.Pool(processes=self.n_core) as pool:
            output = []
            for item in tqdm(pool.map(self.f, self.DATA), total=len(self.DATA)):
                output.append(item)
         
        ord_data_list = []
        rev_data_list = []
        for ii in range(self.n_core):
            ord_data_list.extend(output[ii][0])
            rev_data_list.extend(output[ii][1])
        
        return ord_data_list, rev_data_list
    
    def f1(self, data_list):
        output_list = []
        while len(output_list)<len(data_list):
            rp1 = np.random.permutation(len(data_list))[0]
            rp2 = np.random.rand(1)
            rp3 = np.random.permutation(len(data_list))[0]
            rp4 = np.random.rand(1)
            sent1 = data_list[rp1].split('[CLS]')[-1].split('[SEP]')[int(rp2>0.5)]
            if rp1!=rp3-1:
                sent2 = data_list[rp3].split('[CLS]')[-1].split('[SEP]')[int(rp4>0.5)]
                out_sent = '[CLS] '+sent1+' [SEP] '+sent2+' [SEP]'
                if len(self.tok.tokenize(out_sent)) < self.longest_sentence:
                    output_list.append(out_sent)
        return output_list
    
    def generate_random_data_list(self, data_list, n_core = 5):
        sep_data_list = [[]]*n_core
        for ii in range(n_core):
            sep_data_list[ii] = data_list[ii*(int(len(data_list)/n_core)+1):(ii+1)*(int(len(data_list)/n_core)+1)]
        with mp.Pool(processes=n_core) as  pool:
            output = pool.map(self.f1, sep_data_list)
        OUTPUT = []
        for ii in range(n_core):
            OUTPUT.extend(output[ii])
        return OUTPUT
            
        



if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-voc','--voc',required=True, type=str, help='path to the vocabulary file')
    ap.add_argument('-ls','--ls',required=True, type=int, help='set the number of the max length')
    ap.add_argument('-dp','--data_path',required=True, type=str, help='path to the data.txt')
    ap.add_argument('-n_c','--n_core',default=5, type=int, help='set the number of cpus you want to use during pre-processing')
    ap.add_argument('-rd','--random',default=1, type=int, help='input 1: use random order sentence as classification task, input 0 : use reverse sentence')
    ap.add_argument('-sp','--save_path', required=True, type=str, help='path to storage')
    args = vars(ap.parse_args())
    dp = DataPreprocessor(args['voc'], args['ls'])
    dp.load_data(args['data_path'], n_core=args['n_core'])
    print('processing data, thie might take some time')
    ord_data, rev_data = dp.process()
    if args['random']==1:
        print('generating random sequence')
        rand_data = dp.generate_random_data_list(ord_data)
        with open(args['save_path'],'wb') as f:
            pickle.dump([ord_data, rand_data],f)
    else:
        with open(args['save_path'],'wb') as f:
            pickle.dump([ord_data, rev_data],f)
    
    
    
    
