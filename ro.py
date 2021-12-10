import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#import seaborn as sns
#import transformers
#import json
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
#from transformers import RobertaModel, RobertaTokenizer
import logging
import argparse
import utility_hs as util
logging.basicConfig(level=logging.ERROR)
from transformers import RobertaTokenizer, RobertaForSequenceClassification


parser = argparse.ArgumentParser(description='Hate Speech Model')
parser.add_argument('--traindata', type=str, default='../e_hsp/english_dataset/english_dataset.tsv', help='location of the training data')
parser.add_argument('--traindata2', type=str, default='../e_hsp/English_2020/hasoc_2020_en_train_new.xlsx', help='location of the training data 2')
parser.add_argument('--traindata3', type=str, default='../e_hsp/en_Hasoc2021_train.csv', help='location of the training data 3')
parser.add_argument('--testdata', type=str, default='../e_hsp/english_dataset/hasoc2019_en_test-2919.tsv', help='location of the test data')
parser.add_argument('--testdata2', type=str, default='../e_hsp/English_2020/hasoc_2020_en_test_new.xlsx', help='location of the test data 2')
parser.add_argument('--testdata3', type=str, default='../e_hsp/en_Hasoc2021_test_task1.csv', help='location of the test data 3')
parser.add_argument('--savet', type=str, default='modelt5small_hasoc_task1a.pt', help='filename of the model checkpoint')
parser.add_argument('--pikle', type=str, default='modelt5small_hasoc_task1a.pkl', help='pickle filename of the model checkpoint')
parser.add_argument('--ofile1', type=str, default='outputfile_task1a.txt', help='output file')
parser.add_argument('--submission1', type=str, default='submitfile_task1a.csv', help='submission file')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size')
args = parser.parse_args()


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, t_data = util.read_data()
    print(data.head())

    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    # outputs = model(**inputs, labels=labels)
    # loss, logits = outputs[:2]
    # print("LOSS: ",loss)
