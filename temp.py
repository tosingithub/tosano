import torch
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='Hate Speech Model')
parser.add_argument('--traindata', type=str, default='english_dataset/english_dataset.tsv', help='location of the training data')
parser.add_argument('--traindata2', type=str, default='English_2020/hasoc_2020_en_train_new.xlsx', help='location of the training data 2')
parser.add_argument('--traindata3', type=str, default='en_Hasoc2021_train.csv', help='location of the training data 3')
parser.add_argument('--testdata', type=str, default='english_dataset/hasoc2019_en_test-2919.tsv', help='location of the test data')
parser.add_argument('--testdata2', type=str, default='English_2020/hasoc_2020_en_test_new.xlsx', help='location of the test data 2')
parser.add_argument('--testdata3', type=str, default='en_Hasoc2021_test_task1.csv', help='location of the test data 3')
args = parser.parse_args()


if __name__ == '__main__':
    data1 = pd.read_csv(args.traindata, sep='\t', header=0, encoding="latin1").fillna(method="ffill")
    test_data1 = pd.read_csv(args.testdata, sep='\t', header=0, encoding="latin1").fillna(method="ffill")
    #print(len(test_data1))
    data2 = pd.read_excel(args.traindata2, dtype={'tweet_id':'string'}, engine='openpyxl')
    test_data2 = pd.read_excel(args.testdata2, dtype={'tweet_id':'string'}, engine='openpyxl')
    data3 = pd.read_csv(args.traindata3, header=0, encoding="latin1").fillna(method="ffill")
    test_data3 = pd.read_csv(args.testdata3)

    data1 = data1[['text_id', 'text', 'task_1', 'task_2']].copy() # .drop(axis=1, columns=['text_id','task_2','task_3']) # drop unneeded columns
    test_data1 = test_data1[['text_id', 'text', 'task_1', 'task_2']].copy()
    data1 = data1.rename(columns={'text_id': '_id'}, inplace=False)
    test_data1 = test_data1.rename(columns={'text_id': '_id'}, inplace=False)
    data2 = data2[['tweet_id', 'text', 'task_1', 'task_2']].copy()
    test_data2 = test_data2[['tweet_id', 'text', 'task_1', 'task_2']].copy()
    data2 = data2.rename(columns={'tweet_id': '_id'}, inplace=False)
    test_data2 = test_data2.rename(columns={'tweet_id': '_id'}, inplace=False)
    data3 = data3[['_id', 'text', 'task_1', 'task_2']].copy()
    test_data3 = test_data3[['_id', 'text']].copy()

    traindata1, valdata1 = train_test_split(data1, test_size=0.1, shuffle=True)
    traindata2, valdata2 = train_test_split(data2, test_size=0.1, shuffle=True)
    traindata3, valdata3 = train_test_split(data3, test_size=0.1, shuffle=True)

    traindata1.to_csv('has19_traindata.csv', index=False)
    valdata1.to_csv('has19_devdata.csv', index=False)
    test_data1.to_csv('has19_testdata.csv', index=False)
    
    traindata2.to_csv('has20_traindata.csv', index=False)
    valdata2.to_csv('has20_devdata.csv', index=False)
    test_data2.to_csv('has20_testdata.csv', index=False)

    traindata3.to_csv('has21_traindata.csv', index=False)
    valdata3.to_csv('has21_devdata.csv', index=False)
    test_data3.to_csv('has21_testdata.csv', index=False)
