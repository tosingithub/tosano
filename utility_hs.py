import pandas as pd
import argparse


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


def read_data(task_pref = "binary classification: "):
    data1 = pd.read_csv(args.traindata, sep='\t', header=0, encoding="latin1").fillna(method="ffill")
    test_data1 = pd.read_csv(args.testdata, sep='\t', header=0, encoding="latin1").fillna(method="ffill")
    #print(len(test_data1))
    data2 = pd.read_excel(args.traindata2, dtype={'tweet_id':'string'}, engine='openpyxl')
    test_data2 = pd.read_excel(args.testdata2, dtype={'tweet_id':'string'}, engine='openpyxl')
    data3 = pd.read_csv(args.traindata3, header=0, encoding="latin1").fillna(method="ffill")
    test_data3 = pd.read_csv(args.testdata3)
    
    # combine data
    data1 = data1[['text_id', 'text', 'task_1']].copy() # .drop(axis=1, columns=['text_id','task_2','task_3']) # drop unneeded columns
    test_data1 = test_data1[['text_id', 'text', 'task_1']].copy()
    data1 = data1.rename(columns={'text_id': '_id'}, inplace=False)
    test_data1 = test_data1.rename(columns={'text_id': '_id'}, inplace=False)
    data2 = data2[['tweet_id', 'text', 'task_1']].copy()
    test_data2 = test_data2[['tweet_id', 'text', 'task_1']].copy()
    data2 = data2.rename(columns={'tweet_id': '_id'}, inplace=False)
    test_data2 = test_data2.rename(columns={'tweet_id': '_id'}, inplace=False)
    data3 = data3[['_id', 'text', 'task_1']].copy()
    test_data3 = test_data3[['_id', 'text']].copy()
    data = pd.concat([data1, data2, data3])
    data = data.drop_duplicates(keep='first')      # drop duplicates, if any
    
    task_prefix = task_pref
    data['text'] = task_prefix + data['text']
    test_data = test_data2
    test_data['text'] = task_prefix + test_data['text']

    return data, test_data