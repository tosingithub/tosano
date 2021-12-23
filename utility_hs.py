import pandas as pd
import argparse
import re


parser = argparse.ArgumentParser(description='Hate Speech Model')
parser.add_argument('--has19_traindata', type=str, default='/home/shared_data/h/has19_traindata.csv', help='location of the training data')
parser.add_argument('--has19_devdata', type=str, default='/home/shared_data/h/has19_devdata.csv', help='location of the dev data')
parser.add_argument('--has19_testdata', type=str, default='/home/shared_data/h/has19_testdata.csv', help='location of the test data')
parser.add_argument('--has20_traindata', type=str, default='/home/shared_data/h/has20_traindata.csv', help='location of the training data')
parser.add_argument('--has20_devdata', type=str, default='/home/shared_data/h/has20_devdata.csv', help='location of the dev data')
parser.add_argument('--has20_testdata', type=str, default='/home/shared_data/h/has20_testdata.csv', help='location of the test data')
parser.add_argument('--has21_traindata', type=str, default='/home/shared_data/h/has21_traindata.csv', help='location of the training data')
parser.add_argument('--has21_devdata', type=str, default='/home/shared_data/h/has21_devdata.csv', help='location of the dev data')
parser.add_argument('--has21_testdata', type=str, default='/home/shared_data/h/has21_testdata.csv', help='location of the test data')
# Additional datasets

# parser.add_argument('--traindata', type=str, default='../e_hsp/english_dataset/english_dataset.tsv', help='location of the training data')
# parser.add_argument('--traindata2', type=str, default='../e_hsp/English_2020/hasoc_2020_en_train_new.xlsx', help='location of the training data 2')
# parser.add_argument('--traindata3', type=str, default='../e_hsp/en_Hasoc2021_train.csv', help='location of the training data 3')
# parser.add_argument('--testdata', type=str, default='../e_hsp/english_dataset/hasoc2019_en_test-2919.tsv', help='location of the test data')
# parser.add_argument('--testdata2', type=str, default='../e_hsp/English_2020/hasoc_2020_en_test_new.xlsx', help='location of the test data 2')
# parser.add_argument('--testdata3', type=str, default='../e_hsp/en_Hasoc2021_test_task1.csv', help='location of the test data 3')
# parser.add_argument('--savet', type=str, default='modelt5small_hasoc_task1a.pt', help='filename of the model checkpoint')
# parser.add_argument('--pikle', type=str, default='modelt5small_hasoc_task1a.pkl', help='pickle filename of the model checkpoint')
# parser.add_argument('--ofile1', type=str, default='outputfile_task1a.txt', help='output file')
# parser.add_argument('--submission1', type=str, default='submitfile_task1a.csv', help='submission file')
# parser.add_argument('--seed', type=int, default=1111, help='random seed')
# parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
# parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
# parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size')
args = parser.parse_args()


def get_data(datatype, datayear='2020', combined_traindata=False):
    """ Select the dataset to use """
    if datatype == 'hasoc':
        if not combined_traindata:
            if datayear == '2020':
                data1 = pd.read_csv(args.has20_traindata)
                data2 = pd.read_csv(args.has20_devdata)
                data3 = pd.read_csv(args.has20_testdata)
                traindata, devdata, testdata = data1, data2, data3
            else:
                data1 = pd.read_csv(args.has21_traindata)
                data2 = pd.read_csv(args.has21_devdata)
                data3 = pd.read_csv(args.has21_testdata)
                traindata, devdata, testdata = data1, data2, data3
        else:
            data1a = pd.read_csv(args.has19_traindata)
            data2b = pd.read_csv(args.has19_devdata)
            data1aa = pd.read_csv(args.has20_traindata)
            data2bb = pd.read_csv(args.has20_devdata)
            data1aaa = pd.read_csv(args.has21_traindata)
            data2bbb = pd.read_csv(args.has21_devdata)
            traindata, devdata = pd.concat([data1a, data1aa, data1aaa]), pd.concat([data2b, data2bb, data2bbb])
            if datayear == '2020':
                testdata = pd.read_csv(args.has20_testdata)
            else:
                testdata = pd.read_csv(args.has21_testdata)
            
    elif datatype =='hateval':
        data1 = pd.read_csv(args.has19_traindata)
    else:
        data1 = pd.read_csv(args.has19_traindata)
    return traindata.drop_duplicates(keep='first'), devdata.drop_duplicates(keep='first'), testdata.drop_duplicates(keep='first')


def preprocess_pandas(data, columns):
    ''' <data> is a dataframe which contain  a <text> column  '''
    df_ = pd.DataFrame(columns=columns)
    df_ = data
    df_['text'] = data['text'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # remove emails
    df_['text'] = data['text'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # remove IP address
    df_['text'] = data['text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)          # remove URLs
    df_['text'] = data['text'].str.replace('[#,@,&,<,>,\,/,-]','')                                             # remove special characters
    df_['text'] = data['text'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)                           # remove emojis+
    df_['text'] = data['text'].str.replace('[','')
    df_['text'] = data['text'].str.replace(']','')
    df_['text'] = data['text'].str.replace('\n', ' ')
    df_['text'] = data['text'].str.replace('\t', ' ')
    df_['text'] = data['text'].str.replace(' {2,}', ' ', regex=True)                                           # remove 2 or more spaces
    df_['text'] = data['text'].str.lower()
    df_['text'] = data['text'].str.strip()
    df_['text'] = data['text'].replace('\d', '', regex=True)                                                   # remove numbers
    df_.drop_duplicates(subset=['text'], keep='first')
    df_.dropna()
    return df_


#def read_data(task_pref = "binary classification: "):
#     data1 = pd.read_csv(args.traindata, sep='\t', header=0, encoding="latin1").fillna(method="ffill")
#     test_data1 = pd.read_csv(args.testdata, sep='\t', header=0, encoding="latin1").fillna(method="ffill")
#     #print(len(test_data1))
#     data2 = pd.read_excel(args.traindata2, dtype={'tweet_id':'string'}, engine='openpyxl')
#     test_data2 = pd.read_excel(args.testdata2, dtype={'tweet_id':'string'}, engine='openpyxl')
#     data3 = pd.read_csv(args.traindata3, header=0, encoding="latin1").fillna(method="ffill")
#     test_data3 = pd.read_csv(args.testdata3)
    
#     # combine data
#     data1 = data1[['text_id', 'text', 'task_1']].copy() # .drop(axis=1, columns=['text_id','task_2','task_3']) # drop unneeded columns
#     test_data1 = test_data1[['text_id', 'text', 'task_1']].copy()
#     data1 = data1.rename(columns={'text_id': '_id'}, inplace=False)
#     test_data1 = test_data1.rename(columns={'text_id': '_id'}, inplace=False)
#     data2 = data2[['tweet_id', 'text', 'task_1']].copy()
#     test_data2 = test_data2[['tweet_id', 'text', 'task_1']].copy()
#     data2 = data2.rename(columns={'tweet_id': '_id'}, inplace=False)
#     test_data2 = test_data2.rename(columns={'tweet_id': '_id'}, inplace=False)
#     data3 = data3[['_id', 'text', 'task_1']].copy()
#     test_data3 = test_data3[['_id', 'text']].copy()
#     data = pd.concat([data1, data2, data3])
#     data = data.drop_duplicates(keep='first')      # drop duplicates, if any
    
#     task_prefix = task_pref
#     data['text'] = task_prefix + data['text']
#     test_data = test_data2
#     test_data['text'] = task_prefix + test_data['text']

#     return data, test_data
