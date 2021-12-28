import pandas as pd
import argparse
import re


parser = argparse.ArgumentParser(description='Hate Speech')
# --datatype: hasoc, trol, sema, semb, hos, olid (--olidtask = a, b or c)
# USAGE EXAMPLE in the terminal: python ht5.py --datatype trol
parser.add_argument('--datatype', type=str, default='hasoc', help='data of choice')
parser.add_argument('--has19_traindata', type=str, default='/home/shared_data/h/has19_traindata.csv', help='location of the training data')
parser.add_argument('--has19_devdata', type=str, default='/home/shared_data/h/has19_devdata.csv', help='location of the dev data')
parser.add_argument('--has19_testdata', type=str, default='/home/shared_data/h/has19_testdata.csv', help='location of the test data')
parser.add_argument('--has20_traindata', type=str, default='/home/shared_data/h/has20_traindata.csv', help='location of the training data')
parser.add_argument('--has20_devdata', type=str, default='/home/shared_data/h/has20_devdata.csv', help='location of the dev data')
parser.add_argument('--has20_testdata', type=str, default='/home/shared_data/h/has20_testdata.csv', help='location of the test data')
parser.add_argument('--has21_traindata', type=str, default='/home/shared_data/h/has21_traindata.csv', help='location of the training data')
parser.add_argument('--has21_devdata', type=str, default='/home/shared_data/h/has21_devdata.csv', help='location of the dev data')
parser.add_argument('--has21_testdata', type=str, default='/home/shared_data/h/has21_testdata.csv', help='location of the test data')
# Trolling & Aggression
parser.add_argument('--trol_traindata', type=str, default='/home/shared_data/h/eng_trolling_agression/trac2_eng_train.csv', help='location of the training data')
parser.add_argument('--trol_devdata', type=str, default='/home/shared_data/h/eng_trolling_agression/trac2_eng_dev.csv', help='location of the dev data')
# HatEval SemEval A
parser.add_argument('--sema_traindata', type=str, default='/home/shared_data/h/HatEval_SemEval_A/train_en.tsv', help='location of the training data')
parser.add_argument('--sema_devdata', type=str, default='/home/shared_data/h/HatEval_SemEval_A/dev_en.tsv', help='location of the dev data')
# HatEval SemEval B
parser.add_argument('--semb_traindata', type=str, default='/home/shared_data/h/HatEval_SemEval_B/train_en.tsv', help='location of the training data')
parser.add_argument('--semb_devdata', type=str, default='/home/shared_data/h/HatEval_SemEval_B/train_en.tsv', help='location of the dev data')
# HOS
parser.add_argument('--hos_traindata', type=str, default='/home/shared_data/h/HOS_train_data.csv', help='location of the training data')
parser.add_argument('--hos_devdata', type=str, default='/home/shared_data/h/HOS_dev_data.csv', help='location of the dev data')
# OLID
parser.add_argument('--olid_traindata', type=str, default='/home/shared_data/h/OLIDv1.0/olid-training.csv', help='location of the training data')
parser.add_argument('--olid_devdata', type=str, default='/home/shared_data/h/OLIDv1.0/olid-dev.csv', help='location of the dev data')
parser.add_argument('--olidtask', type=str, default="a", help='Task Alphabet')      # a, b or c

args = parser.parse_args()


def get_data(datatype, datayear='2020', combined_traindata=False):
    """ Select the dataset to use """
    if datatype == 'hasoc':
        if not combined_traindata:
            if datayear == '2020':
                print('Using Hasoc 2020 data... ')
                traindata = pd.read_csv(args.has20_traindata)
                devdata = pd.read_csv(args.has20_devdata)
                testdata = pd.read_csv(args.has20_testdata)
            else:
                print('Using Hasoc 2021 data... ')
                traindata = pd.read_csv(args.has21_traindata)
                devdata = pd.read_csv(args.has21_devdata)
                testdata = pd.read_csv(args.has21_testdata)
        else:
            data1a = pd.read_csv(args.has19_traindata)
            data2b = pd.read_csv(args.has19_devdata)
            data1aa = pd.read_csv(args.has20_traindata)
            data2bb = pd.read_csv(args.has20_devdata)
            data1aaa = pd.read_csv(args.has21_traindata)
            data2bbb = pd.read_csv(args.has21_devdata)
            traindata, devdata = pd.concat([data1a, data1aa, data1aaa]), pd.concat([data2b, data2bb, data2bbb])
            if datayear == '2020':
                print('Using Hasoc combined data for Hasoc 2020 test data... ')
                testdata = pd.read_csv(args.has20_testdata)
            else:
                print('Using Hasoc combined data for Hasoc 2021 test data... ')
                testdata = pd.read_csv(args.has21_testdata)     
    elif datatype =='sema':
        print('Using HatEval SemEval_A data... ')
        traindata = pd.read_csv(args.sema_traindata, sep='\t', header=0, encoding="latin1").fillna(method="ffill")
        devdata = pd.read_csv(args.sema_devdata, sep='\t', header=0, encoding="latin1").fillna(method="ffill")
        testdata = pd.DataFrame()   # empty DF since no testset
    elif datatype =='semb':
        print('Using HatEval SemEval_B data... ')
        traindata = pd.read_csv(args.semb_traindata, sep='\t', header=0, encoding="latin1").fillna(method="ffill")
        devdata = pd.read_csv(args.semb_devdata, sep='\t', header=0, encoding="latin1").fillna(method="ffill")
        testdata = pd.DataFrame()   # empty DF since no testset
    elif datatype == 'trol':
        print('Using Trol data... ')
        traindata = pd.read_csv(args.trol_traindata)
        devdata = pd.read_csv(args.trol_devdata)
        testdata = pd.DataFrame()   # empty DF since no testset
    elif datatype == 'hos':
        print('Using HOS data... ')
        traindata = pd.read_csv(args.hos_traindata)
        devdata = pd.read_csv(args.hos_devdata)
        testdata = pd.DataFrame()   # empty DF since no testset
    elif datatype == 'olid':
        print('Using OLID data... ')
        traindata = pd.read_csv(args.olid_traindata)
        devdata = pd.read_csv(args.olid_devdata)
        testdata = pd.DataFrame()   # empty DF since no testset
    else:
        print('Using No data... ')

    return traindata.drop_duplicates(keep='first'), devdata.drop_duplicates(keep='first'), testdata.drop_duplicates(keep='first')


def preprocess_pandas(data, columns):
    ''' <data> is a dataframe which contain  a <text> column  '''
    df_ = pd.DataFrame(columns=columns)
    df_ = data
    if args.datatype == 'trol':
        df_['Text'] = data['Text'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # remove emails
        df_['Text'] = data['Text'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # remove IP address
        df_['Text'] = data['Text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)          # remove URLs
        df_['Text'] = data['Text'].str.replace('[#,@,&,<,>,\,/,-]','')                                             # remove special characters
        df_['Text'] = data['Text'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)                           # remove emojis+
        df_['Text'] = data['Text'].str.replace('[','')
        df_['Text'] = data['Text'].str.replace(']','')
        df_['Text'] = data['Text'].str.replace('\n', ' ')
        df_['Text'] = data['Text'].str.replace('\t', ' ')
        df_['Text'] = data['Text'].str.replace(' {2,}', ' ', regex=True)                                           # remove 2 or more spaces
        df_['Text'] = data['Text'].str.lower()
        df_['Text'] = data['Text'].str.strip()
        df_['Text'] = data['Text'].replace('\d', '', regex=True)                                                   # remove numbers
        df_.drop_duplicates(subset=['Text'], keep='first')
        df_.dropna()
    elif args.datatype == 'hos' or args.datatype == 'olid':
        df_['tweet'] = data['tweet'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # remove emails
        df_['tweet'] = data['tweet'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # remove IP address
        df_['tweet'] = data['tweet'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)          # remove URLs
        df_['tweet'] = data['tweet'].str.replace('[#,@,&,<,>,\,/,-]','')                                             # remove special characters
        df_['tweet'] = data['tweet'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)                           # remove emojis+
        df_['tweet'] = data['tweet'].str.replace('[','')
        df_['tweet'] = data['tweet'].str.replace(']','')
        df_['tweet'] = data['tweet'].str.replace('\n', ' ')
        df_['tweet'] = data['tweet'].str.replace('\t', ' ')
        df_['tweet'] = data['tweet'].str.replace(' {2,}', ' ', regex=True)                                           # remove 2 or more spaces
        df_['tweet'] = data['tweet'].str.lower()
        df_['tweet'] = data['tweet'].str.strip()
        df_['tweet'] = data['tweet'].replace('\d', '', regex=True)                                                   # remove numbers
        df_.drop_duplicates(subset=['tweet'], keep='first')
        df_.dropna()
    else:               #args.datatype == 'hasoc' or args.datatype == 'sema' or args.datatype == 'semb':
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
