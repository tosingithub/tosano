from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import argparse
from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef, accuracy_score
import pandas as pd
import numpy as np
import re
import utility_hs as util


parser = argparse.ArgumentParser(description='Hate Speech Model')
# --datatype: hasoc, trol, sema, semb, hos, olid (--olidtask = a, b or c)
# USAGE EXAMPLE in the terminal: python ht5.py --datatype trol
parser.add_argument('--datatype', type=str, default='pcl', help='data of choice')
parser.add_argument('--modeldir', type=str, default='/home/oluade/e_tosano/t5base_pcl', help='directory of the model checkpoint')

#parser.add_argument('--testdata2', type=str, default='English_2020/hasoc_2020_en_test_new.xlsx', help='location of the test data 2')
parser.add_argument('--pcl_train', type=str, default='/home/oluade/e_tosano/semeval/task4_train20.csv', help='location of the training data')
parser.add_argument('--pcl_test', type=str, default='/home/oluade/e_tosano/semeval/task4_test.tsv', help='location of the dev data')
#parser.add_argument('--pcl_test', type=str, default='/home/oluade/e_tosano/semeval/task4_test.tsv', help='location of the test data') # has21_testdata
parser.add_argument('--testdata3gt', type=str, default='/home/shared_data/h/Hasoc_21_actuallabels.csv', help='location of the test data 3 ground truth')
parser.add_argument('--testdata3gt2', type=str, default='/home/shared_data/h/has21_testwithlabels.csv', help='location of the test data 3 task 2 ground truth')
parser.add_argument('--outf_path', type=str, default='semeval/task1_20small.txt', help='output file of predictions')
parser.add_argument('--task_pref', type=str, default="classification: ", help='Task prefix')
parser.add_argument('--pred_ts', type=str, default='task_pcl', help='CSV output file of predictions')
parser.add_argument('--pred_txt', type=str, default='task1p.txt', help='TXT output file of predictions')
parser.add_argument('--taskno', type=str, default="1", help='Task Number')
args = parser.parse_args()


def f1_score_func(preds, labels):
    preds_flat = []
    preds_flat = ['0' if a == '' or len(a) > 1 else a for a in preds]   # TODO: USE LARGEST LABEL; get rid of empty & lengthy predictions
    return f1_score(labels, preds_flat, average=None), f1_score(labels, preds_flat, average="weighted"), f1_score(labels, preds_flat, average="macro")

def confusion_matrix_func(preds, labels):
    #if args.datatype == 'hasoc':
    preds_flat = []
    preds_flat = ['0' if a == '' or len(a) > 1 else a for a in preds]   # get rid of empty & lengthy predictions
    print(confusion_matrix(labels, preds_flat, labels=['0', '1']))
    tn, fp, fn, tp = confusion_matrix(labels, preds_flat, labels=['0', '1']).ravel()
    return tn, fp, fn, tp


def preprocess_pandas(data, columns):
    df_ = pd.DataFrame(columns=columns)
    df_ = data
    if args.datatype == 'pcl':
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
        df_ = df_.dropna()
    return df_


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(args.modeldir)
    model = T5ForConditionalGeneration.from_pretrained(args.modeldir)

    traindata = pd.read_csv(args.pcl_train)
    test_data = pd.read_csv(args.pcl_test, header=None, sep='\t')
    test_data.columns =['tno', 'art_id', 'keyword', 'country_code', 'text']
    #print("Total in Test set: ", len(test_data))
    #test_data = pd.read_csv(args.pcl_test)
    #test_data3gt = pd.read_csv(args.pcl_test)
    #print(test_data3gt.head())

    test_data = preprocess_pandas(test_data, list(test_data.columns))
    print("Total after : ", len(test_data))

    test_data['text'] = args.task_pref + test_data['text']
    #test_data = test_data[:10]

    cnter = 0
    predictions, tvals = [], []
    # if not 'prediction' in test_data.columns:
    #     test_data.insert(len(test_data.columns), 'prediction', '')
    
    label_dict = {}             # For associating raw labels with indices/nos
    possible_labels = traindata.label.unique()
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    print(label_dict)       # {0: 0, 1: 1} DANGER - Always check this for each run
    for rno in range(len(test_data)):
        #print(rno)
        input_ids = tokenizer(test_data['text'][rno], padding=True, truncation=True, return_tensors='pt').input_ids
        outputs = model.generate(input_ids)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(pred)
        if pred == '':
            pred = '0'
            cnter += 1
        elif len(pred) > 1:
            pred = '0'
            cnter += 1
        predictions.append(pred)        # store the predictions in a list
        #with open(args.outf_path,'a+') as outf:
    	    #for pi in p:
            #outf.write(pred+'\n')
    print("Counter: ", cnter)
    # test_data3gt['label'] = test_data3gt.label.replace(label_dict)                 # replace labels with their nos
    # test_data3gt['label'] = test_data3gt['label'].apply(str)  # string conversion
    # tvals = test_data3gt['label'].values.tolist()

    # res_df = pd.DataFrame(predictions)
    # print("Total Result after : ", len(res_df))
    # res_df.to_csv(args.pred_ts, index=False)
    # np_array = res_df.to_numpy()
    # np.savetxt(args.pred_txt, np_array, fmt = "%s")


    # # the below takes some time (several minutes) to complete
    # f1, f1_w, f1_m = f1_score_func(predictions, tvals)
    # print('tn, fp, fn, tp ', confusion_matrix_func(predictions, tvals))
    # print(f'F1: {f1}, weighted F1: {f1_w}, macro F1: {f1_m}')
    # print(test_data.head())
    # test_data.to_csv(args.pred_ts, index=False)
