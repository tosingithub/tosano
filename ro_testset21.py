import torch
import argparse
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import pandas as pd
import re
import utility_hs as util
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup


parser = argparse.ArgumentParser(description='Hate Speech Model')
parser.add_argument('--datatype', type=str, default='olid', help='data of choice')
parser.add_argument('--olidtask', type=str, default="a", help='Task Alphabet')      # a, b or c
# the modeldir line below should point to the folder where the model for the competition is saved
parser.add_argument('--modeldir', type=str, default='/home/oluade/e_tosano/robase_olidb', help='directory of the model checkpoint')
#parser.add_argument('--testdata2', type=str, default='English_2020/hasoc_2020_en_test_new.xlsx', help='location of the test data 2')
parser.add_argument('--has21_traindata', type=str, default='/home/shared_data/h/has21_traindata.csv', help='location of the training data')
parser.add_argument('--has21_testdata', type=str, default='/home/shared_data/h/has21_testdata.csv', help='location of the test data')
parser.add_argument('--olid_traindata', type=str, default='/home/shared_data/h/OLIDv1.0/olid-training.csv', help='location of the training data')
parser.add_argument('--olid_testa', type=str, default='/home/shared_data/h/OLIDv1.0/testset-levela.tsv', help='location of the test data ground truth')
parser.add_argument('--olid_testb', type=str, default='/home/shared_data/h/OLIDv1.0/testset-levelb.tsv', help='location of the test data ground truth')
parser.add_argument('--olid_testc', type=str, default='/home/shared_data/h/OLIDv1.0/testset-levelc.tsv', help='location of the test data ground truth')
parser.add_argument('--testolidgta', type=str, default='/home/shared_data/h/OLIDv1.0/labels-levela.csv', help='location of the test data ground truth')
parser.add_argument('--testolidgtb', type=str, default='/home/shared_data/h/OLIDv1.0/labels-levelb.csv', help='location of the test data ground truth')
parser.add_argument('--testolidgtc', type=str, default='/home/shared_data/h/OLIDv1.0/labels-levelc.csv', help='location of the test data ground truth')

#parser.add_argument('--testdata3', type=str, default='/home/shared_data/h/en_Hasoc2021_test_task1.csv', help='location of the 2021 test data')
parser.add_argument('--testdata3gt', type=str, default='/home/shared_data/h/Hasoc_21_actuallabels.csv', help='location of the test data 3 ground truth')
parser.add_argument('--testdata3gt2', type=str, default='/home/shared_data/h/has21_testwithlabels.csv', help='location of the test data 3 task 2 ground truth')
parser.add_argument('--pred_ts', type=str, default='pred_roberta', help='CSV output file of predictions')
parser.add_argument('--taskno', type=str, default="1", help='Task Number')
args = parser.parse_args()


def f1_score_func(preds_flat, labels):
    return f1_score(labels, preds_flat, average=None), f1_score(labels, preds_flat, average="weighted"), f1_score(labels, preds_flat, average="macro")


if __name__=="__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(args.modeldir, truncation=True, do_lower_case=True)
    model = RobertaForSequenceClassification.from_pretrained(args.modeldir)
    print("No of classes: ", model.num_labels)

    if args.datatype == 'hasoc':
        traindata = pd.read_csv(args.has21_traindata)
        test_data = pd.read_csv(args.has21_testdata)
        if args.taskno == '1': test_data3gt = pd.read_csv(args.testdata3gt)
        else: test_data3gt = pd.read_csv(args.testdata3gt2)
    else:
        traindata = pd.read_csv(args.olid_traindata)
        if args.olidtask == 'a':
            test_data = pd.read_csv(args.olid_testa, sep='\t')
            test_data3gt = pd.read_csv(args.testolidgta, header=None)
        elif args.olidtask == 'b':
            test_data = pd.read_csv(args.olid_testb, sep='\t')
            test_data3gt = pd.read_csv(args.testolidgtb, header=None)
        else:
            test_data = pd.read_csv(args.olid_testc, sep='\t')
            test_data3gt = pd.read_csv(args.testolidgtc, header=None)

    test_data = util.preprocess_pandas(test_data, list(test_data.columns))
    # if args.datatype == 'hasoc': test_data['text'] = args.task_pref + test_data['text']
    # else: test_data['tweet'] = args.task_pref + test_data['tweet']

    predictions, tvals = [], []
    if not 'prediction' in test_data.columns:
        test_data.insert(len(test_data.columns), 'prediction', '')
    
    label_dict = {}             # For associating raw labels with indices/nos
    if args.datatype == 'hasoc':
        if args.taskno == '1':
            args.pred_ts = args.pred_ts + '_hasa.csv'
            possible_labels = traindata.task_1.unique()
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index
            print(label_dict)       # {'HOF': 0, 'NOT': 1}
            for rno in range(len(test_data['_id'])):
                input_ids = tokenizer(test_data['text'][rno], add_special_tokens=True, padding=True, truncation=True, return_tensors='pt').input_ids
                outputs = model(input_ids)
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                predictions.append(logits)        # store the predictions in a list
            predictions = np.concatenate(predictions, axis=0)
            preds_flat = np.argmax(predictions, axis=1).flatten()
            for rno in range(len(test_data['_id'])):
                #pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
                test_data.loc[(test_data['_id'] == test_data['_id'][rno]) & (preds_flat[rno] == 0), 'prediction'] = 'HOF'
                test_data.loc[(test_data['_id'] == test_data['_id'][rno]) & (preds_flat[rno] == 1), 'prediction'] = 'NOT'
            test_data3gt['label'] = test_data3gt.label.replace(label_dict)                 # replace labels with their nos
            tvals = test_data3gt['label'].values.tolist()
        else:
            args.pred_ts = args.pred_ts + '_hasb.csv'
            possible_labels = traindata.task_2.unique()
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index
            print(label_dict)       # {'NONE': '0', 'PRFN': '1', 'OFFN': '2', 'HATE': '3'}
            for rno in range(len(test_data['_id'])):
                input_ids = tokenizer(test_data['text'][rno], add_special_tokens=True, padding=True, truncation=True, return_tensors='pt').input_ids
                outputs = model(input_ids)
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                predictions.append(logits)        # store the predictions in a list
            predictions = np.concatenate(predictions, axis=0)
            preds_flat = np.argmax(predictions, axis=1).flatten()
            for rno in range(len(test_data['_id'])):
                test_data.loc[(test_data['_id'] == test_data['_id'][rno]) & (preds_flat[rno] == 0), 'prediction'] = 'NONE'
                test_data.loc[(test_data['_id'] == test_data['_id'][rno]) & (preds_flat[rno] == 1), 'prediction'] = 'PRFN'
                test_data.loc[(test_data['_id'] == test_data['_id'][rno]) & (preds_flat[rno] == 2), 'prediction'] = 'OFFN'
                test_data.loc[(test_data['_id'] == test_data['_id'][rno]) & (preds_flat[rno] == 3), 'prediction'] = 'HATE'
            test_data3gt['task_2'] = test_data3gt.task_2.replace(label_dict)                 # replace labels with their nos
            tvals = test_data3gt['task_2'].values.tolist()
    else:
        if args.olidtask == 'a':
            args.pred_ts = args.pred_ts + '_olida.csv'
            possible_labels = traindata['subtask_a'].unique()
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index
            print(label_dict)       # {'NOT': '0', 'OFF': '1'}
            for rno in range(len(test_data['id'])):
                input_ids = tokenizer(test_data['tweet'][rno], add_special_tokens=True, padding=True, truncation=True, return_tensors='pt').input_ids
                outputs = model(input_ids)
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                predictions.append(logits)        # store the predictions in a list
            predictions = np.concatenate(predictions, axis=0)
            preds_flat = np.argmax(predictions, axis=1).flatten()
            for rno in range(len(test_data['id'])):
                test_data.loc[(test_data['id'] == test_data['id'][rno]) & (preds_flat[rno] == 0), 'prediction'] = 'NOT'
                test_data.loc[(test_data['id'] == test_data['id'][rno]) & (preds_flat[rno] == 1), 'prediction'] = 'OFF'
            test_data3gt[1] = test_data3gt[1].replace(label_dict)                 # replace labels with their nos
            tvals = test_data3gt[1].values.tolist()
        elif args.olidtask == 'b':
            args.pred_ts = args.pred_ts + '_olidb.csv'
            possible_labels = traindata['subtask_b'].unique()
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index
            print(label_dict)       # {'TIN': 0, 'UNT': 1}
            for rno in range(len(test_data['id'])):
                input_ids = tokenizer(test_data['tweet'][rno], add_special_tokens=True, padding=True, truncation=True, return_tensors='pt').input_ids
                outputs = model(input_ids)
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                predictions.append(logits)        # store the predictions in a list
            predictions = np.concatenate(predictions, axis=0)
            preds_flat = np.argmax(predictions, axis=1).flatten()
            for rno in range(len(test_data['id'])):
                test_data.loc[(test_data['id'] == test_data['id'][rno]) & (preds_flat[rno] == 0), 'prediction'] = 'TIN'
                test_data.loc[(test_data['id'] == test_data['id'][rno]) & (preds_flat[rno] == 1), 'prediction'] = 'UNT'
            test_data3gt[1] = test_data3gt[1].replace(label_dict)                 # replace labels with their nos
            tvals = test_data3gt[1].values.tolist()
        else:
            args.pred_ts = args.pred_ts + '_olidc.csv'
            possible_labels = traindata['subtask_c'].unique()
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index
            print(label_dict)       # {'GRP': 0, 'IND': 1, 'OTH': 2}
            for rno in range(len(test_data['id'])):
                input_ids = tokenizer(test_data['tweet'][rno], add_special_tokens=True, padding=True, truncation=True, return_tensors='pt').input_ids
                outputs = model(input_ids)
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                predictions.append(logits)        # store the predictions in a list
            predictions = np.concatenate(predictions, axis=0)
            preds_flat = np.argmax(predictions, axis=1).flatten()
            for rno in range(len(test_data['id'])):
                test_data.loc[(test_data['id'] == test_data['id'][rno]) & (preds_flat[rno] == 0), 'prediction'] = 'GRP'
                test_data.loc[(test_data['id'] == test_data['id'][rno]) & (preds_flat[rno] == 1), 'prediction'] = 'IND'
                test_data.loc[(test_data['id'] == test_data['id'][rno]) & (preds_flat[rno] == 1), 'prediction'] = 'OTH'
            test_data3gt[1] = test_data3gt[1].replace(label_dict)                 # replace labels with their nos
            tvals = test_data3gt[1].values.tolist()


    # the below takes some time (several minutes) to complete
    f1, f1_w, f1_m = f1_score_func(preds_flat, tvals)
    print(f'F1: {f1}, weighted F1: {f1_w}, macro F1: {f1_m}')
    print(test_data.head())
    test_data.to_csv(args.pred_ts, index=False)
