from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import argparse
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import pandas as pd
import re
import utility_hs as util


parser = argparse.ArgumentParser(description='Hate Speech Model')
# the modeldir line below should point to the folder where the model for the competition is saved
parser.add_argument('--modeldir', type=str, default='/home/oluade/e_tosano/t5basemodel_save', help='directory of the model checkpoint')
#parser.add_argument('--modeldir', type=str, default='/home/oluade/e_tosano/robasemodel_save', help='directory of the model checkpoint')

#parser.add_argument('--testdata2', type=str, default='English_2020/hasoc_2020_en_test_new.xlsx', help='location of the test data 2')
parser.add_argument('--has21_traindata', type=str, default='/home/shared_data/h/has21_traindata.csv', help='location of the training data')
parser.add_argument('--has21_testdata', type=str, default='/home/shared_data/h/has21_testdata.csv', help='location of the test data')

#parser.add_argument('--testdata3', type=str, default='/home/shared_data/h/en_Hasoc2021_test_task1.csv', help='location of the 2021 test data')
parser.add_argument('--testdata3gt', type=str, default='/home/shared_data/h/Hasoc_21_actuallabels.csv', help='location of the test data 3 ground truth')
parser.add_argument('--pred_ts', type=str, default='pred_testset_2021.csv', help='CSV output file of predictions')
parser.add_argument('--task_pref', type=str, default="binary classification: ", help='Task prefix')
parser.add_argument('--taskno', type=str, default="1", help='Task Number')
args = parser.parse_args()


def f1_score_func(preds, labels):
    preds_flat = []
    preds_flat_ = ['1' if a == '' or len(a) > 1 else a for a in preds]   # get rid of empty & lengthy predictions
    preds_flat.extend(preds_flat_)
    return f1_score(labels, preds_flat, average=None), f1_score(labels, preds_flat, average="weighted"), f1_score(labels, preds_flat, average="micro")


if __name__=="__main__":
    tokenizer = T5Tokenizer.from_pretrained(args.modeldir)
    model = T5ForConditionalGeneration.from_pretrained(args.modeldir)

    traindata = pd.read_csv(args.has21_traindata)
    test_data = pd.read_csv(args.has21_testdata)
    test_data3gt = pd.read_csv(args.testdata3gt)
    test_data = util.preprocess_pandas(test_data, list(test_data.columns))
    test_data['text'] = args.task_pref + test_data['text']
    predictions, tvals = [], []
    if not 'prediction' in test_data.columns:
        test_data.insert(len(test_data.columns), 'prediction', '')
    
    label_dict = {}             # For associating raw labels with indices/nos
    if args.taskno == '1':
        possible_labels = traindata.task_1.unique()
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        print(label_dict)       # {'NOT': '0', 'HOF': '1'}

        for rno in range(len(test_data['_id'])):
            input_ids = tokenizer(test_data['text'][rno], return_tensors='pt').input_ids
            outputs = model.generate(input_ids)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            test_data.loc[(test_data['_id'] == test_data['_id'][rno]) & (pred == '1'), 'prediction'] = 'HOF'
            test_data.loc[(test_data['_id'] == test_data['_id'][rno]) & (pred == '0'), 'prediction'] = 'NOT'
            predictions.append(pred)        # store the predictions in a list

        test_data3gt['label'] = test_data3gt.label.replace(label_dict)                 # replace labels with their nos
        test_data3gt['label'] = test_data3gt['label'].apply(str)  # string conversion
        tvals = test_data3gt['label'].values.tolist()

    else:
        possible_labels = traindata.task_2.unique()
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        print(label_dict)       # {'NONE': '0', 'PRFN': '1', 'OFFN': '2', 'HATE': '3'}

        for rno in range(len(test_data['_id'])):
            input_ids = tokenizer(test_data['text'][rno], return_tensors='pt').input_ids
            outputs = model.generate(input_ids)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            test_data.loc[(test_data['_id'] == test_data['_id'][rno]) & (pred == '0'), 'prediction'] = 'NONE'
            test_data.loc[(test_data['_id'] == test_data['_id'][rno]) & (pred == '1'), 'prediction'] = 'PRFN'
            test_data.loc[(test_data['_id'] == test_data['_id'][rno]) & (pred == '2'), 'prediction'] = 'OFFN'
            test_data.loc[(test_data['_id'] == test_data['_id'][rno]) & (pred == '3'), 'prediction'] = 'HATE'
            predictions.append(pred)        # store the predictions in a list
    
    # the below takes some time (several minutes) to complete
    f1, f1_w, f1_m = f1_score_func(predictions, tvals)
    print(f'F1: {f1}, weighted F1: {f1_w}, micro F1: {f1_m}')
    print(test_data.head())
    test_data.to_csv(args.pred_ts, index=False)
