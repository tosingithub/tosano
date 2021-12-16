from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, get_linear_schedule_with_warmup, get_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from datasets import load_dataset, load_metric, list_metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import os
from tqdm.auto import tqdm
import pandas as pd
import argparse
import time
import pickle
import numpy as np
import re


# commandline arguments
""" If you are running this in a jupyter notebook then you need to change the parser lines below
to equality e.g. traindata = "english_dataset/english_dataset.tsv" and then remove args. where applicable
"""
parser = argparse.ArgumentParser(description='Hate Speech Model')
# --datatype: hasoc, ...
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
# Additional datasets

parser.add_argument('--task_pref', type=str, default="binary classification: ", help='Task prefix')
parser.add_argument('--datayear', type=str, default="2020", help='Data year')
parser.add_argument('--taskno', type=str, default="2", help='Task Number')
parser.add_argument('--savet', type=str, default='modelt5base_hasoc_task1a.pt', help='filename of the model checkpoint')
parser.add_argument('--pikle', type=str, default='modelt5base_hasoc_task1a.pkl', help='pickle filename of the model checkpoint')
parser.add_argument('--msave', type=str, default='t5basemodel_save/', help='folder to save the finetuned model')
parser.add_argument('--ofile1', type=str, default='outputfile_task1.txt', help='output file')
parser.add_argument('--ofile2', type=str, default='outputfile_task2.txt', help='output file')
parser.add_argument('--submission1', type=str, default='submitfile_task1a.csv', help='submission file')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=2, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size') # smaller batch size for big model to fit GPU
args = parser.parse_args()

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
    

def f1_score_func(preds, labels):
    preds_flat = []
    preds_flat_ = ['0' if a == '' or len(a) > 1 else a for a in preds]   # get rid of empty & lengthy predictions
    preds_flat.extend(preds_flat_)
    return f1_score(labels, preds_flat, average=None), f1_score(labels, preds_flat, average="weighted"), f1_score(labels, preds_flat, average="micro")

# def accuracy_score_func(preds, labels):
#     preds_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()
#     return accuracy_score(labels_flat, preds_flat, normalize='False')

def confusion_matrix_func(preds, labels):
    preds_flat = []
    preds_flat_ = ['0' if a == '' or len(a) > 1 else a for a in preds]   # get rid of empty & lengthy predictions
    preds_flat.extend(preds_flat_)
    print(confusion_matrix(labels, preds_flat))


def train(train_data, train_tags):
    """One epoch of a training loop"""
    print("Training...")
    epoch_loss, train_steps, train_loss = 0, 0, 0

    # tokenizer.encode() converts the text to a list of unique integers before returning tensors
    einput_ids = tokenizer(train_data, padding=True, truncation=True, return_tensors='pt')
    input_ids, attention_mask = einput_ids.input_ids, einput_ids.attention_mask
    labels = tokenizer(train_tags, padding=True, truncation=True, return_tensors='pt').input_ids

    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_mask = attention_mask.to(device)

    train_tensor = TensorDataset(input_ids, attention_mask, labels)
    train_sampler = RandomSampler(train_tensor)
    train_dataloader = DataLoader(train_tensor, sampler=train_sampler, batch_size=args.batch_size)

    model.train()  # Turn on training mode
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        batch_input_ids, batch_att_mask, batch_labels = batch
        loss = model(input_ids=batch_input_ids, attention_mask=batch_att_mask, labels=batch_labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        train_steps += 1
    epoch_loss = train_loss / train_steps
    return epoch_loss


def evaluate(val_data, val_tags):
    """One epoch of an evaluation loop"""
    print("Evaluation...")
    epoch_loss, val_steps, val_loss = 0, 0, 0

    # tokenizer.encode() converts the text to a list of unique integers before returning tensors
    einput_ids = tokenizer(val_data, padding=True, return_tensors='pt')
    input_ids, attention_mask = einput_ids.input_ids, einput_ids.attention_mask
    labels = tokenizer(val_tags, padding=True, return_tensors='pt').input_ids

    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_mask = attention_mask.to(device)

    val_tensor = TensorDataset(input_ids, attention_mask, labels) #, ids)
    val_sampler = SequentialSampler(val_tensor)
    val_dataloader = DataLoader(val_tensor, sampler=val_sampler, batch_size=args.batch_size)
    predictions = []

    model.eval()  # Turn on evaluation mode
    for batch in tqdm(val_dataloader):
        with torch.no_grad():
            batch_input_ids, batch_att_mask, batch_labels = batch
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_att_mask, labels=batch_labels)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1)
            prediction = tokenizer.batch_decode(prediction, skip_special_tokens=True)
            for a in prediction:            # pick each element - no list comprehension
                predictions.append(a)

            val_loss += outputs.loss.item()
            val_steps += 1
    true_vals = val_tags
    epoch_loss = val_loss / val_steps
    return epoch_loss, predictions, true_vals

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

# def random_seeding(seed_value, device):                           # set for reproducibility
#     #numpy.random.seed(seed_value)
#     #random.seed(seed_value)
#     torch.manual_seed(seed_value)
#     if device == "cuda": torch.cuda.manual_seed_all(seed_value)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    tokenizer.pad_token = tokenizer.eos_token # to avoid an error
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) #, betas=(0.7, 0.99))
    
    # get_data has the following args: (datatype, datayear='2020', combined_traindata=False)
    # datayear: 2020 or 2021;
    traindata, devdata, testdata = get_data(args.datattype, args.datayear, combined_traindata=False)
    # Comment out the below if preprocessing not needed
    traindata = preprocess_pandas(traindata, list(traindata.columns))
    valdata = preprocess_pandas(devdata, list(devdata.columns))
    test_data = preprocess_pandas(testdata, list(testdata.columns))

    #Add task prefix for T5 better performance
    traindata['text'] = args.task_pref + traindata['text']
    valdata['text'] = args.task_pref + traindata['text']
    test_data['text'] = args.task_pref + test_data['text']

    train_data = traindata['text'].values.tolist()
    val_data = valdata['text'].values.tolist()
    test_data_texts = test_data['text'].values.tolist()

    outfile = ''
    label_dict = {}         # For associating raw labels with indices/nos
    if args.datatype == 'hasoc' and args.taskno == '1':
        possible_labels = traindata.task_1.unique()
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        print(label_dict)       # NOT: 0; HOF: 1

        traindata['task_1'] = traindata.task_1.replace(label_dict)                 # replace labels with their nos
        traindata['task_1'] = traindata['task_1'].apply(str)  # string conversion
        valdata['task_1'] = valdata.task_1.replace(label_dict)                 # replace labels with their nos
        valdata['task_1'] = valdata['task_1'].apply(str)  # string conversion
        test_data['task_1'] = test_data.task_1.replace(label_dict)                 # replace labels with their nos
        test_data['task_1'] = test_data['task_1'].apply(str)  # string conversion

        train_tags = traindata['task_1'].values.tolist()

        #valdata['_id'] = valdata['_id'].apply(str)    # string conversion
        #val_ids = valdata['_id'].values.tolist()
        val_tags = valdata['task_1'].values.tolist()

        #test_data['_id'] = test_data['_id'].apply(str)  # string conversion
        #test_ids = test_data['_id'].values.tolist()
        test_data_labels = test_data['task_1'].values.tolist()
        outfile = args.ofile1

    elif args.datatype == 'hasoc' and args.taskno == '2':
        possible_labels = traindata.task_2.unique()
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        print(label_dict)       # for sanity check

        traindata['task_2'] = traindata.task_1.replace(label_dict)                 # replace labels with their nos
        traindata['task_2'] = traindata['task_2'].apply(str)  # string conversion
        valdata['task_2'] = valdata.task_1.replace(label_dict)                 # replace labels with their nos
        valdata['task_2'] = valdata['task_2'].apply(str)  # string conversion
        test_data['task_2'] = test_data.task_1.replace(label_dict)                 # replace labels with their nos
        test_data['task_2'] = test_data['task_2'].apply(str)  # string conversion

        train_tags = traindata['task_2'].values.tolist()

        #valdata['_id'] = valdata['_id'].apply(str)    # string conversion
        #val_ids = valdata['_id'].values.tolist()
        val_tags = valdata['task_2'].values.tolist()

        #test_data['_id'] = test_data['_id'].apply(str)  # string conversion
        #test_ids = test_data['_id'].values.tolist()
        test_data_labels = test_data['task_2'].values.tolist()
        outfile = args.ofile2

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_data)*args.epochs)


    best_val_wf1 = None
    best_model = None
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(train_data, train_tags)
        val_loss, predictions, true_vals = evaluate(val_data, val_tags) # val_ids added for Hasoc submission
        val_f1, val_f1_w, val_f1_mic = f1_score_func(predictions, true_vals)
        epoch_time_elapsed = time.time() - epoch_start_time
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f} '.format(epoch, train_loss, val_loss) + f'F1: {val_f1}, weighted F1: {val_f1_w}, micro F1: {val_f1_mic}') # metric_sc['f1']))        
        with open(outfile, "a+") as f:
            s = f.write('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f} '.format(epoch, train_loss, val_loss) + f'F1: {val_f1}, weighted F1: {val_f1_w}, micro F1: {val_f1_mic}' + "\n")
        if not best_val_wf1 or val_f1_w > best_val_wf1:
            with open(args.savet, 'wb') as f:        # create file but deletes implicitly 1st if already exists
                #No need to save the models for now so that they don't use up space 
                #torch.save(model.state_dict(), f)    # save best model's learned parameters (based on lowest loss)
                best_model = model
                #model_to_save = model.module if hasattr(model, 'module') else model
                #model_to_save.save_pretrained(args.msave)  # transformers save
                #tokenizer.save_pretrained(args.msave)

            #with open(args.pikle, 'wb') as file:    # save the classifier as a pickle file
                #pickle.dump(model, file)
            best_val_wf1 = val_f1_w
    
    # Hasoc 2021 test set will be run according to Hasoc format in order to prepare, so...
    if args.datatype == 'hasoc' and not args.datayear == '2021':
        model = best_model
        eval_loss, predictions, true_vals = evaluate(test_data_texts, test_data_labels) # test_ids added for Hasoc submission
        eval_f1, eval_f1_w, eval_f1_mic = f1_score_func(predictions, true_vals)
        print('Test Loss: {:.4f} '.format(eval_loss) + f'F1: {eval_f1}, weighted F1: {eval_f1_w}, micro F1: {eval_f1_mic}')       
        with open(outfile, "a+") as f:
            s = f.write('Test Loss: {:.4f} '.format(eval_loss) + f'F1: {eval_f1}, weighted F1: {eval_f1_w}, micro F1: {eval_f1_mic}' + "\n")
