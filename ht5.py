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
import utility_hs as util


# commandline arguments
""" If you are running this in a jupyter notebook then you need to change the parser lines below
to equality e.g. traindata = "english_dataset/english_dataset.tsv" and then remove args. where applicable
"""
parser = argparse.ArgumentParser(description='Hate Speech Model')
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

parser.add_argument('--task_pref', type=str, default="classification: ", help='Task prefix')
parser.add_argument('--datayear', type=str, default="2021", help='Data year')           # 2020 or 2021
parser.add_argument('--taskno', type=str, default="1", help='Task Number')              # 1 or 2
parser.add_argument('--olidtask', type=str, default="a", help='Task Alphabet')      # a, b or c
parser.add_argument('--savet', type=str, default='modelt5base_hasoc_task1a.pt', help='filename of the model checkpoint')
parser.add_argument('--pikle', type=str, default='modelt5base_hasoc_task1a.pkl', help='pickle filename of the model checkpoint')
parser.add_argument('--msave', type=str, default='t5basemodel_save/', help='folder to save the finetuned model')
parser.add_argument('--ofile1', type=str, default='outputfile_', help='output file')
parser.add_argument('--ofile2', type=str, default='outputfile_', help='output file')
parser.add_argument('--submission1', type=str, default='submitfile_task1a.csv', help='submission file')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate') # bestloss at 0.0002; dpred- 1; weighted F1: 0.9386351943374753, micro F1: 0.9376623376623376; test #weighted F1: 0.8210865645981863, micro F1: 0.8227946916471507
# task_pref: classification
# bestloss at 0.0002; Validation Loss: 0.1557 weighted F1: 0.9589171159419094, micro F1: 0.958441558441558; test: F1: [0.87049083 0.74856487], weighted F1: 0.824518748338588, micro F1: 0.8290398126463701
parser.add_argument('--epochs', type=int, default=6, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size') # smaller batch size for big model to fit GPU
args = parser.parse_args()


def f1_score_func(preds, labels):
    preds_flat = []
    preds_flat_ = ['1' if a == '' or len(a) > 1 else a for a in preds]   # get rid of empty & lengthy predictions
    preds_flat.extend(preds_flat_)
    labels_flat = labels            # only for consistency
    return f1_score(labels_flat, preds_flat, average=None), f1_score(labels_flat, preds_flat, average="weighted"), f1_score(labels_flat, preds_flat, average="micro")

# def accuracy_score_func(preds, labels):
#     preds_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()
#     return accuracy_score(labels_flat, preds_flat, normalize='False')

def confusion_matrix_func(preds, labels):
    #if args.datatype == 'hasoc':
    preds_flat = []
    preds_flat_ = ['1' if a == '' or len(a) > 1 else a for a in preds]   # get rid of empty & lengthy predictions
    preds_flat.extend(preds_flat_)
    labels_flat = labels
    # else:
    #     preds_flat = np.argmax(preds, axis=1).flatten()
    #     labels_flat = labels.flatten()
    print(confusion_matrix(labels_flat, preds_flat))


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
    
    traindata, devdata, testdata = util.get_data(args.datatype, args.datayear, combined_traindata=False)
    # Comment out the below if preprocessing not needed
    traindata = util.preprocess_pandas(traindata, list(traindata.columns))
    valdata = util.preprocess_pandas(devdata, list(devdata.columns))
    if args.datatype == 'hasoc': # not args.datatype == 'trol' or not args.datatype == 'sema' or not args.datatype == 'semb':                                         # skip line below for trol, 
        test_data = util.preprocess_pandas(testdata, list(testdata.columns))

    ### Text column
    # Add task prefix for T5 better performance
    if args.datatype == 'hasoc':
        traindata['text'] = args.task_pref + traindata['text']
        valdata['text'] = args.task_pref + valdata['text']
        test_data['text'] = args.task_pref + test_data['text']
        train_data = traindata['text'].values.tolist()
        val_data = valdata['text'].values.tolist()
        if not args.datayear == '2021':                                         # skip line below for 2021, 
            test_data_texts = test_data['text'].values.tolist()
    elif args.datatype == 'trol':
        traindata['Text'] = args.task_pref + traindata['Text']
        valdata['Text'] = args.task_pref + valdata['Text']
        train_data = traindata['Text'].values.tolist()
        val_data = valdata['Text'].values.tolist()
    elif args.datatype == 'hos' or args.datatype == 'olid':
        traindata['tweet'] = args.task_pref + traindata['tweet']
        valdata['tweet'] = args.task_pref + valdata['tweet']
        train_data = traindata['tweet'].values.tolist()
        val_data = valdata['tweet'].values.tolist()
    else:                                                               # for sema, semb
        traindata['text'] = args.task_pref + traindata['text']
        valdata['text'] = args.task_pref + valdata['text']
        train_data = traindata['text'].values.tolist()
        val_data = valdata['text'].values.tolist()

    ### Label column
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
        print("Trainset distribution: \n", traindata['task_1'].value_counts())                           # check data distribution
        if args.datatype == 'hasoc' and not args.datayear == '2021':            # we'll do 2021 inference on testset elsewhere
            test_data['task_1'] = test_data.task_1.replace(label_dict)                 # replace labels with their nos
            test_data['task_1'] = test_data['task_1'].apply(str)  # string conversion
            test_data_labels = test_data['task_1'].values.tolist()

        train_tags = traindata['task_1'].values.tolist()
        val_tags = valdata['task_1'].values.tolist()
        outfile = args.ofile1 + 'task1_'

    elif args.datatype == 'hasoc' and args.taskno == '2':
        possible_labels = traindata.task_2.unique()
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        print(label_dict)       # for sanity check {'NONE': 0, 'PRFN': 1, 'OFFN': 2, 'HATE': 3}

        traindata['task_2'] = traindata.task_2.replace(label_dict)                 # replace labels with their nos
        traindata['task_2'] = traindata['task_2'].apply(str)  # string conversion
        valdata['task_2'] = valdata.task_2.replace(label_dict)                 # replace labels with their nos
        valdata['task_2'] = valdata['task_2'].apply(str)  # string conversion
        print("Trainset distribution: \n", traindata['task_2'].value_counts())                           # check data distribution
        if args.datatype == 'hasoc' and not args.datayear == '2021':            # we'll do 2021 inference on testset elsewhere
            test_data['task_2'] = test_data.task_2.replace(label_dict)                 # replace labels with their nos
            test_data['task_2'] = test_data['task_2'].apply(str)  # string conversion
            test_data_labels = test_data['task_2'].values.tolist()

        train_tags = traindata['task_2'].values.tolist()
        val_tags = valdata['task_2'].values.tolist()
        outfile = args.ofile2 + 'task2_'
    elif args.datatype == 'trol':
        possible_labels = traindata['Sub-task A'].unique()
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        print(label_dict)       # for sanity check {'NAG': 0, 'CAG': 1, 'OAG': 2}
        traindata['Sub-task A'] = traindata['Sub-task A'].replace(label_dict)                 # replace labels with their nos
        traindata['Sub-task A'] = traindata['Sub-task A'].apply(str)  # string conversion
        valdata['Sub-task A'] = valdata['Sub-task A'].replace(label_dict)                 # replace labels with their nos
        valdata['Sub-task A'] = valdata['Sub-task A'].apply(str)  # string conversion
        print("Trainset distribution: \n", traindata['Sub-task A'].value_counts())                           # check data distribution
        train_tags = traindata['Sub-task A'].values.tolist()
        val_tags = valdata['Sub-task A'].values.tolist()
        outfile = args.ofile2 + 'trol_'
    elif args.datatype == 'hos':
        traindata['hate_speech'] = traindata['hate_speech'].apply(str)  # string conversion
        valdata['hate_speech'] = valdata['hate_speech'].apply(str)  # string conversion
        print("Trainset distribution: \n", traindata['hate_speech'].value_counts())                           # check data distribution
        train_tags = traindata['hate_speech'].values.tolist()
        val_tags = valdata['hate_speech'].values.tolist()
        outfile = args.ofile2 + 'hos_'
    elif args.datatype == 'sema' or args.datatype == 'semb':
        # labels are already numeric
        traindata['HS'] = traindata['HS'].apply(str)  # string conversion
        valdata['HS'] = valdata['HS'].apply(str)  # string conversion
        print("Trainset distribution: \n", traindata['HS'].value_counts())                           # check data distribution
        train_tags = traindata['HS'].values.tolist()
        val_tags = valdata['HS'].values.tolist()
        outfile = args.ofile2 + 'sem_'
    elif args.datatype == 'olid':
        if args.olidtask == 'a':
            possible_labels = traindata['subtask_a'].unique()
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index
            print(label_dict)       # for sanity check 
            traindata['subtask_a'] = traindata['subtask_a'].replace(label_dict)                 # replace labels with their nos
            traindata['subtask_a'] = traindata['subtask_a'].apply(str)  # string conversion
            valdata['subtask_a'] = valdata['subtask_a'].replace(label_dict)                 # replace labels with their nos
            valdata['subtask_a'] = valdata['subtask_a'].apply(str)  # string conversion
            print("Trainset distribution: \n", traindata['subtask_a'].value_counts())                           # check data distribution
            train_tags = traindata['subtask_a'].values.tolist()
            val_tags = valdata['subtask_a'].values.tolist()
            outfile = args.ofile2 + 'olida_'
        elif args.olidtask == 'b':
            possible_labels = traindata['subtask_b'].unique()
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index
            print(label_dict)       # for sanity check 
            traindata['subtask_b'] = traindata['subtask_b'].replace(label_dict)                 # replace labels with their nos
            traindata['subtask_b'] = traindata['subtask_b'].apply(str)  # string conversion
            valdata['subtask_b'] = valdata['subtask_b'].replace(label_dict)                 # replace labels with their nos
            valdata['subtask_b'] = valdata['subtask_b'].apply(str)  # string conversion
            print("Trainset distribution: \n", traindata['subtask_b'].value_counts())                           # check data distribution
            train_tags = traindata['subtask_b'].values.tolist()
            val_tags = valdata['subtask_b'].values.tolist()
            outfile = args.ofile2 + 'olidb_'
        else:
            possible_labels = traindata['subtask_c'].unique()
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index
            print(label_dict)       # for sanity check 
            traindata['subtask_c'] = traindata['subtask_c'].replace(label_dict)                 # replace labels with their nos
            traindata['subtask_c'] = traindata['subtask_c'].apply(str)  # string conversion
            valdata['subtask_c'] = valdata['subtask_c'].replace(label_dict)                 # replace labels with their nos
            valdata['subtask_c'] = valdata['subtask_c'].apply(str)  # string conversion
            print("Trainset distribution: \n", traindata['subtask_c'].value_counts())                           # check data distribution
            train_tags = traindata['subtask_c'].values.tolist()
            val_tags = valdata['subtask_c'].values.tolist()
            outfile = args.ofile2 + 'olidc_'


    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_data)*args.epochs)


    best_val_wf1 = None
    best_loss = None
    best_model = None
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(train_data, train_tags)
        val_loss, predictions, true_vals = evaluate(val_data, val_tags) # val_ids added for Hasoc submission
        val_f1, val_f1_w, val_f1_mic = f1_score_func(predictions, true_vals)
        epoch_time_elapsed = time.time() - epoch_start_time
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f} '.format(epoch, train_loss, val_loss) + f'F1: {val_f1}, weighted F1: {val_f1_w}, micro F1: {val_f1_mic}') # metric_sc['f1']))        
        with open(outfile + 't5base.txt', "a+") as f:
            s = f.write('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f} '.format(epoch, train_loss, val_loss) + f'F1: {val_f1}, weighted F1: {val_f1_w}, micro F1: {val_f1_mic}' + "\n")
        #if not best_val_wf1 or val_f1_w > best_val_wf1:
        if not best_loss or val_loss < best_loss:
            with open(args.savet, 'wb') as f:        # create file but deletes implicitly 1st if already exists
                #No need to save the models for now so that they don't use up space 
                #torch.save(model.state_dict(), f)    # save best model's learned parameters (based on lowest loss)
                best_model = model
                if args.datatype == 'hasoc' and args.datayear == '2021':     # save model for hasoc 2021 inference
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(args.msave)  # transformers save
                    tokenizer.save_pretrained(args.msave)

            #with open(args.pikle, 'wb') as file:    # save the classifier as a pickle file
                #pickle.dump(model, file)
            #best_val_wf1 = val_f1_w
            best_loss = val_loss
    
    # Hasoc 2021 & OLID test sets will be run according to Hasoc format in order to prepare, so...
    if args.datatype == 'hasoc' and args.datayear == '2020':
        model = best_model
        eval_loss, predictions, true_vals = evaluate(test_data_texts, test_data_labels) # test_ids added for Hasoc submission
        eval_f1, eval_f1_w, eval_f1_mic = f1_score_func(predictions, true_vals)
        print('Test Loss: {:.4f} '.format(eval_loss) + f'F1: {eval_f1}, weighted F1: {eval_f1_w}, micro F1: {eval_f1_mic}')       
        with open(outfile + 't5base.txt', "a+") as f:
            s = f.write('Test Loss: {:.4f} '.format(eval_loss) + f'F1: {eval_f1}, weighted F1: {eval_f1_w}, micro F1: {eval_f1_mic}' + "\n")
