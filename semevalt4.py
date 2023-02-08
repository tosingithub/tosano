from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, get_linear_schedule_with_warmup, get_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from datasets import load_dataset, load_metric, list_metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef, accuracy_score

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


parser = argparse.ArgumentParser(description='PCL Model')
# --datatype: hasoc, trol, sema, semb, hos, olid (--olidtask = a, b or c)
parser.add_argument('--datatype', type=str, default='pcl', help='data of choice')
parser.add_argument('--pcl_train', type=str, default='semeval/task4_train20.csv', help='location of the training data')
parser.add_argument('--pcl_dev', type=str, default='semeval/task4_dev20.csv', help='location of the dev data')
#parser.add_argument('--has_devdata', type=str, default='/home/shared_data/h/has21_devdata.csv', help='location of the dev data')
# Trolling & Aggression
parser.add_argument('--task_pref', type=str, default="classification: ", help='Task prefix')
parser.add_argument('--datayear', type=str, default="2021", help='Data year')           # 2020 or 2021
parser.add_argument('--taskno', type=str, default="1", help='Task Number')              # 1 or 2
parser.add_argument('--savet', type=str, default='t5base.pt', help='filename of the model checkpoint')
parser.add_argument('--pikle', type=str, default='t5base.pkl', help='pickle filename of the model checkpoint')
parser.add_argument('--msave', type=str, default='t5base_nodetest', help='folder to save the finetuned model')
parser.add_argument('--ofile1', type=str, default='outputfile_', help='output file')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=1, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size') # smaller batch size for big model to fit GPU
args = parser.parse_args()


def f1_score_func(preds, labels):
    preds_flat = []
    preds_flat = ['0' if a == '' or len(a) > 1 else a for a in preds]   # TODO: USE LARGEST LABEL; get rid of empty & lengthy predictions
    labels_flat = labels            # only for consistency
    return f1_score(labels_flat, preds_flat, average=None), f1_score(labels_flat, preds_flat, average="weighted"), f1_score(labels_flat, preds_flat, average="macro")

# def accuracy_score_func(preds, labels):
#     preds_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()
#     return accuracy_score(labels_flat, preds_flat, normalize='False')

def confusion_matrix_func(preds, labels):
    #if args.datatype == 'hasoc':
    preds_flat = []
    preds_flat_ = ['0' if a == '' or len(a) > 1 else a for a in preds]   # get rid of empty & lengthy predictions
    preds_flat.extend(preds_flat_)
    labels_flat = labels
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

def get_data(datatype, augment_traindata=False):
    """ Select the dataset to use """
    if datatype == 'pcl':
        if not augment_traindata:
            print('Using PCL data... ')
            traindata = pd.read_csv(args.pcl_train)
            print(len(traindata))
            traindata, devdata = train_test_split(traindata, test_size=0.10, shuffle=True)
        else:
            print('Using PCL augmented data... ')
            data21a = pd.read_csv(args.has21_traindata)
            data21b = pd.read_csv(args.has21_aug_gen)
            traindata = pd.concat([data21a, data21b])
            traindata, devdata = train_test_split(traindata, test_size=0.10, shuffle=True) #, random_state=42, stratify=traindata['task_1'].values)
    
    return traindata, devdata

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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    tokenizer.pad_token = tokenizer.eos_token # to avoid an error
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) #, betas=(0.7, 0.99))
    
    #traindata, devdata = get_data(args.datatype, augment_traindata=False)
    traindata = pd.read_csv(args.pcl_train)
    devdata = pd.read_csv(args.pcl_dev)

    print(traindata['label'].value_counts())
    print(devdata['label'].value_counts())
    #print(traindata.head())

    # Comment out the below if preprocessing not needed
    traindata = util.preprocess_pandas(traindata, list(traindata.columns))
    valdata = util.preprocess_pandas(devdata, list(devdata.columns))

    ### Text column
    # Add task prefix for T5 better performance
    if args.datatype == 'pcl':
        traindata['text'] = args.task_pref + traindata['text']
        valdata['text'] = args.task_pref + valdata['text']
        traindata['text'] = traindata['text'].apply(str)                # force string conversion for error after augmenting
        train_data = traindata['text'].values.tolist()
        val_data = valdata['text'].values.tolist()

    ### Label column
    outfile = ''
    label_dict = {}         # For associating raw labels with indices/nos
    if args.datatype == 'pcl':
        possible_labels = traindata['label'].unique()
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        print(label_dict)       # for sanity check 
        traindata['label'] = traindata['label'].replace(label_dict)                 # replace labels with their nos
        traindata['label'] = traindata['label'].apply(str)  # string conversion
        valdata['label'] = valdata['label'].replace(label_dict)                 # replace labels with their nos
        valdata['label'] = valdata['label'].apply(str)  # string conversion
        print("Trainset distribution: \n", traindata['label'].value_counts())                           # check data distribution
        train_tags = traindata['label'].values.tolist()
        val_tags = valdata['label'].values.tolist()
        outfile = args.ofile1 + 'pcl_'

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
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f} '.format(epoch, train_loss, val_loss) + f'F1: {val_f1}, weighted F1: {val_f1_w}, macro F1: {val_f1_mic}') # metric_sc['f1']))        
        with open(outfile + 't5base.txt', "a+") as f:
            s = f.write('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f} '.format(epoch, train_loss, val_loss) + f'F1: {val_f1}, weighted F1: {val_f1_w}, macro F1: {val_f1_mic}' + "\n")
        #if not best_val_wf1 or val_f1_w > best_val_wf1:
        if not best_loss or val_loss < best_loss:
            with open(args.savet, 'wb') as f:        # create file but deletes implicitly 1st if already exists
                #No need to save the models for now so that they don't use up space 
                #torch.save(model.state_dict(), f)    # save best model's learned parameters (based on lowest loss)
                best_model = model
                if args.datatype == 'pcl':     # save model
                    args.msave = args.msave + '_pcl'
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(args.msave)  # transformers save
                    tokenizer.save_pretrained(args.msave)

            #with open(args.pikle, 'wb') as file:    # save the classifier as a pickle file
                #pickle.dump(model, file)
            #best_val_wf1 = val_f1_w
            best_loss = val_loss
    
