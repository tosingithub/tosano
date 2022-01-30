import torch
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
#import seaborn as sns
#import transformers
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset,  RandomSampler, SequentialSampler
import logging
import argparse
import pickle
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import utility_hs as util
logging.basicConfig(level=logging.ERROR)
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup


parser = argparse.ArgumentParser(description='Hate Speech Model')
# --datatype: hasoc, trol, sema, semb, hos, olid (--olidtask = a, b or c)
# USAGE EXAMPLE in the terminal: python ht5.py --datatype trol
parser.add_argument('--datatype', type=str, default='hasoc', help='data of choice')
parser.add_argument('--olidtask', type=str, default="b", help='Task Alphabet')      # a, b or c

parser.add_argument('--datayear', type=str, default="2021", help='Data year')
parser.add_argument('--taskno', type=str, default="1", help='Task Number')
parser.add_argument('--savet', type=str, default='robase.pt', help='filename of the model checkpoint')
parser.add_argument('--pikle', type=str, default='robase.pkl', help='pickle filename of the model checkpoint')
parser.add_argument('--msave', type=str, default='robase', help='folder to save the finetuned model')
parser.add_argument('--ofile1', type=str, default='outputfile_', help='output file')
parser.add_argument('--ofile2', type=str, default='outputfile_', help='output file')
parser.add_argument('--submission1', type=str, default='rosubmitfile_task1a.csv', help='submission file')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate')
# bestloss at 0.0002; F1: [0.82906977 0.65083135], weighted F1: 0.7618651197202165, micro F1: 0.7704918032786885

parser.add_argument('--epochs', type=int, default=3, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size') # smaller batch size for big model to fit GPU
parser.add_argument('--maxlen', type=int, default=256, help='maximum length')
args = parser.parse_args()


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average=None), f1_score(labels_flat, preds_flat, average="weighted"), f1_score(labels_flat, preds_flat, average="macro")

def accuracy_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, preds_flat, normalize='False')

def confusion_matrix_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    print(confusion_matrix(labels_flat, preds_flat))


def train(dataloader_train):
    print("Training...")
    loss_train_total = 0

    for batch in tqdm(dataloader_train): #, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False):
        optimizer.zero_grad()
        #model.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2],}       
        outputs = model(**inputs)
        loss = outputs[0] # loss_fn(outputs, batch[2]) #
        #print("Loss ", loss)
        #print("Loss item ", loss.item())
        loss_train_total += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
            
    loss_train_avg = loss_train_total/len(dataloader_train)
    return loss_train_avg


def evaluate(dataloader_val):
    print("Evaluation...")
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
        
    for batch in tqdm(dataloader_val):
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2],}

        with torch.no_grad():        
                outputs = model(**inputs)
        loss, logits = outputs[:2]
        #logits = outputs[1]
        loss_val_total += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_val_avg, predictions, true_vals


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loss_fn = torch.nn.CrossEntropyLoss()

    traindata, devdata, testdata = util.get_data(args.datatype, args.datayear, augment_traindata=False)
    # Comment out the below if preprocessing not needed
    traindata = util.preprocess_pandas(traindata, list(traindata.columns))
    valdata = util.preprocess_pandas(devdata, list(devdata.columns))
    if args.datatype == 'hasoc': # not args.datatype == 'trol' or not args.datatype == 'sema' or not args.datatype == 'semb':                                         # skip line below for trol, 
        test_data = util.preprocess_pandas(testdata, list(testdata.columns))

    outfile = ''
    label_dict = {}         # For associating raw labels with indices/nos
    if args.datatype == 'hasoc' and args.taskno == '1':
        possible_labels = traindata.task_1.unique()
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        print(label_dict)       # NOT: 0; HOF: 1
        traindata['task_1'] = traindata.task_1.replace(label_dict)                 # replace labels with their nos
        valdata['task_1'] = valdata.task_1.replace(label_dict)                 # replace labels with their nos
        print("Trainset distribution: \n", traindata['task_1'].value_counts())                           # check data distribution
        if args.datatype == 'hasoc' and not args.datayear == '2021':            # we'll do 2021 inference on testset elsewhere
            test_data['task_1'] = test_data.task_1.replace(label_dict)                 # replace labels with their nos
        outfile = args.ofile1 + 'task1_robert_'
    elif args.datatype == 'hasoc' and args.taskno == '2':
        possible_labels = traindata.task_2.unique()
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        print(label_dict)       # for sanity check {'NONE': 0, 'PRFN': 1, 'OFFN': 2, 'HATE': 3}
        traindata['task_2'] = traindata.task_2.replace(label_dict)                 # replace labels with their nos
        valdata['task_2'] = valdata.task_2.replace(label_dict)                 # replace labels with their nos
        print("Trainset distribution: \n", traindata['task_2'].value_counts())                           # check data distribution
        if args.datatype == 'hasoc' and not args.datayear == '2021':            # we'll do 2021 inference on testset elsewhere
            test_data['task_2'] = test_data.task_2.replace(label_dict)                 # replace labels with their nos
        outfile = args.ofile2 + 'task2_robert_'
    elif args.datatype == 'trol':
        possible_labels = traindata['Sub-task A'].unique()
        for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index
        print(label_dict)       # for sanity check {'NAG': 0, 'CAG': 1, 'OAG': 2}
        traindata['Sub-task A'] = traindata['Sub-task A'].replace(label_dict)                 # replace labels with their nos
        valdata['Sub-task A'] = valdata['Sub-task A'].replace(label_dict)                 # replace labels with their nos
        print("Trainset distribution: \n", traindata['Sub-task A'].value_counts())                           # check data distribution
        outfile = args.ofile2 + 'trol_robert_'
    elif args.datatype == 'hos':
        print("Trainset distribution: \n", traindata['hate_speech'].value_counts())                           # check data distribution
        outfile = args.ofile2 + 'hos_robert_'
    elif args.datatype == 'sema' or args.datatype == 'semb':
        print("Trainset distribution: \n", traindata['HS'].value_counts())                           # check data distribution
        outfile = args.ofile2 + 'sem_robert_'
    elif args.datatype == 'olid':
        if args.olidtask == 'a':
            possible_labels = traindata['subtask_a'].unique()
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index
            print(label_dict)       # for sanity check 
            traindata['subtask_a'] = traindata['subtask_a'].replace(label_dict)                 # replace labels with their nos
            valdata['subtask_a'] = valdata['subtask_a'].replace(label_dict)                 # replace labels with their nos
            print("Trainset distribution: \n", traindata['subtask_a'].value_counts())                           # check data distribution
            outfile = args.ofile2 + 'olida_robert_'
        elif args.olidtask == 'b':
            possible_labels = traindata['subtask_b'].unique()
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index
            print(label_dict)       # for sanity check 
            traindata['subtask_b'] = traindata['subtask_b'].replace(label_dict)                 # replace labels with their nos
            valdata['subtask_b'] = valdata['subtask_b'].replace(label_dict)                 # replace labels with their nos
            print("Trainset distribution: \n", traindata['subtask_b'].value_counts())                           # check data distribution
            outfile = args.ofile2 + 'olidb_robert_'
        else:
            possible_labels = traindata['subtask_c'].unique()
            for index, possible_label in enumerate(possible_labels):
                label_dict[possible_label] = index
            print(label_dict)       # for sanity check 
            traindata['subtask_c'] = traindata['subtask_c'].replace(label_dict)                 # replace labels with their nos
            valdata['subtask_c'] = valdata['subtask_c'].replace(label_dict)                 # replace labels with their nos
            print("Trainset distribution: \n", traindata['subtask_c'].value_counts())                           # check data distribution
            outfile = args.ofile2 + 'olidc_robert_'


    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
    model = RobertaForSequenceClassification.from_pretrained('roberta-base').to(device)

    if args.datatype == 'hasoc':
        encoded_data_train = tokenizer.batch_encode_plus(traindata.text.values, add_special_tokens=True, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
        encoded_data_val = tokenizer.batch_encode_plus(valdata.text.values, add_special_tokens=True, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
        if args.datayear == '2020':
            encoded_data_test = tokenizer.batch_encode_plus(test_data.text.values, add_special_tokens=True, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
            input_ids_test = encoded_data_test['input_ids'].to(device)
            attention_masks_test = encoded_data_test['attention_mask'].to(device)
            labels_test = torch.tensor(test_data.task_1.values)
            labels_test = labels_test.to(device)
            dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
            dataloader_test = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=args.batch_size)
            if args.taskno == '1':
                labels_train = torch.tensor(traindata.task_1.values)
                labels_val = torch.tensor(valdata.task_1.values)
            else:
                labels_train = torch.tensor(traindata.task_2.values, dtype=torch.float)
                labels_val = torch.tensor(valdata.task_2.values, dtype=torch.float)
        else:
            if args.taskno == '1':
                labels_train = torch.tensor(traindata.task_1.values)
                labels_val = torch.tensor(valdata.task_1.values)
            else:
                labels_train = torch.tensor(traindata.task_2.values, dtype=torch.float)
                labels_val = torch.tensor(valdata.task_2.values, dtype=torch.float)
    elif args.datatype == 'trol':
        encoded_data_train = tokenizer.batch_encode_plus(traindata.Text.values, add_special_tokens=True, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
        encoded_data_val = tokenizer.batch_encode_plus(valdata.Text.values, add_special_tokens=True, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
        labels_train = torch.tensor(traindata['Sub-task A'].values)
        labels_val = torch.tensor(valdata['Sub-task A'].values)
    elif args.datatype == 'hos':
        encoded_data_train = tokenizer.batch_encode_plus(traindata.tweet.values, add_special_tokens=True, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
        encoded_data_val = tokenizer.batch_encode_plus(valdata.tweet.values, add_special_tokens=True, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
        labels_train = torch.tensor(traindata['hate_speech'].values)
        labels_val = torch.tensor(valdata['hate_speech'].values)
    elif args.datatype == 'sema' or args.datatype == 'semb':
        encoded_data_train = tokenizer.batch_encode_plus(traindata.text.values, add_special_tokens=True, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
        encoded_data_val = tokenizer.batch_encode_plus(valdata.text.values, add_special_tokens=True, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
        labels_train = torch.tensor(traindata['HS'].values)
        labels_val = torch.tensor(valdata['HS'].values)
    elif args.datatype == 'olid':
        encoded_data_train = tokenizer.batch_encode_plus(traindata.tweet.values, add_special_tokens=True, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
        encoded_data_val = tokenizer.batch_encode_plus(valdata.tweet.values, add_special_tokens=True, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
        if args.olidtask == 'a':
            labels_train = torch.tensor(traindata['subtask_a'].values)
            labels_val = torch.tensor(valdata['subtask_a'].values)
        elif args.olidtask == 'b':
            labels_train = torch.tensor(traindata['subtask_b'].values)
            labels_val = torch.tensor(valdata['subtask_b'].values)
        else:
            labels_train = torch.tensor(traindata['subtask_c'].values, dtype=torch.float)
            labels_val = torch.tensor(valdata['subtask_c'].values, dtype=torch.float)
    
    input_ids_train = encoded_data_train['input_ids'].to(device)
    attention_masks_train = encoded_data_train['attention_mask'].to(device)
    input_ids_val = encoded_data_val['input_ids'].to(device)
    attention_masks_val = encoded_data_val['attention_mask'].to(device)
    labels_train = labels_train.to(device)
    labels_val = labels_val.to(device)


    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=args.batch_size)
    dataloader_validation = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*args.epochs)

    best_val_wf1 = None
    best_loss = None
    best_model = None
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(dataloader_train)
        val_loss, predictions, true_vals = evaluate(dataloader_validation)
        val_f1, val_f1_w, val_f1_mac = f1_score_func(predictions, true_vals)
        epoch_time_elapsed = time.time() - epoch_start_time
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f} '.format(epoch, train_loss, val_loss) + f'F1: {val_f1}, weighted F1: {val_f1_w}, macro F1: {val_f1_mac}')       
        with open(args.ofile1 + 'roberta.txt', "a+") as f:
            s = f.write('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f} '.format(epoch, train_loss, val_loss) + f'F1: {val_f1}, weighted F1: {val_f1_w}, macro F1: {val_f1_mac}' + "\n")
        #if not best_val_wf1 or val_f1_w > best_val_wf1:
        if not best_loss or val_loss < best_loss:
            with open(args.savet, 'wb') as f:           # We do not need to save models for now
                #torch.save(model.state_dict(), f)    # save best model's learned parameters (based on lowest loss)
                best_model = model
                if args.datatype == 'hasoc' and args.datayear == '2021':     # save model for hasoc 2021 inference
                    if args.taskno == '2': args.msave = args.msave + '_t2'
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(args.msave)  # transformers save
                    tokenizer.save_pretrained(args.msave)
                elif args.datatype == 'olid':     # save model for inference
                    if args.olidtask == 'a': args.msave = args.msave + '_olida'
                    elif args.olidtask == 'b': args.msave = args.msave + '_olidb'
                    else: args.msave = args.msave + '_olidc'
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(args.msave)  # transformers save
                    tokenizer.save_pretrained(args.msave)

            #with open(args.pikle, 'wb') as file:    # save the classifier as a pickle file
                #pickle.dump(model, file)
            #best_val_wf1 = val_f1_w
            best_loss = val_loss
    
    # Evaluate test set
    if args.datatype == 'hasoc' and args.datayear == '2020':
        model = best_model
        eval_loss, predictions, true_vals = evaluate(dataloader_test) # test_ids added for Hasoc submission
        eval_f1, eval_f1_w, eval_f1_mac = f1_score_func(predictions, true_vals)
        print('Test Loss: {:.4f} '.format(eval_loss) + f'F1: {eval_f1}, weighted F1: {eval_f1_w}, micro F1: {val_f1_mac}')       
        with open(args.ofile1 + 'roberta.txt', "a+") as f:
            s = f.write('Test Loss: {:.4f} '.format(eval_loss) + f'F1: {eval_f1}, weighted F1: {eval_f1_w}, micro F1: {val_f1_mac}' + "\n")
