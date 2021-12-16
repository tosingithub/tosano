import torch
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
#import seaborn as sns
#import transformers
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset,  RandomSampler, SequentialSampler
#from transformers import RobertaModel, RobertaTokenizer
import logging
import argparse
import pickle
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import utility_hs as util
logging.basicConfig(level=logging.ERROR)
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup


parser = argparse.ArgumentParser(description='Hate Speech Model')
parser.add_argument('--savet', type=str, default='modelrobertabase_hasoc_task1a.pt', help='filename of the model checkpoint')
parser.add_argument('--pikle', type=str, default='modelrobertabase_hasoc_task1a.pkl', help='pickle filename of the model checkpoint')
parser.add_argument('--ofile1', type=str, default='outputfile_task1a.txt', help='output file')
parser.add_argument('--submission1', type=str, default='submitfile_task1a.csv', help='submission file')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=2, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--maxlen', type=int, default=256, help='maximum length')
args = parser.parse_args()


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average=None), f1_score(labels_flat, preds_flat, average="weighted")

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
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
            
            #progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
        #print("Saving the model.... ")
        #torch.save(model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')
        #tqdm.write(f'\nEpoch {epoch}')
    loss_train_avg = loss_train_total/len(dataloader_train)
        #tqdm.write(f'Training loss: {loss_train_avg}')
    return loss_train_avg


def evaluate(dataloader_val):
    print("Evaluation...")
    model.eval()
        
    loss_val_total = 0
    predictions, true_vals = [], []
        
    for batch in dataloader_val:
            
        batch = tuple(b.to(device) for b in batch)
            
        inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }

        with torch.no_grad():        
                outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
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

    df, test_data = util.read_data()
    #print(df.describe())
    #print(df['task_1'].value_counts())

    # Pre-processing ?
    possible_labels = df.task_1.unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    print(label_dict)               # NOT: 0; HOF: 1
    df['task_1'] = df.task_1.replace(label_dict)                 # replace labels with their nos
    test_data['task_1'] = test_data.task_1.replace(label_dict)                 # replace labels with their nos
    traindata, valdata = train_test_split(df, test_size=0.2, shuffle=True)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
    model = RobertaForSequenceClassification.from_pretrained('roberta-base').to(device)

    encoded_data_train = tokenizer.batch_encode_plus(
        traindata.text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        truncation=True,
        return_tensors='pt' # padding=True, truncation=True
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        valdata.text.values, 
        add_special_tokens=True, 
        return_attention_mask=True,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    encoded_data_test = tokenizer.batch_encode_plus(
        test_data.text.values, 
        add_special_tokens=True, 
        return_attention_mask=True,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids'].to(device)
    attention_masks_train = encoded_data_train['attention_mask'].to(device)
    labels_train = torch.tensor(traindata.task_1.values)
    labels_train = labels_train.to(device)

    input_ids_val = encoded_data_val['input_ids'].to(device)
    attention_masks_val = encoded_data_val['attention_mask'].to(device)
    labels_val = torch.tensor(valdata.task_1.values)
    labels_val = labels_val.to(device)

    input_ids_test = encoded_data_test['input_ids'].to(device)
    attention_masks_test = encoded_data_test['attention_mask'].to(device)
    labels_test = torch.tensor(test_data.task_1.values)
    labels_test = labels_test.to(device)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

    dataloader_train = DataLoader(dataset_train, 
                                sampler=RandomSampler(dataset_train), 
                                batch_size=args.batch_size)

    dataloader_validation = DataLoader(dataset_val, 
                                    sampler=SequentialSampler(dataset_val), 
                                    batch_size=args.batch_size)

    dataloader_test = DataLoader(dataset_test, 
                                    sampler=SequentialSampler(dataset_test), 
                                    batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*args.epochs)

    best_val_wf1 = None
    best_model = None
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(train_data, train_tags)
        val_loss, predictions, true_vals = evaluate(val_data, val_tags) # val_ids added for Hasoc submission
        val_f1, val_f1_w, val_f1_mic = f1_score_func(predictions, true_vals)
        epoch_time_elapsed = time.time() - epoch_start_time
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f} '.format(epoch, train_loss, val_loss) + f'F1: {val_f1}, weighted F1: {val_f1_w}, micro F1: {val_f1_mic}') # metric_sc['f1']))        
        with open(args.ofile1, "a+") as f:
            s = f.write('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f} '.format(epoch, train_loss, val_loss) + f'F1: {val_f1}, weighted F1: {val_f1_w}, micro F1: {val_f1_mic}' + "\n")
        if not best_val_wf1 or val_f1_w > best_val_wf1:
            with open(args.savet, 'wb') as f:        # create file but deletes implicitly 1st if already exists
                torch.save(model.state_dict(), f)    # save best model's learned parameters (based on lowest loss)
                best_model = model
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained('t5basemodel_save/')  # transformers save
                tokenizer.save_pretrained('t5basemodel_save/')

            with open(args.pikle, 'wb') as file:    # save the classifier as a pickle file
                pickle.dump(model, file)
            best_val_wf1 = val_f1_w

    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    # outputs = model(**inputs, labels=labels)
    # loss, logits = outputs[:2]
    # print("LOSS: ",loss)
