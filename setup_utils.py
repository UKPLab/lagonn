import torch
import os
import re
import random
import json
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.metrics import average_precision_score, f1_score, accuracy_score, classification_report

MODEL_SEED = 0

os.environ["WANDB_DISABLED"] = "true"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def evaluation(predictions, args):
    if args.mode in ['ROBERTA_FREEZE', 'ROBERTA_FULL']:
        
        model_outputs = predictions.predictions
        targets = predictions.label_ids
        outputs = np.argmax(model_outputs, axis=-1)
        logit_outputs = 0
        if len(model_outputs.shape) == 2:
            logit_outputs = model_outputs[:, 1]
    else:
        logit_outputs, outputs, targets = predictions
 
    if len(set(targets)) > 2:
        f1 = f1_score(targets, outputs, average='macro')*100    
    else:
        f1 = f1_score(targets, outputs)*100
    
    if type(logit_outputs) == int:
        avg_pre = 0
    else:
        try:
            avg_pre = average_precision_score(targets, logit_outputs)*100
        except ValueError:
            avg_pre = 0
    
    print(classification_report(targets, outputs))
    
    print("Average precision = {}".format(avg_pre))
    print()
    print('F1 = {}'.format(f1))
    
    return avg_pre, f1

def custom_evaluation(predictions):
    
    logit_outputs, outputs, targets = predictions
 
    if len(set(targets)) > 2:
        binary = False
        f1_mac = f1_score(targets, outputs, average='macro')*100
        f1_mic = f1_score(targets, outputs, average='micro')*100    
    else:
        binary = True
        f1_bin = f1_score(targets, outputs)*100
        try:
            avg_pre = average_precision_score(targets, logit_outputs)*100
        except ValueError:
            avg_pre = 0
    acc = accuracy_score(targets, outputs)*100
    
    print(classification_report(targets, outputs))
    if len(set(targets)) > 2:
        return f1_mac, f1_mic, acc, binary
    else:
        return f1_bin, avg_pre, acc, binary
     

def get_eval_dict(args, train_predictions, test_predictions, custom=False):

    if not custom:

        print('TRAINING EVALUATION FOR {} ON SEED NUMBER = {}'.format(args.mode, args.seed))
        train_avg_pre, train_f1 = evaluation(train_predictions, args)
        
        print('TESTING EVALUATION FOR {} ON SEED NUMBER = {}'.format(args.mode,  args.seed))
        test_avg_pre, test_f1 = evaluation(test_predictions, args)
        
        eval_dict = {'train_avg_pre': train_avg_pre, 
                    'train_f1': train_f1, 
                    'test_avg_pre': test_avg_pre, 
                    'test_f1': test_f1}
    
    else:
        print('Training evalutation')
        f1_mac, f1_mic, acc, binary = custom_evaluation(train_predictions)
        if binary:
            train_bin = f1_mac
            train_ap = f1_mic
            train_acc = acc
            print('Testing evluation')
            t_f1_mac, t_f1_mic, t_acc, _ = custom_evaluation(test_predictions)
            test_bin = t_f1_mac
            test_ap = t_f1_mic
            test_acc = t_acc
            eval_dict = {'train_ap': train_ap, 
                        'train_f1_binary': train_bin,
                        'train_accuracy': train_acc, 
                        'test_ap': test_ap, 
                        'test_f1_binary': test_bin,
                        'test_accuracy': test_acc}
        else:
            train_mac = f1_mac
            train_mic = f1_mic
            train_acc = acc
            print('Testing evluation')
            t_f1_mac, t_f1_mic, t_acc, _ = custom_evaluation(test_predictions)
            eval_dict = {'train_f1_macro': train_mac,
                         'train_f1_micro': train_mic,
                         'train_accuracy': train_acc,
                         'test_f1_macro': t_f1_mac,
                         'test_f1_micro': t_f1_mic,
                         'test_accuracy': t_acc}
    return eval_dict


def write_eval_jsons(eval_dict, args, step, balance):
    folder = 'out_jsons/{}/{}/{}/{}/{}/'.format(args.task, args.mode, balance, args.seed, step)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    baselines = ['KNN', 'LOG_REG', 'PROBE', 'ROBERTA_FREEZE', 'ROBERTA_FULL', 'SETFIT', 'SETFIT_LITE']
    
    if args.mode in baselines:
        config = 'results.json'
    else:
        config = '{}!results.json'.format(args.lagonnconfig)
    
    writefile = folder + config
    #try:
    #    os.remove(writefile)
    #except OSError:
    #    pass
    with open(writefile, 'a') as f:
        f.write(json.dumps(eval_dict)+'\n')



def fix_liar(args, split):
    st_modes = ['LAGONN_CHEAP', 'LAGONN', 'LAGONN_LITE', 'LAGONN_EXP', 
               'KNN', 'LOG_REG', 'SETFIT', 'PROBE', 'SETFIT_LITE']
    
    df = pd.read_csv('dataframes_with_val/{}_need_fix_{}.csv'.format('liar', split)).dropna()
    
    if args.mode in st_modes:
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/{}'.format(args.st_model))
    elif args.mode in ['ROBERTA_FREEZE', 'ROBERTA_FULL']:
        tokenizer = AutoTokenizer.from_pretrained(args.transformer_clf)
    
    outtext = []
    for txt, contxt in zip(df.text.to_list(), df.context.to_list()):
        outtxt = '{} {} {}'.format(txt, tokenizer.sep_token, contxt).strip()
        outtxt = re.sub(' +', ' ', outtxt)
        outtext.append(outtxt)
    
    outdf = df[['label_text', 'labels']].copy(deep=True)
    outdf['text'] = pd.Series(outtext).values

    return outdf  


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def predict_with_sklearn(X, y, clf):
    y_logit = clf.predict_proba(X)
    if len(y_logit.shape) == 2:
        y_logit = y_logit[:, 1]
    y_pred = clf.predict(X)
    predictions = (y_logit, y_pred, y)
    
    return predictions

def sample_df_convert_ds(train_df, balance, step, args):
    
    bi_ratio_dict = {'extreme': (0.98, 0.02), 'imbalanced': (0.9, 0.1), 'moderate': (0.75, 0.25)}
    tri_ratio_dict = {'extreme': (0.02, 0.95, 0.03), 'imbalanced': (0.05, 0.8, 0.15), 'moderate': (0.10, 0.65, 0.25)}
    initial = 100

    num_labs = len(set(train_df['labels']))
    
    sample_size = initial*step
    if balance not in ['balanced']:
        sample = pd.DataFrame()
        if args.task in ['hate_speech_offensive']:
            ratios = tri_ratio_dict[balance]
        else:    
            ratios = bi_ratio_dict[balance]
        for idx, ratio in enumerate(ratios):
            samp_size_ratio = sample_size*ratio
            try:
                lab_sample = train_df[train_df['labels']==idx].sample(n=int(samp_size_ratio), random_state=args.seed)
            except ValueError: #sample with replacement when there are no other samples
                lab_sample = train_df[train_df['labels']==idx].sample(n=int(samp_size_ratio), replace=True, random_state=args.seed)
            sample = pd.concat([sample, lab_sample], ignore_index=True)
    
    elif balance in ['balanced']:
        try:
            sample = train_df.groupby('labels').apply(lambda x: x.sample(n=int(sample_size/num_labs), random_state=args.seed))
        except ValueError: #sample with replacement when there are no other samples
            sample = train_df.groupby('labels').apply(lambda x: x.sample(n=int(sample_size/num_labs), replace=True, random_state=args.seed))
    return Dataset.from_pandas(sample).shuffle(seed=args.seed)