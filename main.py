import argparse
import time
import pandas as pd

from datasets import Dataset

from use_setfit import do_setfit
from setup_utils import (str2bool, 
                         sample_df_convert_ds, 
                         fix_liar,
                         write_eval_jsons)
from modelling import (train_once, 
                       train_head_every_step, 
                       train_every_step, 
                       setfit_once, 
                       setfit_every_step, 
                       linear_probe)

from lagonn import LaGoNN


parser = argparse.ArgumentParser(description='configs LaGoNN')
parser.add_argument('--WRITE', action="store", type=str2bool, dest='write', default='True')
parser.add_argument('--TRANSFORMER_CLF', action="store", dest='transformer_clf', default='roberta-base')
parser.add_argument('--ST_MODEL', action="store", dest='st_model', default="sentence-transformers/paraphrase-mpnet-base-v2")
parser.add_argument('--SEED', action="store", type=int, dest='seed', default=0)
parser.add_argument('--TASK', action="store", dest='task', default='insincere-questions')
parser.add_argument('--MODE', action="store", dest='mode', default='LAGONN_CHEAP')
parser.add_argument('--LAGONN_CONFIG', action="store", dest='lagonnconfig', default='LABEL')




if __name__ == '__main__':
    args = parser.parse_args()
    balances = ['extreme', 'imbalanced', 'moderate', 'balanced']
    
    assert args.mode in ['LAGONN_CHEAP', 'LAGONN', 'LAGONN_LITE', 'LAGONN_EXP', 'PROBE', 'KNN', 'LOG_REG',
                        'SETFIT', 'SETFIT_LITE', 'ROBERTA_FREEZE', 'ROBERTA_FULL']
    
    if args.task in ['liar']:
        input_test_ds = Dataset.from_pandas(fix_liar(args, 'test'))
        input_val_ds = Dataset.from_pandas(fix_liar(args, 'val'))
        train_df = fix_liar(args, 'train')
    
    else:
        input_test_ds = Dataset.from_pandas(pd.read_csv('dataframes_with_val/{}_test.csv'.format(args.task)).dropna()[['text', 'labels', 'label_text']])
        input_val_ds = Dataset.from_pandas(pd.read_csv('dataframes_with_val/{}_val.csv'.format(args.task)).dropna()[['text', 'labels', 'label_text']])
        train_df = pd.read_csv('dataframes_with_val/{}_train.csv'.format(args.task)).dropna()[['text', 'labels', 'label_text']]
    
    for balance in balances:            
                
        for step in range(1, 11):                
            input_train_ds = sample_df_convert_ds(train_df, balance, step, args)
            if args.mode in ['PROBE']:
                eval_dict = linear_probe(input_train_ds, input_test_ds, args)
            
            elif args.mode in ['KNN', 'LOG_REG']:
                if step == 1:
                    sbert_trainer = do_setfit(args, input_train_ds, input_val_ds)
                eval_dict = setfit_once(input_train_ds, input_test_ds, sbert_trainer, step, args)

            elif args.mode in ['SETFIT']: 
                eval_dict = setfit_every_step(input_train_ds, input_val_ds, input_test_ds, args)            

            elif args.mode in ['SETFIT_LITE']:
                if step < 5:
                    eval_dict, sbert_trainer = setfit_every_step(input_train_ds, input_val_ds, input_test_ds, args)
                else:
                    eval_dict = setfit_once(input_train_ds, input_test_ds, sbert_trainer, step, args)
            
            elif args.mode in ['ROBERTA_FREEZE']:                
                if step == 1:
                    transformer_trainer, eval_dict = train_once(input_train_ds, input_val_ds, input_test_ds, args, balance)
                
                elif step > 1:
                    eval_dict = train_head_every_step(input_train_ds, input_val_ds, input_test_ds, args, transformer_trainer, balance)
                        
            elif args.mode in ['ROBERTA_FULL']:
                eval_dict = train_every_step(input_train_ds, input_val_ds, input_test_ds, args, balance)          
            
            elif args.mode in ['LAGONN_CHEAP', 'LAGONN_EXP']:
                lgn = LaGoNN(input_train_ds, input_val_ds, input_test_ds, args, step)
                eval_dict = lgn.predict()
            
            elif args.mode in ['LAGONN']:
                if step == 1:
                    lgn = LaGoNN(input_train_ds, input_val_ds, input_test_ds, args, step)
                    eval_dict, sbert_trainer = lgn.predict()
                else:
                    lgn = LaGoNN(input_train_ds, input_val_ds, input_test_ds, args, step, sbert_trainer)
                    eval_dict = lgn.predict()
            
            elif args.mode in ['LAGONN_LITE']:
                if step < 5:
                    lgn = LaGoNN(input_train_ds, input_val_ds, input_test_ds, args, step)
                    eval_dict, sbert_trainer = lgn.predict()
                else:
                    lgn = LaGoNN(input_train_ds, input_val_ds, input_test_ds, args, sbert_trainer)
                    eval_dict = lgn.predict()

            train_avg_pre = eval_dict['train_avg_pre']
            train_f1 = eval_dict['train_f1']
            test_avg_pre = eval_dict['test_avg_pre']
            test_f1 = eval_dict['test_f1']
            
            print('TRAINING RESULTS')
            print('AP: {}'.format(train_avg_pre))
            print('F1: {}'.format(train_f1))
            print()
            print('TESTING RESULTS')
            print('AP: {}'.format(test_avg_pre))
            print('F1: {}'.format(test_f1))
            print()
            print('writing')

            if args.write:
                write_eval_jsons(eval_dict, args, step, balance)

            print('Seed {} done for step {} of {} done! \n'.format(args.seed, step, balance+' '+args.task)) 
        
print("Job's done!")