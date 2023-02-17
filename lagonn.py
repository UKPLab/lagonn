import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from datasets import Dataset
from use_setfit import do_setfit, predict_with_setfit
from setup_utils import seed_everything, MODEL_SEED, get_eval_dict, predict_with_sklearn


class LaGoNN(object):
    def __init__(self, train_ds, val_ds, test_ds, args, step, sbert_trainer=None):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.mod_train = None
        self.mod_val = None
        self.mod_test = None
        self.knn = None
        self.label_map = dict()
        self.label_dict = dict()
        self.nn_dict = dict()
        self.args = args
        self.step = step
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/{}'.format(self.args.st_model), padding=True)
        self.sbert_trainer = sbert_trainer
    
    def get_mod_train(self, sbert):
        X_train = self.train_ds['text']
        y_train = self.train_ds['labels']
        y_labels = self.train_ds['label_text']
    
        X_train_embeddings = sbert.encode(X_train)
    
        knn = NearestNeighbors(n_neighbors=2)
        knn.fit(X_train_embeddings)
        dists, indices = knn.kneighbors(X_train_embeddings, return_distance=True)
        dists = dists[:, 1]
        indices = indices[:, 1]
    
        out_txt = []
        for text, idx, dist in zip(X_train, indices, dists):
            add_txt = X_train[idx]
            pred_lab_txt = y_labels[idx]

            lab_txt = '{}{} {}{}'.format(self.tokenizer.cls_token[0], pred_lab_txt, dist, self.tokenizer.cls_token[-1])
            
            if self.args.lagonnconfig in ['LABEL']:
                out_str = '{} {} {}'.format(text, self.tokenizer.sep_token, lab_txt).strip()
        
            elif self.args.lagonnconfig in ['TEXT']:
                out_str = '{} {} {} {}'.format(text, self.tokenizer.sep_token, lab_txt, add_txt).strip()
            
            out_txt.append(out_str)
        
        train_df = pd.DataFrame({'text':out_txt, 'labels':y_train, 'label_text': y_labels})
        self.mod_train = Dataset.from_pandas(train_df)
        self.knn = knn
    
    def get_mod_test(self, sbert, val):

        X_train = self.train_ds['text'] 
        y_labels = self.train_ds['label_text']
        
        if val:
            X_test = self.val_ds['text']
            y_test = self.val_ds['labels']
        else:    
            X_test = self.test_ds['text']
            y_test = self.test_ds['labels']
    
        X_test_embeddings = sbert.encode(X_test)

        dists, indices = self.knn.kneighbors(X_test_embeddings, return_distance=True)
        dists = dists[:, 0]
        indices = indices[:, 0]

        out_txt = []
        for text, idx, dist in zip(X_test, indices, dists):
            add_txt = X_train[idx]
            pred_lab_txt = y_labels[idx]
            lab_txt = '{}{} {}{}'.format(self.tokenizer.cls_token[0], pred_lab_txt, dist, self.tokenizer.cls_token[-1])
        
            if self.args.lagonnconfig in ['LABEL']:
                out_str = '{} {} {}'.format(text, self.tokenizer.sep_token, lab_txt).strip()
       
            elif self.args.lagonnconfig in ['TEXT']:
                out_str = '{} {} {} {}'.format(text, self.tokenizer.sep_token, lab_txt, add_txt).strip()
        
            out_txt.append(out_str)
        
        if val:
            val_df = pd.DataFrame({'text':out_txt, 'labels':y_test, 'label_text': self.val_ds['label_text']})
            self.mod_val = Dataset.from_pandas(val_df)
        
        else:
            test_df = pd.DataFrame({'text':out_txt, 'labels':y_test, 'label_text': self.test_ds['label_text']})
            self.mod_test = Dataset.from_pandas(test_df)
        
    def get_label_map(self):
        for lab, label_text in zip(self.train_ds['labels'], self.train_ds['label_text']):
            if lab in self.label_map:
                continue
            else:
                self.label_map[lab] = label_text
    
    def get_label_dict(self):
        for idx, (add_txt, lab, lab_txt) in enumerate(zip(self.train_ds['text'], self.train_ds['labels'], self.train_ds['label_text'])):
            if lab not in self.label_dict:
                self.label_dict[lab] = [(add_txt, lab, lab_txt, idx)]
            else:
                self.label_dict[lab].append((add_txt, lab, lab_txt, idx))
        
    def get_nn_dict(self, sbert):
        self.get_label_dict()
        for d_lab, tups in self.label_dict.items():
            txt_lst, label_lst = [], []
            for tup in tups:
                txt = tup[0]
                txt_lst.append(txt)
                lab_txt = tup[2]
                label_lst.append(lab_txt)
            X_lab_embeddings = sbert.encode(txt_lst)
            knn = NearestNeighbors(n_neighbors=2)
            knn.fit(X_lab_embeddings)
            self.nn_dict[d_lab] = (knn, txt_lst, label_lst)
    
    def get_text_idx_dict(self, sbert, X_embeddings, X, train=True):
        if len(self.nn_dict) == 0:
            self.get_nn_dict(sbert)
        text_idx_dict = dict()
        for lab, tup in self.nn_dict.items():
            knn, txt_lst, label_lst = tup
            dists, indices = knn.kneighbors(X_embeddings, return_distance=True)
            if train:
                indices = indices[:, 1]
                dists = dists[:, 1]
            else:
                indices = indices[:, 0]
                dists = dists[:, 0]
            for idx, (text, nn_idx, dist) in enumerate(zip(X, indices, dists)):
                keep_text = txt_lst[nn_idx]
                compare_tup = (keep_text, dist, lab)
                if idx not in text_idx_dict:
                    text_idx_dict[idx] = [compare_tup]
                else:
                    text_idx_dict[idx].append(compare_tup)
        return text_idx_dict
    
    def get_out_txt(self, text_idx_dict, split):
        if len(self.label_map) == 0:
            self.get_label_map()
        out_txt = []
        if split == 'train':
            X = self.train_ds['text']
        elif split == 'val':
            X = self.val_ds['text']
        elif split == 'test':
            X = self.test_ds['text']
        for idx, text in enumerate(X):
            lst = text_idx_dict[idx]
            lst.sort(key=lambda x:x[1])
            add_text = ''
            for tup in lst:
                keep_text, dist, lab = tup
                lab_txt = self.label_map[lab]
                lab_txt = '{}{} {}{}'.format(self.tokenizer.cls_token[0], lab_txt, dist, self.tokenizer.cls_token[-1])
                add_text += '{} {} {} '.format(self.tokenizer.sep_token, lab_txt, keep_text)
            out_str = '{} {}'.format(text, add_text).strip()
            out_str = re.sub(' +', ' ', out_str)
            out_txt.append(out_str)
        return out_txt

    def both_train(self, sbert):
        X_train_embeddings = sbert.encode(self.train_ds['text'])
    
        text_idx_dict = self.get_text_idx_dict(sbert, X_train_embeddings, self.train_ds['text'], train=True)

        out_txt = self.get_out_txt(text_idx_dict, 'train')
        train_df = pd.DataFrame({'text':out_txt, 'labels':self.train_ds['labels'], 'label_text':self.train_ds['label_text']})
        self.mod_train = Dataset.from_pandas(train_df)

    
    def both_test(self, sbert, split):
        if split == 'val':
            X = self.val_ds['text']
            X_embeddings = sbert.encode(X)
        
        else:
            X = self.test_ds['text']
            X_embeddings = sbert.encode(X)
        
        text_idx_dict = self.get_text_idx_dict(sbert, X_embeddings, X, train=False)
        
        out_txt = self.get_out_txt(text_idx_dict, split)
        
        if split == 'val':
            val_df = pd.DataFrame({'text':out_txt, 'labels':self.val_ds['labels'], 'label_text':self.val_ds['label_text']})
            self.mod_val = Dataset.from_pandas(val_df)
        
        else:
            test_df = pd.DataFrame({'text':out_txt, 'labels':self.test_ds['labels'], 'label_text':self.test_ds['label_text']})
            self.mod_test = Dataset.from_pandas(test_df)
        
    def mod_st(self, sbert):
        if self.args.lagonnconfig in ['LABEL', 'TEXT']:
            self.get_mod_train(sbert)
            self.get_mod_test(sbert, val=False)
        
        elif self.args.lagonnconfig in ['BOTH']:
            self.both_train(sbert)
            split = 'test'
            self.both_test(sbert, split)
        
        X_train = sbert.encode(self.mod_train['text'])
        y_train = self.mod_train['labels']
        
        clf = LogisticRegression(random_state=MODEL_SEED).fit(X_train, y_train)
        train_predictions = predict_with_sklearn(X_train, y_train, clf)

        X_test = sbert.encode(self.mod_test['text'])
        y_test = self.mod_test['labels']

        test_predictions = predict_with_sklearn(X_test, y_test, clf)

        eval_dict = get_eval_dict(self.args, train_predictions, test_predictions)

        return eval_dict
    
    def mod_for_setfit(self):
        sbert = SentenceTransformer(self.args.st_model)
        if self.args.lagonnconfig in ['LABEL', 'TEXT']:
            self.get_mod_train(sbert)
            self.get_mod_test(sbert, val=True)
        
        elif self.args.lagonnconfig in ['BOTH']:
            self.both_train(sbert)
            split = 'val'
            self.both_test(sbert, split)
    
    def modify_after_setfit(self, sbert_trainer):
        sbert = sbert_trainer.model.model_body
        if self.args.lagonnconfig in ['LABEL', 'TEXT']:
            self.get_mod_train(sbert)
            self.get_mod_test(sbert, val=False)
        
        elif self.args.lagonnconfig in ['BOTH']:
            self.both_train(sbert)
            split = 'test'
            self.both_test(sbert, split)
        
        X_train = self.mod_train['text']
        y_train = self.mod_train['labels']
        train_predictions = predict_with_setfit(X_train, y_train, sbert_trainer)

        X_test = self.mod_test['text']
        y_test = self.mod_test['labels']
        test_predictions = predict_with_setfit(X_test, y_test, sbert_trainer)

        eval_dict = get_eval_dict(self.args, train_predictions, test_predictions)

        return eval_dict
    
    def train_setfit_head(self):
        sbert = self.sbert_trainer.model.model_body
        if self.args.lagonnconfig in ['LABEL', 'TEXT']:
            self.get_mod_train(sbert)
            self.get_mod_test(sbert, val=False)
        
        elif self.args.lagonnconfig in ['BOTH']:
            self.both_train(sbert)
            split = 'test'
            self.both_test(sbert, split)
        
        X_train = sbert.encode(self.mod_train['text'])
        y_train = self.mod_train['labels']
        
        clf = LogisticRegression(random_state=MODEL_SEED).fit(X_train, y_train)
        train_predictions = predict_with_sklearn(X_train, y_train, clf)

        X_test = sbert.encode(self.mod_test['text'])
        y_test = self.mod_test['labels']

        test_predictions = predict_with_sklearn(X_test, y_test, clf)

        eval_dict = get_eval_dict(self.args, train_predictions, test_predictions)

        return eval_dict
    
    def expensive(self):
        sbert = SentenceTransformer(self.args.st_model)
        if self.args.lagonnconfig in ['LABEL', 'TEXT']:
            self.get_mod_train(sbert)
            self.get_mod_test(sbert, val=True)
            self.get_mod_test(sbert, val=False)
        
        elif self.args.lagonnconfig in ['BOTH']:
            self.both_train(sbert)
            self.both_test(sbert, 'val')
            self.both_test(sbert, 'test')
        
        sbert_trainer = do_setfit(self.args, self.mod_train, self.mod_val)
        X_train = self.mod_train['text']
        y_train = self.mod_train['labels']
        train_predictions = predict_with_setfit(X_train, y_train, sbert_trainer)

        X_test = self.mod_test['text']
        y_test = self.mod_test['labels']
        test_predictions = predict_with_setfit(X_test, y_test, sbert_trainer)

        eval_dict = get_eval_dict(self.args, train_predictions, test_predictions)

        return eval_dict
    
    def predict(self):
        if self.args.mode in ['LAGONN_CHEAP']:
            sbert = SentenceTransformer(self.args.st_model)
            eval_dict = self.mod_st(sbert)
            return eval_dict
        
        elif self.args.mode in ['LAGONN', 'LAGONN_LITE']:
            if not self.sbert_trainer:
                self.mod_for_setfit()
                sbert_trainer = do_setfit(self.args, self.mod_train, self.mod_val)
                eval_dict = self.modify_after_setfit(sbert_trainer)
                return eval_dict, sbert_trainer
            
            else:
                eval_dict = self.train_setfit_head()
                return eval_dict
        
        elif self.args.mode in ['LAGONN_EXP']:
            if self.step == 1:
                self.mod_for_setfit()
                sbert_trainer = do_setfit(self.args, self.mod_train, self.mod_val)
                eval_dict = self.modify_after_setfit(sbert_trainer)
            else:
                eval_dict = self.expensive()
            return eval_dict




        
        









