from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer
from use_setfit import do_setfit, predict_with_setfit
from setup_utils import get_eval_dict, predict_with_sklearn, MODEL_SEED
from train import train_transformer, train_head, predict_with_trainer

def linear_probe(input_train_ds, input_test_ds, args):
    sbert = SentenceTransformer(args.st_model)
    
    X_train = sbert.encode(input_train_ds['text'])
    y_train = input_train_ds['labels']
    
    clf = LogisticRegression(random_state=MODEL_SEED).fit(X_train, y_train)

    train_predictions = predict_with_sklearn(X_train, y_train, clf)

    X_test = sbert.encode(input_test_ds['text'])
    y_test = input_test_ds['labels']
    
    test_predictions = predict_with_sklearn(X_test, y_test, clf)
    
    eval_dict = get_eval_dict(args, train_predictions, test_predictions)
    
    return eval_dict

def setfit_once(input_train_ds, input_test_ds, sbert_trainer, step, args):
    if step > 1:
        sbert = sbert_trainer.model.model_body  

        X_train = sbert.encode(input_train_ds['text'])
        y_train = input_train_ds['labels']

        X_test = sbert.encode(input_test_ds['text'])
        y_test = input_test_ds['labels']
    
        if args.mode in ['KNN']:
            clf = KNeighborsClassifier(3).fit(X_train, y_train)
        elif args.mode in ['LOG_REG', 'SETFIT_LITE']:
            clf = LogisticRegression(random_state=MODEL_SEED).fit(X_train, y_train)
    
        train_predictions = predict_with_sklearn(X_train, y_train, clf)

        test_predictions = predict_with_sklearn(X_test, y_test, clf)
    
    elif step == 1:
        if args.mode in ['LOG_REG','SETFIT_LITE']:
            X_train = input_train_ds['text']
            y_train = input_train_ds['labels']
            train_predictions = predict_with_setfit(X_train, y_train, sbert_trainer)

            X_test = input_test_ds['text']
            y_test = input_test_ds['labels']
            test_predictions = predict_with_setfit(X_test, y_test, sbert_trainer)
        
        elif args.mode in ['KNN']:
            sbert = sbert_trainer.model.model_body
            X_train = sbert.encode(input_train_ds['text'])
            y_train = input_train_ds['labels']

            X_test = sbert.encode(input_test_ds['text'])
            y_test = input_test_ds['labels']
            
            clf = KNeighborsClassifier(3).fit(X_train, y_train)
            
            train_predictions = predict_with_sklearn(X_train, y_train, clf)

            test_predictions = predict_with_sklearn(X_test, y_test, clf)
    
    eval_dict = get_eval_dict(args, train_predictions, test_predictions)
        
    return eval_dict

def setfit_every_step(input_train_ds, input_val_ds, input_test_ds, args):
    sbert_trainer = do_setfit(args, input_train_ds, input_val_ds)
   
    X_train = input_train_ds['text']
    y_train = input_train_ds['labels']
    
    train_predictions = predict_with_setfit(X_train, y_train, sbert_trainer)

    X_test = input_test_ds['text']
    y_test = input_test_ds['labels']
    
    test_predictions = predict_with_setfit(X_test, y_test, sbert_trainer)
    
    eval_dict = get_eval_dict(args, train_predictions, test_predictions)
    
    if args.mode in ['SETFIT_LITE']:
        return eval_dict, sbert_trainer
    else:
        return eval_dict

def train_head_every_step(input_train_ds, input_val_ds, input_test_ds, args, transformer_trainer, balance):
    transformer_trainer = train_head(input_train_ds, input_val_ds, transformer_trainer, args, balance)
    train_predictions, test_predictions = predict_with_trainer(transformer_trainer, input_train_ds, input_test_ds, args)  
    eval_dict = get_eval_dict(args, train_predictions, test_predictions)
    
    return eval_dict  

def train_every_step(input_train_ds, input_val_ds, input_test_ds, args, balance):
    transformer_trainer = train_transformer(input_train_ds, input_val_ds, args, balance)
    train_predictions, test_predictions = predict_with_trainer(transformer_trainer, input_train_ds, input_test_ds, args)        
    eval_dict = get_eval_dict(args, train_predictions, test_predictions)
    
    return eval_dict

def train_once(input_train_ds, input_val_ds, input_test_ds, args, balance):
    transformer_trainer = train_transformer(input_train_ds, input_val_ds, args, balance)
    train_predictions, test_predictions = predict_with_trainer(transformer_trainer, input_train_ds, input_test_ds, args)
    eval_dict = get_eval_dict(args, train_predictions, test_predictions)

    return transformer_trainer, eval_dict