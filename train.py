import os
import numpy as np
from datasets import Dataset
from sklearn.metrics import average_precision_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from setup_utils import seed_everything, MODEL_SEED

os.environ["WANDB_DISABLED"] = "true"

def tokenize_and_encode(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)
    
def encode_dataset(dataset):
    cols = dataset.column_names
    cols.remove("labels")
    ds_enc = dataset.map(tokenize_and_encode, batched=True, remove_columns=cols)
    ds_enc.set_format("torch")
    return ds_enc

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=-1)  
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')
    return {"metric": f1}

def compute_ap(p):    
    pred, labels = p
    logit_pred = pred[:, 1]
    avg_pre = average_precision_score(labels, logit_pred)
    return {"metric": avg_pre}

def train_head(train_ds, val_ds, trainer, args, balance):
    seed_everything(MODEL_SEED)    
    num_labels = len(set(train_ds['labels']))
          
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.transformer_clf, padding=True)
   
    model = trainer.model
    for name, param in model.named_parameters():
	    if 'classifier' not in name: # classifier layer
		    param.requires_grad = False
    
    train_ds_enc = encode_dataset(train_ds)
    val_ds_enc = encode_dataset(val_ds)

    if num_labels == 2:
        function = compute_ap
    else:
        function = compute_metrics

    training_args = TrainingArguments(
                        "cb_models/training_with_callbacks_{}_{}_{}_{}".format(args.task, args.mode, args.seed, balance),
                        num_train_epochs=70,
                        logging_steps=200,
                        evaluation_strategy="steps",
                        eval_steps = 50,
                        save_total_limit = 5,
                        learning_rate=1e-5,
                        push_to_hub=False,
                        metric_for_best_model ='metric',
                        load_best_model_at_end=True,
                        seed=MODEL_SEED,        
                        )
            
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_enc,
        eval_dataset=val_ds_enc,
        tokenizer=tokenizer,
        compute_metrics=function,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        )
    
    trainer.train()
    return trainer


def train_transformer(train_ds, val_ds, args, balance):
    seed_everything(MODEL_SEED)
    num_labels = len(set(train_ds['labels']))
          
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.transformer_clf, padding=True)
   
    model = AutoModelForSequenceClassification.from_pretrained(args.transformer_clf, num_labels=num_labels).to('cuda')
    
    train_ds_enc = encode_dataset(train_ds)
    val_ds_enc = encode_dataset(val_ds)

    if num_labels == 2:
        function = compute_ap
    else:
        function = compute_metrics
    
    training_args = TrainingArguments(
                "cb_models/training_with_callbacks_{}_{}_{}_{}".format(args.task, args.mode, args.seed, balance),
                num_train_epochs=70,
                logging_steps=200,
                evaluation_strategy="steps",
                eval_steps = 50,
                save_total_limit = 5,
                learning_rate=1e-5,
                push_to_hub=False,
                metric_for_best_model ='metric',
                load_best_model_at_end=True,
                seed=MODEL_SEED,        
                )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_enc,
        eval_dataset=val_ds_enc,
        tokenizer=tokenizer,
        compute_metrics=function,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        )
            
    trainer.train()
    return trainer

def predict_with_trainer(transformer_trainer, train_ds, test_ds, args):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.transformer_clf, padding=True)

    train_ds_enc = encode_dataset(train_ds)
    test_ds_enc = encode_dataset(test_ds)

    train_predictions = transformer_trainer.predict(train_ds_enc)
    test_predictions = transformer_trainer.predict(test_ds_enc)

    return train_predictions, test_predictions