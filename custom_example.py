from datasets import load_dataset
from lagonn import LaGoNN

#create huggingface dataset or load one
sst2 = load_dataset('SetFit/sst2') #for example, sst2 and sst5
sst2 = sst2.rename_column('label', 'labels')

sst5 = load_dataset('SetFit/sst5')
sst5 = sst5.rename_column('label', 'labels')
#dataset needs to have a "text", "labels", and "label_text" field. We also assume a training, validation, and test split.

sst2_train_ds = sst2['train']
sst2_val_ds = sst2['validation']
sst2_test_ds = sst2['test']

sst5_train_ds = sst5['train']
sst5_val_ds = sst5['validation']
sst5_test_ds = sst5['test']


#We need to pass LaGoNN a dictionary of configurations. We used default SetFit settings in our experiments.
config_dict = {'lagonn_mode': 'LAGONN_CHEAP', # Don't finetune the embedding model 
               'lagonn_config': 'LABEL', # Use the gold label and Euclidean distance to modify input text
               'sample_size': 100, # How many examples per label to fine-tune the embedding model
               'st_model': 'paraphrase-mpnet-base-v2', # Choose your Sentence Transformer
               'batch_size': 16, # SetFit batch size
               'model_seed': 0, # Seed for training
               'metric': 'f1', # metric to pass to SetFit Trainer
               'num_iterations': 20, # The number of text pairs to generate for contrastive learning (see https://github.com/huggingface/setfit)
               'num_epochs': 1, # The number of epochs to use for contrastive learning (see https://github.com/huggingface/setfit)
               'sample_seed': 0} # Seed used to sample data

lgn = LaGoNN(train_ds=sst2_train_ds, val_ds=sst2_val_ds, test_ds=sst2_test_ds, config_dict=config_dict)
eval_dict = lgn.custom()
# If the dataset is binary, we compute the average precision, binary F1, and accuracy score.
print(eval_dict)

lgn = LaGoNN(train_ds=sst5_train_ds, val_ds=sst5_val_ds, test_ds=sst5_test_ds, config_dict=config_dict)
eval_dict = lgn.custom()
#If the dataset is multiclass, we compute the macro and micro F1 and the accuracy score.
print(eval_dict)

#Let's try fine-tuning the embedding model on a subset of the training data.
config_dict['lagonn_mode'] = 'LAGONN'

lgn = LaGoNN(train_ds=sst2_train_ds, val_ds=sst2_val_ds, test_ds=sst2_test_ds, config_dict=config_dict)
eval_dict = lgn.custom()
print(eval_dict)

lgn = LaGoNN(train_ds=sst5_train_ds, val_ds=sst5_val_ds, test_ds=sst5_test_ds, config_dict=config_dict)
eval_dict = lgn.custom()
print(eval_dict)

#Finally, we can fine-tune the embedding model on all of the training data (this might take a while...)
config_dict['lagonn_mode'] = 'LAGONN_EXP'

lgn = LaGoNN(train_ds=sst2_train_ds, val_ds=sst2_val_ds, test_ds=sst2_test_ds, config_dict=config_dict)
eval_dict = lgn.custom()
print(eval_dict)

lgn = LaGoNN(train_ds=sst5_train_ds, val_ds=sst5_val_ds, test_ds=sst5_test_ds, config_dict=config_dict)
eval_dict = lgn.custom()
print(eval_dict)

print("Job's done!")