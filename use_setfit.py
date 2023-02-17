from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer
from setup_utils import seed_everything, MODEL_SEED

def do_setfit(args, train_ds, val_ds):
    if len(set(train_ds['labels'])) != 2:
        metric = 'f1'
    else:
        metric = 'ap'
    seed_everything(MODEL_SEED)

    model = SetFitModel.from_pretrained("sentence-transformers/{}".format(args.st_model))
    
    trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    loss_class=CosineSimilarityLoss,
    metric=metric, #was f1
    batch_size=16,
    seed=MODEL_SEED,
    num_iterations=20, # The number of text pairs to generate for contrastive learning
    num_epochs=1, # The number of epochs to use for constrastive learning
    column_mapping={"text": "text", "labels": "label"} # Map dataset columns to text/label expected by trainer
    )

    trainer.train()
    return trainer

def predict_with_setfit(X, y, sbert_trainer):
    print('predicting with setfit')
    clf = sbert_trainer.model
    y_logit = clf.predict_proba(X)
    if len(y_logit.shape) == 2:
        y_logit = y_logit[:, 1]
    y_pred = clf.predict(X)
    predictions = (y_logit, y_pred, y)
    
    return predictions
