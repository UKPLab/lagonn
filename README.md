# Like a Good Nearest Neighbor: LaGoNN

Source code and data for Like a Good Nearest Neighbor: Practical Content Moderation with Sentence Transformers.

Contact person: Luke Bates, luke's_first_name.luke's_last_name@tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

## Project structure
* `main.py` -- code file that uses the other code files
* `modelling.py` -- code file for baselines
* `setup_utils.py` -- util code
* `train.py` -- code file for transformers
* `use_setfit.py` -- code file using and predicting with SetFit
* `dataframe_with_val/` -- data files
* `out_jsons/` -- where result jsons will be written
* `lagonn_both_examples/` -- examples from Appendix of LaGoNN BOTH output


## Requirements
Our results were computed in Python 3.9.13 with a 40 GB NVIDIA A100 Tensor Core GPU. Note that files will be written to disk if the code is run.
Our datasets are sadly too big for GitHub, but you can access them here: https://drive.google.com/drive/folders/1m9lq5VDki4vfDP1GDROEeqY4WT9_ZNfL?usp=sharing
Please make sure to add the folder to the code directory.

## Installation
To setup, please follow the instructions below.
```
python -m venv mvenv
source mvenv/bin/activate
pip install -r requirements.txt
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
 
Then, you can run the code with `python main.py`. You can specifiy which configurations by passing arguments to python. 
* There are the following modes from the paper: 
    * PROBE (Sentence Transformer + logistic regression)
    * LOG_REG (Log Reg)
    * KNN (kNN)
    * SETFIT_LITE (SetFit_lite)
    * SETFIT (SetFit)
    * ROBERTA_FREEZE (RoBERTa_freeze)
    * ROBERTA_FULL (RoBERTa_full)
    * LAGONN_CHEAP (LaGoNN_cheap)
    * LAGONN (LaGoNN)
    * LAGONN_LITE (LaGoNN_lite) 
    * LAGONN_EXP (LaGoNN_exp)
* If you use a LaGoNN-based mode, you will also need to specific a LaGoNN Config:
    * LABEL
    * TEXT
    * BOTH

* You can pass any Sentence Transformer. We used paraphrase-mpnet-base-v2.
* You can pass any Transformer. We used roberta-base.
* You can pass any dataset we used as a 'task'. Note that we assume a `text`, `labels`, and `label_text` field and a training, validation, and test dataset.
* You can pass any seed for sampling data. We used 0, 1, 2, 3, and 4.
* You can turn on/off the file writer.

For example, if you wish to reproduce our LaGoNN_cheap results with seed = 0 on insincere-questions and write files to disk:
```
python main.py --ST_MODEL=paraphrase-mpnet-base-v2\
               --SEED=0\
               --TASK=insincere-questions\
               --MODE=LAGONN_EXP\
               --LAGONN_CONFIG=LABEL\
               --WRITE=True\
```
If you wish to reproduce our RoBERTa_full results on Toxic conversations with seed  = 3 and not write files to disk:
```
python main.py --TRANSFORMER_CLF=roberta-base\
               --SEED=3\
               --TASK=toxic_conversations\
               --MODE=ROBERTA_FULL\
```
If you wish to use LaGoNN_exp on Hate Speech Offensive with the BOTH config and write files to disk with seed = 2:
```
python main.py --ST_MODEL=paraphrase-mpnet-base-v2\
               --SEED=2\
               --TASK=hate_speech_offensive\
               --MODE=LAGONN_EXP\
               --LAGONN_CONFIG=BOTH\
               --WRITE=True\
```

### Expected results
Once finished, results will be written in the following format:
`out_jsons/{task}/{mode}/{balance}/{seed}/{step}/(LaGoNN Config!)results.json`
Note that you will need to complete all five (0-4) seeds. This is because we report the average over the five seed for both the macro F1 and average precision.

### Use LaGoNN in your own work
Coming soon!