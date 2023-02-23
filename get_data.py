import os
from datasets import load_dataset

folder = 'dataframes_with_val'
if not os.path.exists(folder):
    os.makedirs(folder)

for task in ['insincere-questions', 'amazon_counterfactual_en', 'toxic_conversations', 'hate_speech_offensive']:
    ds = load_dataset('UKPLab/{}'.format(task))
    for split, dset in ds.items():
        if split == 'validation':
            dset.to_csv("{}/{}_{}.csv".format(folder, task, 'val'))
        else:
            dset.to_csv("{}/{}_{}.csv".format(folder, task, split))

task = 'liar'
ds = load_dataset('UKPLab/{}'.format(task))
for split, dset in ds.items():
    if split == 'validation':
        dset.to_csv("{}/{}_need_fix_{}.csv".format(folder, task, 'val'))
    else:
        dset.to_csv("{}/{}_need_fix_{}.csv".format(folder, task, split))

print("Job's Done!")

