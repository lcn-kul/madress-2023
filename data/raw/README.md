# Raw Data

## 1. Required Audio Files

- `sample-gr/`: Greek sample audio files for ADReSS-M challenge
  [[download]](https://media.talkbank.org/dementia/English/0extra/ADReSS-M-sample-gr.tgz)
- `train/` : English training audio files for ADReSS-M challenge
  [[download]](https://media.talkbank.org/dementia/English/0extra/ADReSS-M-train.tgz)
- `test-gr/` : Greek test audio files for ADReSS-M challenge
  [[download]](https://media.talkbank.org/dementia/English/0extra/ADReSS-M-test-gr.tgz)

*Note: links are protected behind DementiaBank log-in.*

## 2. Required CSV Files

- `ADReSS-M-format_task1.csv` : test submission file template for task 1 AD 
  [[source]](https://media.talkbank.org/dementia/English/0extra/ADReSS-M-format_task1.csv)
- `ADReSS-M-format_task2.csv` : test submission file template for task 2 MMSE
  [[source]](https://media.talkbank.org/dementia/English/0extra/ADReSS-M-format_task2.csv)
- `ADReSS-M-meta.csv` : test subjects meta data
  [[source]](https://media.talkbank.org/dementia/English/0extra/ADReSS-M-meta.csv)
- `sample-gr-groundtruth.csv` : sample Greek CSV (available with `sample-gr/`)
- `training-groundtruth.csv` : training English CSV (available with `train/`)

*Note: links are protected behind DementiaBank log-in.*

## 3. Generated CSV Files

- `en-balanced-groundtruth.csv` : balanced English dataset (114 AD - 114 Control)
- `en-balanced-train-groundtruth.csv` : 80% balanced English train split (91 AD - 91 Control)
- `en-balanced-val-groundtruth.csv` : 20% balanced English validation split (23 AD - 23 Control)

These files have been generated by running:
```
cd data/raw/
python3 create_en_balanced_csv.py
```

- `sample-gr-train-2folds-0-groundtruth.csv` : sample-gr fold0 train split
- `sample-gr-train-2folds-1-groundtruth.csv` : sample-gr fold1 train split
- `sample-gr-val-2folds-0-groundtruth.csv` : sample-gr fold0 validation split
- `sample-gr-val-2folds-1-groundtruth.csv` : sample-gr fold1 validation split

These files have been generated by running:
```
cd data/raw/
python3 create_balanced_kfolds.py
```