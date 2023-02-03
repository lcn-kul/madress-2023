# madress-2023

## 1. Overview
Source code for LCN submission for MADReSS challenge.

## 2. Installation
*Note: this software was developed for Linux.*

**Clone Repository**
```
git clone https://github.com/lcn-kul/madress-2023.git
cd madress-2023
```

**Create Virtual Environment**
```
make create_environment
source venv/bin/activate
make requirements
```

## 3. Reproducing Results

Run the following commands to reproduce the results.

**1. Download challenge data**

See the [raw data](/data/raw/README.md) for more information.

**2. Process raw CSVs**

```
make csvs
```

**3. Do training**

This step performs
- eGeMAPS feature extraction
- English pretraining
- English+Greek fine-tuning
- model-averaging
- test set prediction 

for the AD and MMSE models. The feature extraction takes 10-15 minutes and the other steps will take another 10-15 minutes.

```
make train
```

**4. Prepare submission files**

Insert predictions from the folders
- `models/trained_model_[config]_avg/prediction.txt`

into the corresponding submission format file:

- AD : `data/raw/ADReSS-M-format_task1.csv`
- MMSE : `data/raw/ADReSS-M-format_task2.csv`
