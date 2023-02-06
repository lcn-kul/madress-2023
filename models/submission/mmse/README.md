# MMSE Models

All models use acoustic features with/without covariates. Acoustic features are a
sequence of 10 eGeMAPS features (dim=25). Covariates are age, gender, education and
the AD probability that is estimated by the AD model (since we don't know ahead of
time which model gives the best predictions, we take the average prediction of the 5
submitted AD models). The only difference between the MMSE models is random
initialization.

Predictions can be inserted into the second column of

```
data/raw/ADReSS-M-format_task2.csv
```
