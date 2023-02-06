# AD Models

All models use acoustic features with covariates. Acoustic features are a sequence of
10 eGeMAPS features (dim=25). Covariates are age, gender and education. The only
difference between the models is random initialization.

Predictions can be inserted into the second column of

```
data/raw/ADReSS-M-format_task1.csv
```
