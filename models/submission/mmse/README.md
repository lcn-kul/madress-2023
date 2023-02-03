# MMSE Models

All models use acoustic features with/without covariates. Acoustic features are a
sequence of 10 eGeMAPS features (dim=25). Covariates are age, gender, education and
the AD probability that is estimated by AD Model5.

- Model1 = eGeMAPS
- Model2 = eGeMAPS + age
- Model3 = eGeMAPS + education
- Model3 = eGeMAPS + ad_prob
- Model5 = eGeMAPS + age/gender/education/ad_prob

Predictions can be inserted into the second column of

```
data/raw/ADReSS-M-format_task2.csv
```
