# Models

Submission models can be found in the `submission` folder.

Trained models will appear here as folders `trained_model_[config]_avg` containing:
- epoch checkpoints (`all-epoch-xxx.ckpt`)
- train/validation epoch losses (`losses.csv`)
- train/validation primary metric (`metric.txt`)
- test predictions (`prediction.txt`)

For averaged models, only `prediction.txt` will be filled in.
