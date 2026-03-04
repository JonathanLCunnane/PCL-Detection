# PCL-Detection
PCL means Patronising and Condescending Language. This project is part of Imperial College London's NLP course where we aim to detect PCL training from the data from
[here](https://github.com/Perez-AlmendrosC/dontpatronizeme).

## Best Model

Had to remove models cause Git LFS limit, oops very sorry.
**NBNB: I reached the Git LFS limit, here is a link to a Google Drive with the model: https://drive.google.com/drive/folders/1QQZtiSTTaTAiaStK4ZqHtDGncVECoQkv?usp=sharing**
**NB: Due to Git LFS restrictions, I have used `split` to break up the larger models so they can be uploaded.**
**If you need to use a model running `cat model.pt.part_* > model.pt` will reconstruct them**

The best model is the one which performed the best on the *validation* set used for hyperparameter tuning.

I have made a *copy* of the notebook for the best model into the [best model folder](./BestModel).
This includes the [notebook](./BestModel/exp_L_verbalizer.ipynb),
[state dictionary](./BestModel/out/exp_L_verbalizer_best_model.pt),
[hyperparameters](./BestModel/out/exp_L_verbalizer_best_params.json),
and any other corresponding outputs for the best model. 
You can see how the model is generated and saved in the notebook, and how it is loaded further down
for evaluation and how it is used in evaluation in the global and local evaluation notebooks.

## Global Evaluation, `dev.txt` , and `test.txt`

In the [global evaluation notebook](./global_evaluation.ipynb) we use our 
[best model](./exp/exp_L_verbalizer.ipynb) to evaluate whether PCL is present (`1`) or not present (`0`), 
with results for the dev set and label-less test set being in [dev.txt](./dev.txt) and [test.txt](./test.txt)
respectively. You can click the links or see them directly in the root of this repository!

## Local Evaluation & EDA

We carry out any local evaluation and error analysis in [this notebook](./error_analysis.ipynb), and EDA is carried out in [this other notebook](./eda.ipynb), both in the root of this repository. Any files generated
from the global evaluation, local evaluation, and EDA should be in the [root output folder](./out/).

## Experiments

All experiments carried out can be found in the [experiments folder](./exp/) from the [initial
attempt/baseline](./exp/baseline_model.ipynb) to the [best model](./exp/exp_L_verbalizer.ipynb), and likewise
the outputs of these experiments, including models should be in the [experiments output folder](./exp/out/).

Note, I did not have time to run the final experiment nooo....
