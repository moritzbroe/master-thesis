# Code for Master's Thesis
Explanations for each experiment can be found in the folders.\
Setting up a conda environment like this should allow running all code:\
```conda create -n localssl -c rapidsai -c conda-forge -c nvidia cuml cudf 'cuda-version>=12.0,<=12.8' 'pytorch=*=*cuda*' torchvision```\
```conda activate localssl```\
```pip install optuna cmaes scikit-learn albumentations matplotlib scikit-image```\
In particular, the non-standard librariy cuML is used for accelerating logistic regression in experiments 2 and 3, and albumentations is used for accelerating image augmentations in experiment 2
