## Purpose
These python files allow for training models with a self-supervised loss using gradient-isolated blocks, where each block uses different augmentations. These augmentation parameters can then be optimized using the optuna library. Everything only tested with STL-10.

## Implementation and Usage
### ssl_loader.py
Creates mostly standard pytorch dataloaders that return pairs of augmented images. For efficiency, the dataloader is created with ```get_basic_ssl_loader``` and then different transformation parameters can be set using ```set_transformations```. This prevents the dataset being reloaded for the training of every block (as each block uses different augmentations and would hence require a new dataloader). Run ssl_loader.py directly to visualize samples, change the parameters at the end of the code to use different augmentations.

### train_test.py
Most important functions in here. This defines the logic to train a single block/layer of a standard torch.nn.Sequential model, and implements a function to test a hyperparameter configuration consisting of the augmentation parameters min_size, color_jitter_strength, flip and of the learning rate and invariance coefficient for the VICReg loss used in each layer of a model. It trains each block with the corresponding parameters and then evaluates the final model through linear evaluation.

### main.py
Runs hyperparameter optimization for 8 layer CNN to optimize the augmentations and other hyperparameters for each layer with the TPE sampler in optuna. Running ```main.py``` continues the study ```study_tpe.db``` which is also provided - e.g. renaming the existing study and running main would start a new study.

### main2.py
Takes the existing tpe study, extracts the 5 best hyperparmeter combinations, averages them and finetunes them using the CMA-ES sampler. Writes to ```study_cma_finetune.db``` which is also provided.

### final_evaluation.py
Script with some methods to check the final performance of the final hyperparamters (or manually entered/modified combinations)
