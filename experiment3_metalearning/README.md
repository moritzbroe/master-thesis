## Purpose
This is for adapting the hyperparameters of parametric learning rules through hypergradient descent, hence performing metalearning. In particular, consider a neural network (or other feature extractor/...) with parameters $\theta$ and a parametric update rule $u$ which takes $\theta$, some hyperparameters $\psi$ and an input (which could be a single input, batch of inputs, batch of augmented images, input-target pairs etc) and returns new parameters $\theta'$. Usually, this update rule would pass inputs through the neural net and then somehow calculate the new parameters. Now this update is applied $N$ times, then some loss is evaluated on the final parameters $\theta_N$, and this loss is used to backpropagate through all the $N$ updates to get gradients for $\psi$ and update it.

The main problem with regards to implementing this is that using a standard pytorch model with nn.Parameter entries for $\theta$ makes it diffucult to
1. unroll the updates
2. use checkpointing - i.e. to not save intermediate activations, only the parameters, and recompute the activations during the backward pass, which is necessary to not run out of memory.

Hence, the parameters $\theta$ have to be seperated into a structure like a list/dictionary of tensors and passed to the update function (and also to the model for calculating outputs). The implementation here only implements fully local, hebbian-like learning rules for lateral inhibitory networks, but one could relatively easily add other types of learning rules and models.

## Usage
MNIST is automatically downloaded.\
the three baseline files and find_oja.py can be run without arguments, see explanations below.\
main.py provides some examples for running it (those from the thesis) at the top of the code as comments, but offers a lot more command line arguments. see comments in code.

## Implementation
### data
```mnist_datasets.py``` contains code for standard mnist datasets and mnist ssl dataset for using augmented image pairs. could easily add other datasets in a similar fashion. 

### models
only models implemented so far are lateral inhibitory layers in ```models_fully_local.py``` and sequences of them, i.e. lateral inhibitory MLPs. Again, a model does not store its own parameters, these are passed to it for the forward pass. other models could be implemented similarly, but this takes a bit more effort than the standard pytorch models.

### learning_rules
learning rules are standard pytorch modules with nn.Parameter entries for their trainable hyperparameters. They implement functions as explained above, which take parameters $\theta$ and inputs $x$ (the hyperparameters $\psi$ are stored in the learning rule object) and return new parameters $\theta'$. Here, only two learning rules are implemented, which apply polynomial hebbian-like updates to the parameters of the implemented models. One of them uses simply batches of inputs (i.e. it is a parametric, fully local, unsupervised learning rule) while the other one takes two batches, i.e. a batch of pairs, as input and can hence in principle perform self-supervised learning.

### initializers
initializers are pytorch modules that can be called to get initial parameters for a model. in particular, a model (which does not store its own parameters) is passed to one of the initializer's function to get initial weights, biases etc. the initializer implemented here works for the models presented above. it has trainable initialization hyperparameters - in particular, forward weights, biases and lateral weights are initialized with gaussian distributions where mean and std can be made trainable.

### baselines
these three python files are for setting baselines: \
```baseline_pca.py``` can be run directly and does uses different values for the feature dimension k from 1 to 28^2, extracts the first k principal components from the mnist dataset, and calculates the accuarcy obtained with a linear classifier.\
```baseline_mlp.py``` uses a single fully connected layer with k outputs (k can be set at the start of the code), randomly initialized, to extract features from mnist and calculates the accuracy of a linear classifier evaluated on top of them. the initialization parameters are then trained through hypergradient descent (proper initialization significantly improves linear evaluation performance for large k).\
```baseline_vicreg``` trains a simple MLP with layer sizes 256,32 with vicreg on mnist and checks the linear classifier's performance trained on top of the features.\

### train_test
```train.py``` applies a learning rule for several steps and returns the updated parameters. it implements checkpointing (which is simple with the above implementation).\
```linear_probe.py``` trains a linear classifier on top of the representations extracted with a model on some dataset and passes a final batch through it, yielding a loss value that is connected to the computational graph \
```hypgraddesc.py``` implements the hypergradient descent procedure, mainly using the function from ```train.py```, with many options. In particular, one can also implement things like truncated backpropagation (where not all of the updates are differentiated through) with different options. also, one passes a loss function which is evaluated on the final parameters and is used to guide the evolution of the hyperparameters\

### ```main.py```
Creates a lateral inhibitory MLP and the initializer and learning rule (either for unsupervised or self-supervised learning) described above and runs the hypergradient descent function on it. in particular, the linear_probe loss is used as a loss function.

### ```find_oja.py```
Implements a simple single linear neuron, as used by Oja's rule, and uses a learning rule that has Oja's rule as a special case (i.e. one of the hyperparameter combinations correspond to Oja's rule). then, the learning rule is adapted using the hypergradient descent method to minimize the distance of the weight vector to the first principal direction, which mostly recovers Oja's rule.
