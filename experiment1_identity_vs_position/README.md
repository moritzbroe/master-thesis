## Usage
Run main.py without arguments to generate the data from the thesis.

## Implementation
### shapes.py
Defines an image dataset where white objects (squares, circles and triangles) move around on a black background with constant velocity and one can generate images at different time points. Running it with ```python3 shapes.py``` will visualize one such sequence where multiple objects move around. Playing around with the settings at the bottom of the file leads to different such datasets. In main.py, the dataset is used for generating image triples where a single object moves around.

### loss.py
Defines the standard VICReg Loss and a variant for penalizing the second derivative, taking triples of features as input.
