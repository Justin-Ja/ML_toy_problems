# ML Toy Problems

## Description

This repo contains some machine learning (ML) models that are trained to solve some [toy problems](https://medium.com/@vishu54784/what-are-some-good-toy-problems-that-can-be-done-over-a-weekend-by-a-single-coder-in-data-science-6674c88fecff) 

Current models are trained to learn the moon dataset from sklearn and spirals from CS231n

## Running the Program

To be able to run the files, you'll need python 3.8 or later to install [pytorch](https://pytorch.org/get-started/locally/) (Cuda or CPU, code is device agnostic), along with mathplotlib and sklearn installed for python.

Any of the learnX.py files are the files that can be executed to run a model to solve a problem. You can run the models in the learnX.py files with python3:

```
python3 learnMoons.py
```

To adjust how efficient the ML model learns, or to change the input sample to be more spread, you can edit the constants at the top of the learn files.

At some point, I will implement a way of user input to update these values without having to edit the file directly

## Example Input/Output Images

Moon dataset, input set and the output when the model was trained and tested:
![moon_input](https://github.com/Justin-Ja/ML_toy_problems/assets/95664856/efe40259-f4fd-4976-8749-16fcf7121b2d)

![moon_output](https://github.com/Justin-Ja/ML_toy_problems/assets/95664856/92e8fa43-734a-4efd-8f66-066bd2c2cefe)

A spiral dataset:

![spiral_input](https://github.com/Justin-Ja/ML_toy_problems/assets/95664856/c838fdd0-fac7-47d7-8c2d-14d4d69b54aa)

![spiral_output](https://github.com/Justin-Ja/ML_toy_problems/assets/95664856/3986c77c-9cae-4e6f-a97e-eddf321723eb)


### Limitations

Haven't tested with GPU since only using CPU pytorch, but, in theory, it should work (famous last words)
