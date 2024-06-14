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

![updated_moon_input](https://github.com/Justin-Ja/ML_toy_problems/assets/95664856/a88e978e-b5a3-4456-8ea6-34c371d6cd99)

![updated_moon_output](https://github.com/Justin-Ja/ML_toy_problems/assets/95664856/45ff74b3-2532-42b1-a037-7b371806770d)


A spiral dataset:

![spiral_input](https://github.com/Justin-Ja/ML_toy_problems/assets/95664856/c838fdd0-fac7-47d7-8c2d-14d4d69b54aa)

![spiral_output](https://github.com/Justin-Ja/ML_toy_problems/assets/95664856/3986c77c-9cae-4e6f-a97e-eddf321723eb)


### Limitations

Haven't tested with GPU since only using CPU pytorch, but, in theory, it should work (famous last words)
