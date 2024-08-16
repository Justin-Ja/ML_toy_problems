# ML Toy Problems

## Description

This repo contains some machine learning (ML) models that are trained to solve some [toy problems](https://medium.com/@vishu54784/what-are-some-good-toy-problems-that-can-be-done-over-a-weekend-by-a-single-coder-in-data-science-6674c88fecff)

Current models are trained to learn the moon dataset from sklearn and spirals from CS231n

## Running the Program

To be able to run the files, you'll need python 3.8 or later to install [pytorch](https://pytorch.org/get-started/locally/) (Cuda or CPU, code is device agnostic), along with mathplotlib and sklearn installed for python.

Any of the learnX.py files are the files that are used to run a model to solve a problem.  You can run said files with the command python3:

```text
python3 learnMoons.py
```

```text
python3 learnSpirals.py
```
The helper file will not run any model code and simply exit if ran directly.

### Passing in Command Line Arguments

Using the above method to execute the program will default the hyperparameters to default vaules. To adjust these parameters, or to change the input sample size/spread, you can pass in command line arguments. To see what parameters you can adjust, add a '-h' flag:

```text
python3 learnMoons.py -h

usage: 
A ML program that trains to learn a binary classification problem
 [-h] [-N SAMPLES] [-n NOISE] [-e EPOCHS] [-l LEARNING_RATE] [-u UNITS] [-s SEED]

options:
  -h, --help            show this help message and exit
  -N SAMPLES, --samples SAMPLES
                        Number of samples to generate
  -n NOISE, --noise NOISE
                        A value between 0 and 1 of how spread out the points are. A higher value means points are further apart.
  -e EPOCHS, --epochs EPOCHS
                        The number of loops should the model train/test for
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        The rate that the optimizer will update the model's parameters at.
  -u UNITS, --units UNITS
                        Number of neurons to use per ML layer. Higher value helps the model be more accurate, at the cost of training time
  -s SEED, --seed SEED  Value of seed for RNG
```

```text
python3 learnSpirals.py -h

usage: 
A ML program that trains to learn a multi-class classification problem
 [-h] [-N SAMPLES] [-c CLASSES] [-t THETA] [-e EPOCHS] [-l LEARNING_RATE] [-u UNITS] [-s SEED]

options:
  -h, --help            show this help message and exit
  -N SAMPLES, --samples SAMPLES
                        Number of points PER CLASS to generate
  -c CLASSES, --classes CLASSES
                        Number of groups/classes to create when generating the spiral
  -t THETA, --theta THETA
                        A value between 0 and 1 of how spread out the points are from their "central line". A higher value means points are further apart.
  -e EPOCHS, --epochs EPOCHS
                        The number of loops should the model train/test for
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        The rate that the optimizer will update the model's parameters at.
  -u UNITS, --units UNITS
                        Number of neurons to use per ML layer. Higher value helps the model be more accurate, at the cost of training time
  -s SEED, --seed SEED  Value of seed for RNG
```

## Example Input/Output Images

### LearnMoons.py

Moon input set and the output after the model was trained and tested:

![updated_moon_input](https://github.com/Justin-Ja/ML_toy_problems/assets/95664856/a88e978e-b5a3-4456-8ea6-34c371d6cd99 "Moons input: An XY plot with two groups of points, yellow and black, both forming the shape of cresent moons")

- - - -

![updated_moon_output](https://github.com/Justin-Ja/ML_toy_problems/assets/95664856/45ff74b3-2532-42b1-a037-7b371806770d "Moons output: The same plot as the input, but with a line created by the program to separate both groups of points")

### LearnSpirals.py

A spiral input set and the output when the model was finished being trained and tested:

![spiral_input](https://github.com/Justin-Ja/ML_toy_problems/assets/95664856/c838fdd0-fac7-47d7-8c2d-14d4d69b54aa "Spiral input: An XY plot of six groups of points forming a spiral")

- - - -

![spiral_output](https://github.com/Justin-Ja/ML_toy_problems/assets/95664856/3986c77c-9cae-4e6f-a97e-eddf321723eb "Spirals output: The same spiral as the input, except the groups are separated by lines")

### Q&A

* Why is the accuracy decreasing/loss increasing?

This is most likely due to the model overfitting the data - it's become too familiar with it and is struggling to generalize it. Try either increasing the sample size and/or increasing the learning rate or epochs.

### Limitations

Haven't tested with GPU since only using CPU pytorch, but, in theory, it should work (famous last words)
