# ML Toy Problems

Need pytorch, sklearn and matplotlib installed to run

## Running the Program

To be able to run the files, you'll need pytorch (Cuda or CPU, code is device agnostic), along with mathplotlib and sklearn installed.
Any of the learnX.py files are the files that can be executed to run a model to solve a problem. You can run them with python3:

```
python3 learnMoons.py
```

To adjust how efficient the ML model learns, or to change the input sample to be more spread, you can edit the constants at the top of the learn files.

## Description

This repo contains some machine learning (ML) models that are trained to solve some toy problems (LINK) 
So far there are models to learn the moon dataset from sklearn and spirals from CS231n

### Q & A
Why print plots to images instead of plt.show()?

Im on WSL, cant view plots via terminal