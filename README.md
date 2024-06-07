# ML Toy Problems

## Description

This repo contains some machine learning (ML) models that are trained to solve some [toy problems](https://medium.com/@vishu54784/what-are-some-good-toy-problems-that-can-be-done-over-a-weekend-by-a-single-coder-in-data-science-6674c88fecff) 

So far, there are models to learn the moon dataset from sklearn and spirals from CS231n

## Running the Program

To be able to run the files, you'll need pytorch (Cuda or CPU, code is device agnostic), along with mathplotlib and sklearn installed.
Any of the learnX.py files are the files that can be executed to run a model to solve a problem. You can run them with python3:

```
python3 learnMoons.py
```

To adjust how efficient the ML model learns, or to change the input sample to be more spread, you can edit the constants at the top of the learn files.

### Limitations

Haven't tested with GPU since only using CPU pytorch, but, in theory, it should work (famous last words)
