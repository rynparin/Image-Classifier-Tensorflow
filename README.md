# Image-Classifier-Tensorflow
In this project I apply neural networks and transfer learning techniques to classify flower images and build command line app

##
Intro to Machine Learning - TensorFlow Project

Project code for Udacity's Intro to Machine Learning with TensorFlow Nanodegree program.

## Data
Oxford of 102 flower Image

## Contents
- `Project_Image_Classifier_Project.ipynb`: Jupyter notebook showing all the steps to create a classifier model
- `Project_Image_Classifier_Project.html`: HTML version of the jupyter notebook
- `ryn_model.h5`: Classifier model
- `predict.py`: Python application that uses our classifier model to predict flower type

## Command line app
In `predict.py` I build a command line app that can use to predict image which you can choose image, model, class_name

### Usage
```
> predict.py [image_path] [model_path] --top_k [number_of_classes] --category_names [json_file_of_class_names]
```
Sample:
```
> predict.py ./myflower.jpg my_model --top_k 3
> predict.py ./myflower.jpg my_model --category_names map.json
```
