import argparse
import json
import logging

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image


def parse_args():
    '''create parser'''
    parser = argparse.ArgumentParser(description='Find a flower type')

    # Input Command line parameter. Reads input image path.
    parser.add_argument('input_image', action='store',
                        type=str, help='input image path')
    parser.add_argument('model', action='store',
                        default='./ryn_model.h5', type=str, help='model for predict')
    parser.add_argument('--top_k', dest="top_k", action='store', default=5,
                        type=int, help='number of top predictions')
    parser.add_argument('--category_names', action='store', default='label_map.json',
                        type=str, dest="category_names", help='category names for prediction labels')

    return parser.parse_args()


def load_model(model_path):
    '''load keras model'''
    loaded_model = tf.keras.models.load_model(
        model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)

    return loaded_model


def load_class_names(label_map):
    '''load class names for match with labels'''
    with open(label_map, 'r') as f:
        return json.load(f)


def process_image(image):
    '''normalize image'''
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image


def predict(image_path, model, class_names, top_k):
    ''' Predict topK classes with highest probability.'''
    # convert type
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)

    # predict
    prob_predicts = model.predict(image)
    prob_predicts = prob_predicts[0].tolist()

    # find top k
    prob_and_class = [(prob, index)
                      for index, prob in enumerate(prob_predicts)]
    prob_and_class.sort(reverse=True)
    prob_and_class_names = [(prob, class_names[str(index+1)])
                            for prob, index in prob_and_class]
    return prob_and_class_names[:top_k]


def main():
    in_args = parse_args()

    img_path = in_args.input_image
    model = load_model(in_args.model)
    class_names = load_class_names(in_args.category_names)
    top_k = in_args.top_k

    print('\nThis is top {} probability of this flower tend to be :\n'.format(
        top_k), predict(img_path, model, class_names, top_k))


if __name__ == "__main__":
    main()
