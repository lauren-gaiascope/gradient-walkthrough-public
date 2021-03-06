from random import randint

plotting = True
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    print('Matplotlib not detected - images plotting not available')
    plotting = False

import numpy as np
from PIL import Image as pilimage
import requests
import tensorflow as tf
import argparse
import getpass

def get_image_from_drive(path):
    # Load the image
    image = pilimage.open(path)
    image = image.convert('L')
    image = np.resize(image, (28,28,1))
    image = np.array(image)
    image = image.reshape(28,28)
    return image


def get_random_image_from_dataset(x_test, y_test):
    image_index=randint(0, 9999)
    print('target class (from random test sample): %d' % y_test[image_index])
    return x_test[image_index]


def show_selected_image(image):
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    plt.tight_layout()
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def make_vector(image):
    vector = []
    for item in image.tolist():
        vector.extend(item)
    return vector


def make_prediction_request(image, prediction_url, auth):
    vector = make_vector(image)
    json = {
        "inputs": [vector]
    }
    response = requests.post(prediction_url, json=json, auth=auth)

    print('HTTP Response %s' % response.status_code)
    print(response.text)


def main():
    parser = argparse.ArgumentParser(description='Test MNIST Tensorflow Server')
    parser.add_argument('-u', '--url', help='Prediction HOST URL', default='http://127.0.0.1:8501/v1/models/mnist:predict')
    parser.add_argument('-p', '--path', help='Example image path')
    parser.add_argument('-U', '--username', help='Basic Auth username')
    parser.add_argument('-P', '--password', help='Basic Auth password')
    parser.add_argument('-i', '--iterations', type=int, help='Number of iterations; use -1 for forever')
    args = parser.parse_args()

    i = 1
    req_cnt = 0
    if args.iterations:
        i = args.iterations
        ploting = False
    auth = None
    if args.username:
       if args.password is None:
           args.password = getpass.getpass()
       auth = (args.username, args.password)

    # Load image from drive if specified, if not load example image from mnist dataset
    if args.path:
        image = get_image_from_drive(args.path)
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        image = get_random_image_from_dataset(x_test, y_test)

    if plotting:
        show_selected_image(image)

    while i != 0:
        if args.iterations:
            req_cnt += 1
            print('Iteration: %d' % req_cnt)
        make_prediction_request(image, args.url, auth)
        if i > 0:
            i -= 1
        if i != 0 and args.path is None:
            image = get_random_image_from_dataset(x_test, y_test)


if __name__ == '__main__':
    main()
