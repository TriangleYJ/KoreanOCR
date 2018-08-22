#!/usr/bin/env python

import argparse
import io
import os
import sys
import cv2

import tensorflow as tf

TRAINED_SIZE = 64

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default paths.
DEFAULT_LABEL_FILE = os.path.join(
    SCRIPT_PATH, '../labels/2350-common-hangul.txt'
)
DEFAULT_GRAPH_FILE = os.path.join(
    SCRIPT_PATH, '../saved-model/optimized_hangul_tensorflow.pb'
)

'read image using opencv'
def read_image(file):
    """Read an image file and convert it into a 1-D floating point array."""
    image = cv2.imread(file)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
    
    height, width = thresh.shape

    fx = float(TRAINED_SIZE) / float(height)

    resized = cv2.resize(thresh, None, fx=fx, fy=fx, interpolation=cv2.INTER_LINEAR)

    characters = []

    number_of_characters = int(width / TRAINED_SIZE)

    for i in range(number_of_characters):
        char_img = resized[:, 64*i:64*(i+1)]

        normalize = char_img / 255.0

        characters.append(normalize.reshape(-1))

    return characters


def recognize(args):
    """Recognize a sentence.

    This method will import the saved model from the given graph file, and will
    pass in the given image pixels as input for the recognition. The top
    five predictions will be printed.
    """
    labels = io.open(args.label_file,
                     'r', encoding='utf-8').read().splitlines()

    if not os.path.isfile(args.image):
        print('Error: Image %s not found.' % args.image)
        sys.exit(1)

    # Load graph and parse file.
    with tf.gfile.GFile(args.graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name='hangul-model',
            producer_op_list=None
        )

    # Get relevant nodes.
    x = graph.get_tensor_by_name('hangul-model/input:0')
    y = graph.get_tensor_by_name('hangul-model/output:0')
    keep_prob = graph.get_tensor_by_name('hangul-model/keep_prob:0')

    image_array = read_image(args.image)

    with tf.Session(graph=graph) as graph_sess:
        predictions = graph_sess.run(y, feed_dict={x: image_array,
                                                   keep_prob: 1.0})
        

    # Get the indices that would sort the array, then only get the indices that
    # correspond to the top 5 predictions.
    number_of_characters = predictions.shape[0]
    for i in range(number_of_characters):
        prediction = predictions[i]
        index = prediction.argsort()[::-1][:1][0]     
        label = labels[index]
        confidence = prediction[index]
        print('%s (confidence = %.5f)' % (label, confidence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str,
                        help='Image to pass to model for classification.')
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--graph-file', type=str, dest='graph_file',
                        default=DEFAULT_GRAPH_FILE,
                        help='The saved model graph file to use for '
                             'classification.')
    recognize(parser.parse_args())
