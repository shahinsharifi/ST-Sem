import os
import sys
from termcolor import colored
import numpy as np
sys.path.insert(0, '../../python')
os.environ['GLOG_minloglevel'] = '2'
import caffe

class SceneRecognizer:


    def __init__(self):
        caffe.set_device(0)  # if we have multiple GPUs, pick the first one
        caffe.set_mode_gpu()
        self.prepare_network()


    def prepare_network(self):

        model_def = 'models/resnet152/resnet-152-torch-places365.prototxt'
        model_weights = 'models/resnet152/resnet-152-torch-places365.caffemodel'

        self.net = caffe.Net(model_def,  # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)  # use test mode (e.g., don't perform dropout)

        # create transformer for the input called 'data'
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})

        self.transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
        self.transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
        self.transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR


    def recognize(self, input):

      image = caffe.io.load_image(input)

      labels_file = 'labels/categories_places365.txt'
      labels = np.loadtxt(labels_file, str, delimiter='\t')

      sub_labels_file = 'labels/categories_places41.txt'
      sub_labels = np.loadtxt(sub_labels_file, str, delimiter='\t')

      self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)

      # perform classification
      self.net.forward()


      # obtain the output probabilities
      output_prob = self.net.blobs[self.net.outputs[0]].data[0]

      # sort top five predictions from softmax output
      top_inds = output_prob.argsort()

      del self.net

      return top_inds, output_prob, labels, sub_labels


    def releaseMemory(self):
        del self.net



