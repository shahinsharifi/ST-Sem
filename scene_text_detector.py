import matplotlib
matplotlib.use('Agg')
import numpy as np
import ntpath
import matplotlib.pyplot as plt
import math
from termcolor import colored
from util.nms import nms
from util.crop_image import crop_image
import cv2
import os
import sys
#os.environ['GLOG_minloglevel'] = '2'
sys.path.insert(0, '../../python')
import caffe


class SceneTextDetector:

    def __init__(self):
        caffe.set_device(0)
        caffe.set_mode_gpu()
        self.config = {
            'model_def': './models/icdar15/deploy.prototxt',
            'model_weights': './models/icdar15/model_icdar15.caffemodel',
            'det_visu_path': './temp/demo_det_result.jpg',
            'rec_visu_path': './temp/demo_rec_result.jpg',
            'det_save_dir': './temp/detection_result/',
            'rec_save_dir': './temp/recognition_result/',
            'crop_dir': './temp/crops/',
            'lexicon_path': './temp/icdar_generic_lexicon.txt',
            'use_lexcion': True,
            'input_height': 768,
            'input_width': 768,
            'overlap_threshold': 0.2,
            'det_score_threshold': 0.5,
            'f_score_threshold': 0.5,
            'visu_detection': True,
            'visu_recognition': True,
            'apply_recognition': True
        }
        self.net = self.prepare_network()


    def detect(self, inputPath):

        img = cv2.imread(os.path.join(inputPath), cv2.IMREAD_UNCHANGED)
        height = img.shape[0]
        width = img.shape[1]

        transformer = caffe.io.Transformer({'data': (1, 3, height, width)})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        transformer.set_raw_scale('data',
                                  249)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data',
                                     (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

        self.net.blobs['data'].reshape(1, 3, height, width)

        image = caffe.io.load_image(inputPath)
        transformed_image = transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        image_height, image_width, channels = image.shape
        detections = self.net.forward()['detection_out']
        # Parse the outputs.
        bboxes = self.extract_detections(detections, self.config['det_score_threshold'], image_height, image_width)
        # apply non-maximum suppression
        results = self.apply_quad_nms(bboxes, self.config['overlap_threshold'])
        self.save_and_visu(image, results, inputPath)


        # recognition
        if self.config['apply_recognition']:
            crop_image(inputPath, results, self.config['crop_dir'],6)
            crop_image(inputPath, results, self.config['crop_dir'], 3)

        del self.net

        return ntpath.basename(inputPath).split(".")[0]



    def prepare_network(self):

        net = caffe.Net(self.config['model_def'],  # defines the structure of the model
                        self.config['model_weights'],  # contains the trained weights
                        caffe.TEST)  # use test mode (e.g., don't perform dropout)
        return net



    def extract_detections(self, detections, det_score_threshold, image_height, image_width):
        det_conf = detections[0, 0, :, 2]
        det_x1 = detections[0, 0, :, 7]
        det_y1 = detections[0, 0, :, 8]
        det_x2 = detections[0, 0, :, 9]
        det_y2 = detections[0, 0, :, 10]
        det_x3 = detections[0, 0, :, 11]
        det_y3 = detections[0, 0, :, 12]
        det_x4 = detections[0, 0, :, 13]
        det_y4 = detections[0, 0, :, 14]
        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= det_score_threshold]
        top_conf = det_conf[top_indices]
        top_x1 = det_x1[top_indices]
        top_y1 = det_y1[top_indices]
        top_x2 = det_x2[top_indices]
        top_y2 = det_y2[top_indices]
        top_x3 = det_x3[top_indices]
        top_y3 = det_y3[top_indices]
        top_x4 = det_x4[top_indices]
        top_y4 = det_y4[top_indices]

        bboxes = []
        for i in xrange(top_conf.shape[0]):
            x1 = int(round(top_x1[i] * image_width))
            y1 = int(round(top_y1[i] * image_height))
            x2 = int(round(top_x2[i] * image_width))
            y2 = int(round(top_y2[i] * image_height))
            x3 = int(round(top_x3[i] * image_width))
            y3 = int(round(top_y3[i] * image_height))
            x4 = int(round(top_x4[i] * image_width))
            y4 = int(round(top_y4[i] * image_height))
            x1 = max(1, min(x1, image_width - 1))
            x2 = max(1, min(x2, image_width - 1))
            x3 = max(1, min(x3, image_width - 1))
            x4 = max(1, min(x4, image_width - 1))
            y1 = max(1, min(y1, image_height - 1))
            y2 = max(1, min(y2, image_height - 1))
            y3 = max(1, min(y3, image_height - 1))
            y4 = max(1, min(y4, image_height - 1))
            score = top_conf[i]
            bbox = [x1, y1, x2, y2, x3, y3, x4, y4, score]
            bboxes.append(bbox)
        return bboxes



    def apply_quad_nms(self, bboxes, overlap_threshold):
        dt_lines = sorted(bboxes, key=lambda x: -float(x[8]))
        nms_flag = nms(dt_lines, overlap_threshold)
        results = []
        for k, dt in enumerate(dt_lines):
            if nms_flag[k]:
                if dt not in results:
                    results.append(dt)
        return results



    def save_and_visu(self, image, results, inputPath):

        image_name = ntpath.basename(inputPath)
        det_save_path = os.path.join(self.config['det_save_dir'], image_name.split('.')[0] + '.txt')
        det_fid = open(det_save_path, 'wt')
        if self.config['visu_detection']:
            # visulization
            plt.clf()
            plt.imshow(image)
            currentAxis = plt.gca()
        for result in results:
            score = result[-1]
            x1 = result[0]
            y1 = result[1]
            x2 = result[2]
            y2 = result[3]
            x3 = result[4]
            y3 = result[5]
            x4 = result[6]
            y4 = result[7]
            result_str = str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(x3) + ',' + str(
                y3) + ',' + str(x4) + ',' + str(y4) + ',' + str(score) + '\r\n'
            det_fid.write(result_str)
            if self.config['visu_detection']:
                quad = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                color_quad = 'r'
                currentAxis.add_patch(plt.Polygon(quad, fill=False, edgecolor=color_quad, linewidth=2))

        det_fid.close()
        if self.config['visu_detection']:
            plt.axis('off')
            plt.savefig(self.config['det_visu_path'], dpi=900)



    def visu_rec_results(self, image, rec_save_dir, f_score_threshold):
        image_name = self.config['image_name']
        result_file_path = os.path.join(rec_save_dir, image_name.split('.')[0] + '.txt')
        rec_result_fid = open(result_file_path, 'r')
        plt.clf()
        plt.imshow(image)
        currentAxis = plt.gca()
        for line in rec_result_fid.readlines():
            line = line.strip()
            x1 = int(line.split(',')[0])
            y1 = int(line.split(',')[1])
            x2 = int(line.split(',')[2])
            y2 = int(line.split(',')[3])
            x3 = int(line.split(',')[4])
            y3 = int(line.split(',')[5])
            x4 = int(line.split(',')[6])
            y4 = int(line.split(',')[7])
            det_score = float(line.split(',')[8])
            rec_score = float(line.split(',')[10])
            rec_str = line.split(',')[9]
            f_score = 2 * math.exp(det_score) * math.exp(rec_score) / (math.exp(det_score) + math.exp(rec_score))
            print(f_score)
            if f_score > f_score_threshold:
                quad = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                color_quad = 'r'
                currentAxis.add_patch(plt.Polygon(quad, fill=False, edgecolor=color_quad, linewidth=2))
                currentAxis.text(x1, y1, rec_str, fontsize=5)

        rec_result_fid.close()
        plt.axis('off')
        plt.savefig(self.config['rec_visu_path'], dpi=900)



    def releaseMemory(self):
        del self.net
