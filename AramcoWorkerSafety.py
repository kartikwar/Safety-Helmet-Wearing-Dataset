#! /usr/bin/env python

import os
import argparse
import json
import cv2
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
import numpy as np
import datetime

class AramcoWorkerSafety():

	###############################
	#   Set some parameter
	###############################
	def __init__(self):
		self.net_h, self.net_w = 416, 416 # a multiple of 32, the smaller the faster
		
		self.obj_thresh, self.nms_thresh = 0.01, 0.45

		dir_path = os.path.dirname(os.path.realpath(__file__))
		
		self.ppe_bbox_img_dir = 'output'
		
		if not os.path.exists(self.ppe_bbox_img_dir):
			os.makedirs(self.ppe_bbox_img_dir)

		###############################
		#   Load the model
		###############################



		self.config = {
			"model" : {
				"anchors":              [24,34, 46,84, 68,185, 116,286, 122,97, 171,180, 214,327, 326,193, 359,359],
				"labels":               ["hardhat", "nohardhat"]
			},

			"train": {

				"saved_weights_name":   os.path.join(dir_path , "HH_aramco.h5"),

			},


		}

		self.infer_model = load_model(self.config['train']['saved_weights_name'])


		temp = 0


		###############################
		#   Predict bounding boxes
		###############################

	def edge_predict(self, image):

		while True:

			batch_boxes = get_yolo_boxes(self.infer_model, [image], self.net_h, self.net_w, self.config['model']['anchors'], self.obj_thresh, self.nms_thresh)


			_,label_json = draw_boxes(image, batch_boxes[0], self.config['model']['labels'], self.obj_thresh)
			prefix = "PPE_"
			image_name = prefix + '.bmp'
			image_path = os.path.join(self.ppe_bbox_img_dir, image_name)
			cv2.imwrite(image_path, image)
			label_json['date'] = str(datetime.datetime.now())

			return label_json



if __name__ == "__main__":
	safety_obj = AramcoWorkerSafety()

	im = cv2.imread('000025.jpg')
	# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

	safety_obj.edge_predict(im)
