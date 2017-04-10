#!/usr/bin/env python

import os
import caffe
import numpy as np
from PIL import Image

	
def main():

	#network init
	caffe.set_device(0)
	caffe.set_mode_gpu()
	model_file = "../tiny_transform_net_deploy.prototxt"
	pretrained_file = "../tiny_transform_net_style.caffemodel"

	null_fds = os.open(os.devnull, os.O_RDWR)
	out_orig = os.dup(2)
	os.dup2(null_fds, 2)
	net = caffe.Net(model_file, pretrained_file, caffe.TEST)
	os.dup2(out_orig, 2)
	os.close(null_fds)
	print "init fine!"

	#input processing
	transformer = caffe.io.Transformer({"content_data": net.blobs["content_data"].data.shape})
	transformer.set_channel_swap("content_data", (2,1,0))
	transformer.set_transpose("content_data", (2,0,1))
	transformer.set_raw_scale("content_data", 1)

	input_img = caffe.io.load_image("../input.jpg")
	new_dims = (1, input_img.shape[2]) + input_img.shape[:2]
	net.blobs["content_data"].reshape(*new_dims)
	transformer.inputs["content_data"] = new_dims
	net_in = transformer.preprocess("content_data", input_img)
	net.blobs["content_data"].data[0] = net_in
	
	#forward
	net.forward()
	print "forward fine!"

	#output processing
	arr = net.blobs["conv4"].data[0]
	arr = arr[::-1]
	arr = arr.transpose((1, 2, 0))
	img=Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
	width, height = img.size
	bbox = (10,10,width-10,height-10)
	output_img = img.crop(bbox)   
	output_img.save('../result.jpg')



if __name__ == "__main__":     
	main()
