# This script receives an image from YARP and predicts its class with a prior trained Caffe model. 
from __future__ import division
import numpy as np
import yarp
#import matplotlib.pyplot as plt
import sys
import caffe
import os
#import matplotlib.pylab
 
# Initialise YARP
yarp.Network.init()
 
# Create a port and connect it to the iCub simulator virtual camera
input_port = yarp.Port()
input_port.open("/python-image-port")
yarp.Network.connect("/cropped_image/out", "/python-image-port")
 
# Create numpy array to receive the image and the YARP image wrapped around it
img_array = np.zeros((256, 256, 3), dtype=np.uint8)
yarp_image = yarp.ImageRgb()
yarp_image.resize(256, 256)
yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])

# Make sure that caffe is on the python path:
caffe_root = '/home/niklas/Downloads/caffe/'  # this file is expected to be in {caffe_root}/../Documents/Python\ Scripts/examples
sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, caffe_root + 'python/Scripts')

if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/caffenet_train_iter_16160.caffemodel'):
    print("CaffeNet model trained with iCubWorld28 dataset not found...")

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/caffenet_train_iter_16160.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
# STANDARD VERSION works with Imagenet_Mean binaryproto instead of iCubWorld28_mean.npy
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/iCubWorld28_mean.npy').mean(1).mean(1)) # mean pixel

transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 50
net.blobs['data'].reshape(50,3,227,227)

while True:
    # Read the data from the port into the image
    input_port.read(yarp_image)
    
    # Predict image
    try:
         net.blobs['data'].data[...] = transformer.preprocess('data', yarp_image)
    except:
         print("yarp_image not received...")
         continue
    out = net.forward()
    print("Predicted class is #{}.".format(out['prob'][0].argmax()))

	# plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
	# sort top k predictions from softmax output
	# top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]	
	# print labels[top_k]
