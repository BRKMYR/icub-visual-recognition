# This script receives an image from YARP and predicts its class with a prior trained Caffe model. 
from __future__ import division
import numpy as np
import yarp
#import matplotlib.pyplot as plt
import sys
import caffe
import os
#import matplotlib.pylab
import scipy
 
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

if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet_iCubWorld28/caffenet_train_iter_16160.caffemodel'):
    print("CaffeNet model trained with iCubWorld28 dataset not found...")

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet_iCubWorld28/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet_iCubWorld28/caffenet_train_iter_16160.caffemodel',
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

# load labels
imagenet_labels_filename = caffe_root + 'data/iCubWorld28/synsets.txt'
try:
	labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
	print("Synsets.txt with iCubWorld28 labels not found...")

while True:

    # Create numpy array to receive the image and the YARP image wrapped around it
    img_array = np.zeros((256, 256, 3), dtype=np.uint8)
    yarp_image = yarp.ImageRgb()
    yarp_image.resize(256, 256)
    yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
    
    # Read the data from the port into the image
    input_port.read(yarp_image)
    filename= "/home/niklas/last_yarp_image.jpeg"
    scipy.misc.toimage(img_array, cmin=0.0, cmax=255.0).save(filename)

    # Predict image
    try:
         #net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(filename))
	 net.blobs['data'].data[...] = transformer.preprocess('data', img_array)
    except:
         print("yarp_image not received...")
         continue
    out = net.forward()
    print("Predicted class is #{}.".format(out['prob'][0].argmax()))
    
    # sort top k predictions from softmax output
    top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    print labels[top_k]
