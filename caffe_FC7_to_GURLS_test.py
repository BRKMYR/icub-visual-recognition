from __future__ import division
import numpy as np
#import matplotlib.pyplot as plt
#matplotlib inline
import yarp
import scipy.io

# Make sure that caffe is on the python path:
caffe_root = '/home/niklas/Downloads/caffe/'  # this file is expected to be in {caffe_root}/../Documents/Python\ Scripts/examples
import sys
sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, caffe_root + 'python/Scripts')

import caffe
import os

### IMAGENET OR ICUBWORLD28?
ORIGINAL = True

if ORIGINAL:
	dataset = "imagenet"
	model = "bvlc_reference_caffenet"
	trained = "bvlc_reference_caffenet.caffemodel"
	mean_file = "ilsvrc_2012_mean.npy"
	labels_file = 'data/ilsvrc12/synsets.txt'
else:
	dataset = "iCubWorld28"
	model = "bvlc_reference_caffenet_iCubWorld28"
	trained = "caffenet_train_iter_16160.caffemodel"
	mean_file = "iCubWorld28_mean.npy"
	labels_file = 'data/iCubWorld28/synsets.txt'

# Initialise YARP and Output port
yarp.Network.init()

# Sending to Matlab port
output_port = yarp.Port()
output_port.open("/FC7_port/out")
yarp.Network.connect("/FC7_port/out", "/matlab/sink")

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/' + model + '/deploy.prototxt',
                caffe_root + 'models/' + model + '/' + trained,
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
# STANDARD VERSION works with Imagenet_Mean binaryproto instead of iCubWorld28_mean.npy
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/' + mean_file).mean(1).mean(1)) # mean pixel

transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 50
net.blobs['data'].reshape(50,3,227,227)

# Reading filenames

# TESTING
filename_txt=open('/home/niklas/Desktop/textfiles/test_filenames.txt','r')
true_classNumbers_txt=open('/home/niklas/Desktop/textfiles/test_TRUE_classNumbers.txt','r')

# TRAINING
#filename_txt=open('/home/niklas/Desktop/textfiles/train_filenames.txt','r')
#true_classNumbers_txt=open('/home/niklas/Desktop/textfiles/train_TRUE_classNumbers.txt','r')


true_classNumbers=true_classNumbers_txt.readlines()
true_classNumbers_list=[]
for line in true_classNumbers:
	values=map(int, line.split())
	true_classNumbers_list.append(values)
true_classNumbers_txt.close()
true_classNumbers_list=true_classNumbers_list[0]

# load labels
imagenet_labels_filename = caffe_root + labels_file
try:
	labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
	print("Synsets.txt with imagenet labels not found...")

iteration=0
filename=''
numNotFound=0

# dimensions: TRAIN (25831), TEST (24884) pics
features=np.array(np.zeros([24884,4096]))

for line in filename_txt:
	filename=line[0:len(line)-1]
	#if iteration==1:
	#	filename='/home/niklas/Downloads/iCubWorld28/train/day3/sponge/sponge2/00004192.ppm'
	
	# Predict image
	try:
		net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(filename))
	except:
    		print(filename+" not found...")
		numNotFound+=1
		print("Total not found: "+str(numNotFound))
		continue

	out = net.forward()
	#print("Predicted class is #{}.".format(out['prob'][0]))	
	
	fc7_data = net.blobs['fc7'].data[0]

	# Create numpy array to receive the image and the YARP image wrapped around it
	#np_array = np.zeros((1, 4096), dtype=np.float)
	#np_array += fc7_data

	features[iteration]=fc7_data	
	iteration+=1
	if iteration % 100 == 0:
		print iteration
		

#yarp_image = yarp.ImageFloat()
#yarp_image.resize(1, 4096)
#yarp_image.setExternal(yarp_array, yarp_array.shape[1], yarp_array.shape[0])
scipy.io.savemat('/home/niklas/Downloads/GURLS/gurls/demo/test_data/fc7.mat', mdict={'fc7_data': features})
scipy.io.savemat('/home/niklas/Downloads/GURLS/gurls/demo/test_data/classnrs.mat', mdict={'classnrs': true_classNumbers_list})
