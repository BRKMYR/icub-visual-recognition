from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '/home/niklas/Downloads/caffe/'  # this file is expected to be in {caffe_root}/../Documents/Python\ Scripts/examples
import sys
sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, caffe_root + 'python/Scripts')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os
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

# Reading filenames
filename_txt=open('/home/niklas/Desktop/textfiles/test_filenames.txt','r')
true_classNumbers_txt=open('/home/niklas/Desktop/textfiles/test_TRUE_classNumbers.txt','r')
true_classNumbers=true_classNumbers_txt.readlines()
true_classNumbers_list=[]
for line in true_classNumbers:
	values=map(int, line.split())
	true_classNumbers_list.append(values)
true_classNumbers_txt.close()
true_classNumbers_list=true_classNumbers_list[0]
predicted_classNumbers_list=[]

# load labels
imagenet_labels_filename = caffe_root + 'data/iCubWorld28/synsets.txt'
try:
	labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
	print("Synsets.txt with iCubWorld28 labels not found...")

# Saving predicted classes into own '.txt'-Files
test_predictedNumbers_txt=open('/home/niklas/Desktop/textfiles/test_predictedNumbers.txt','w')

iteration=0
numNotFound=0
filename=''
for line in filename_txt:
	filename=line[0:len(line)-1]
	#filename='train/day3/sponge/sponge2/00004192.ppm\n'
	
	# Predict image
	try:
		net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(filename))
	except:
    		print(filename+" not found...")
		numNotFound+=1
		print("Total not found: "+str(numNotFound))
		continue

	out = net.forward()
	print("Predicted class is #{}.".format(out['prob'][0].argmax()))

	# plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
	# sort top k predictions from softmax output
	# top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]	
	# print labels[top_k]

	test_predictedNumbers_txt.write(str(format(out['prob'][0].argmax()))+' ')
	predicted_classNumbers_list.append(format(out['prob'][0].argmax()))

test_predictedNumbers_txt.close()

# Compare with true labels (the same as compareTRUEandTESTClassNumbers.py
#test_classNumbers_txt=open('/home/niklas/Desktop/textfiles/test_predictedNumbers.txt','r')
#test_classNumbers=test_classNumbers_txt.readlines()
#predicted_classNumbers_list=[]
#for line in test_classNumbers:
#	values=map(int, line.split())
#	test_classNumbers_list.append(values)
#test_classNumbers_txt.close()
#test_classNumbers_list=test_classNumbers_list[0]

anzahlVector=[0]*28
detectedVector=[0]*28
classAccuracyVector=[0]*28

correctPredictions=0
overallPerformance=0.00
for num in range(0,len(predicted_classNumbers_list)-1):
	#if true_classNumbers[num]==test_predictedNumbers[num]:
	anzahlVector[true_classNumbers_list[num]]+=1
	if true_classNumbers_list[num]==predicted_classNumbers_list[num]:
		correctPredictions+=1
		detectedVector[true_classNumbers_list[num]]+=1

overallPerformance=correctPredictions/len(test_predictedNumbers)
print ("Overall Performance is: "+str(overallPerformance))
print ("\n")

for i in range(0,28):
	classAccuracyVector[i]=detectedVector[i]/anzahlVector[i]
	print ("Accuracy for class "+str(i+1)+"is: "+str(classAccuracyVector[i])+"\n")
	
print ("Detected objects: ")
print detectedVector
print ("\n")
print ("Total number of objects: ")
print anzahlVector
print ("\n")
