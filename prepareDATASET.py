# Custom data preparation script for iCubWorld28 dataset which prepares "train.txt" and "test.txt" for Caffe Deep Learning
# The order of folders is "hardcoded", i.e. specific for the used computer (this is maybe different for other computers)

# Further label explanation for iCubWorld28 (numbers in column are shape_num/material_num/affordance_num):
# classnr: 0-27 (class)
# #categorynr: 0-6 (4 classes equal one category)
# attribute_shape: 'long' (1), 'round' (2), 'rectangular'(3) - (describes shape)
# attribute_material: 'ceramic' (1), 'plastic' (2), 'furry' (3), 'clear' (4), 'wet' (5) (describes material/texture)
# affordance: 'hold' (1), 'drink' (2), 'eat' (3), 'clean' (4), 'open' (5), 'cut' (6) (describes possible actions with object)

###########################################################################################################################
## SET HERE, how Caffe shall be trained ('10_objects', 'attribute_shape', 'attribute_material', 'affordance'):
TRAIN_MODE = '10_objects'

###########################################################################################################################

import os
from os import listdir
from os.path import isfile, join
import csv
import numpy as np

# TRAIN-Part 1
folder = '/home/icub-niklas/Desktop/DATASET/train'
mylist = []
classes=[]
for path, subdirs, files in os.walk(folder):
	for subdir in subdirs:
		classes.append(subdir)
	for name in files:        	
		mylist.append(os.path.join(path, name))

# SYNSETS.TXT
synsets_txt = open('/home/icub-niklas/Desktop/textfiles/synsets.txt', 'w')
ctr=0
for temp_class in classes:
	#print temp_class
	if temp_class!='day1' and temp_class!='day2' and temp_class!='day3' and temp_class!='day4' and ctr<28:            	
		synsets_txt.write(temp_class+'\n')
		ctr=ctr+1
synsets_txt.close()

# TRAIN-Part 2
classnr = 0
attribute_shape = ''
attribute_material =''
affordance = ''

txt = open('/home/icub-niklas/Desktop/textfiles/train.txt', 'w')
filename_txt=open('/home/icub-niklas/Desktop/textfiles/train_filenames.txt', 'w')
classNumbers_txt= open('/home/icub-niklas/Desktop/textfiles/train_classNumbers.txt', 'w')
attribute_shape_txt= open('/home/icub-niklas/Desktop/textfiles/train_attribute_shape.txt', 'w')
attribute_material_txt= open('/home/icub-niklas/Desktop/textfiles/train_attribute_material.txt', 'w')
affordance_txt= open('/home/icub-niklas/Desktop/textfiles/train_affordances.txt', 'w')

#attribute_shape_=np.array(np.zeros([25831,1]))
#attribute_material_=np.array(np.zeros([25831,1]))
#affordance_=np.array(np.zeros([25831,1]))

iteration = 0

for list in mylist:
	#list = mylist.pop(0)

	dir, filename = os.path.basename(os.path.dirname(list)), os.path.basename(list)
	# Classnr according to line in synsets.txt

	if dir=='Banana':
	     	classnr=0
		#categorynr = 1
		attribute_shape = 'long'
		attribute_material ='plastic'
		affordance = 'clean'	
	elif dir=='LaundryDetergent':
	     	classnr=1
		#categorynr = 1
		attribute_shape = 'long'
		attribute_material ='plastic'
		affordance = 'clean'	
	elif dir=='Apple':
	     	classnr=2
		#categorynr = 1
		attribute_shape = 'long'
		attribute_material ='plastic'
		affordance = 'clean'	
	elif dir=='Smartphone':
	     	classnr=3
		#categorynr = 3
		attribute_shape = 'round'
		attribute_material ='ceramic'
		affordance = 'drink'
	elif dir=='Sponge':
	     	classnr=4
		#categorynr = 6
		attribute_shape = 'rectangular'
		attribute_material ='wet'
		affordance = 'clean'
	elif dir=='Orange':
	     	classnr=5
		#categorynr = 1
		attribute_shape = 'long'
		attribute_material ='plastic'
		affordance = 'clean'
	elif dir=='Plate':
	     	classnr=6
		#categorynr = 0
		attribute_shape = 'round'
		attribute_material ='ceramic'
		affordance = 'eat'
	elif dir=='Cup':
	     	classnr=7
		#categorynr = 1
		attribute_shape = 'long'
		attribute_material ='plastic'
		affordance = 'clean'	
	elif dir=='Laptop':
	     	classnr=8
		#categorynr = 1
		attribute_shape = 'long'
		attribute_material ='plastic'
		affordance = 'clean'	

	filename_txt.write(list+'\n')
	classNumbers_txt.write(str(classnr)+' ')
	attribute_shape_txt.write(attribute_shape + '\n')
	attribute_material_txt.write(attribute_material + '\n')	
	affordance_txt.write(affordance + '\n')
	
	# Another matching..: attributes and affordances need to be numeric!
	# Shape:	
	if attribute_shape == 'long':
		shape_num = 1
	elif attribute_shape == 'round':
		shape_num = 2
	elif attribute_shape == 'rectangular':
		shape_num = 3
	else:
		print "Error: No attribute shape is set! (TRAINING)"

	# Material
	if attribute_material == 'ceramic':
		material_num = 1
	elif attribute_material == 'plastic':
		material_num = 2
	elif attribute_material == 'furry':
		material_num = 3
	elif attribute_material == 'clear':
		material_num = 4
	elif attribute_material == 'wet':
		material_num = 5
	else:
		print "Error: No attribute material is set! (TRAINING)"
	
	# Affordances
	if affordance == 'hold':
		affordance_num = 1
	elif affordance == 'drink':
		affordance_num = 2
	elif affordance == 'eat':
		affordance_num = 3
	elif affordance == 'clean':
		affordance_num = 4
	elif affordance == 'open':
		affordance_num = 5
	elif affordance == 'cut':
		affordance_num = 6
	else:
		print "Error: No affordance is set! (TRAINING)"
	
	#attribute_shape_[iteration] = shape_num
	#attribute_material_[iteration] = material_num
	#affordance_[iteration] = affordance_num

	selected_item = ''
	if TRAIN_MODE == '10_objects':
		selected_item=str(classnr)
	elif TRAIN_MODE == 'attribute_shape':
		selected_item=str(shape_num)
	elif TRAIN_MODE == 'attribute_material':
		selected_item=str(material_num)
	elif TRAIN_MODE == 'affordance':
		selected_item=str(affordance_num)
	else:
		print "Error: TRAIN_MODE is set incorrectly!\n"

	txt.write(list+' '+selected_item+'\n')

	iteration += 1

#np.savetxt('/home/icub-niklas/Desktop/textfiles/csv-Files/train_attribute_shape.csv', attribute_shape_, fmt='%d',  delimiter=',')
#np.savetxt('/home/icub-niklas/Desktop/textfiles/csv-Files/train_attribute_material.csv', attribute_material_, fmt='%d',  delimiter=',') 
#np.savetxt('/home/icub-niklas/Desktop/textfiles/csv-Files/train_affordances.csv', affordance_, fmt='%d',  delimiter=',') 

# Additional filenames would be...
#('/home/icub-niklas/Desktop/textfiles/csv-Files/train_classNumbers.csv', 'wb')
#('/home/icub-niklas/Desktop/textfiles/csv-Files/train_TRUE_categoryNumbers.csv', 'wb')

filename_txt.close()
txt.close()
classNumbers_txt.close()
attribute_shape_txt.close()
attribute_material_txt.close()
affordance_txt.close()

# TEST-Folder
folder = '/home/icub-niklas/Desktop/DATASET/test'
mylist = []
classes=[]

#abc_txt = open('/home/icub-niklas/Desktop/textfiles/abc.txt', 'w')

for path, subdirs, files in os.walk(folder):
	for name in files:        	
		mylist.append(os.path.join(path, name))
		#abc_txt.write(os.path.join(path,name)+'\n')
	for subdir in subdirs:
		classes.append(subdir)

# TESTSYNSETS.TXT
TEST_synsets_txt = open('/home/icub-niklas/Desktop/textfiles/TEST_synsets.txt', 'w')
TEST_synset_words_txt= open('/home/icub-niklas/Desktop/textfiles/TEST_detsynset_words.txt', 'w')
ctr=0
for temp_class in classes:
	if temp_class!='day1' and temp_class!='day2' and temp_class!='day3' and temp_class!='day4' and ctr<28:            	
		TEST_synsets_txt.write(temp_class+'\n')
		TEST_synset_words_txt.write(temp_class+' '+temp_class+'\n')
		ctr=ctr+1
TEST_synsets_txt.close()
TEST_synset_words_txt.close()

classnr=0
attribute_shape = ''
attribute_material =''
affordance = ''

txt = open('/home/icub-niklas/Desktop/textfiles/test.txt', 'w')
filename_txt= open('/home/icub-niklas/Desktop/textfiles/test_filenames.txt', 'w')
classNumbers_txt= open('/home/icub-niklas/Desktop/textfiles/test_classNumbers.txt', 'w')
attribute_shape_txt= open('/home/icub-niklas/Desktop/textfiles/test_attribute_shape.txt', 'w')
attribute_material_txt= open('/home/icub-niklas/Desktop/textfiles/test_attribute_material.txt', 'w')
affordance_txt= open('/home/icub-niklas/Desktop/textfiles/test_affordances.txt', 'w')


#for list in mylist:
#	dir2, filename = os.path.basename(os.path.dirname(list)), os.path.basename(list)
#	txt.write(dir2 + '/' + filename + ' '+str(classnr)+'\n')
#txt.close()

#attribute_shape_=np.array(np.zeros([24884,1]))
#attribute_material_=np.array(np.zeros([24884,1]))
#affordance_=np.array(np.zeros([24884,1]))

iteration = 0

for list in mylist:
	#list = mylist.pop(0)

	dir, filename = os.path.basename(os.path.dirname(list)), os.path.basename(list)
	
	if dir=='Banana':
	     	classnr=0
		#categorynr = 1
		attribute_shape = 'long'
		attribute_material ='plastic'
		affordance = 'clean'	
	elif dir=='LaundryDetergent':
	     	classnr=1
		#categorynr = 1
		attribute_shape = 'long'
		attribute_material ='plastic'
		affordance = 'clean'	
	elif dir=='Apple':
	     	classnr=2
		#categorynr = 1
		attribute_shape = 'long'
		attribute_material ='plastic'
		affordance = 'clean'	
	elif dir=='Smartphone':
	     	classnr=3
		#categorynr = 3
		attribute_shape = 'round'
		attribute_material ='ceramic'
		affordance = 'drink'
	elif dir=='Sponge':
	     	classnr=4
		#categorynr = 6
		attribute_shape = 'rectangular'
		attribute_material ='wet'
		affordance = 'clean'
	elif dir=='Orange':
	     	classnr=5
		#categorynr = 1
		attribute_shape = 'long'
		attribute_material ='plastic'
		affordance = 'clean'
	elif dir=='Plate':
	     	classnr=6
		#categorynr = 0
		attribute_shape = 'round'
		attribute_material ='ceramic'
		affordance = 'eat'
	elif dir=='Cup':
	     	classnr=7
		#categorynr = 1
		attribute_shape = 'long'
		attribute_material ='plastic'
		affordance = 'clean'	
	elif dir=='Laptop':
	     	classnr=8
		#categorynr = 1
		attribute_shape = 'long'
		attribute_material ='plastic'
		affordance = 'clean'

	classNumbers_txt.write(str(classnr)+' ')
	attribute_shape_txt.write(attribute_shape + '\n')
	attribute_material_txt.write(attribute_material + '\n')	
	affordance_txt.write(affordance + '\n')

	# Another matching..: attributes and affordances need to be numeric!
	# Shape:	
	if attribute_shape == 'long':
		shape_num = 1
	elif attribute_shape == 'round':
		shape_num = 2
	elif attribute_shape == 'rectangular':
		shape_num = 3
	else:
		print "Error: No attribute shape is set! (TESTING)"

	# Material
	if attribute_material == 'ceramic':
		material_num = 1
	elif attribute_material == 'plastic':
		material_num = 2
	elif attribute_material == 'furry':
		material_num = 3
	elif attribute_material == 'clear':
		material_num = 4
	elif attribute_material == 'wet':
		material_num = 5
	else:
		print "Error: No attribute material is set! (TESTING)"
	
	# Affordances
	if affordance == 'hold':
		affordance_num = 1
	elif affordance == 'drink':
		affordance_num = 2
	elif affordance == 'eat':
		affordance_num = 3
	elif affordance == 'clean':
		affordance_num = 4
	elif affordance == 'open':
		affordance_num = 5
	elif affordance == 'cut':
		affordance_num = 6
	else:
		print "Error: No affordance is set! (TESTING)"
	
	#attribute_shape_[iteration] = shape_num
	#attribute_material_[iteration] = material_num
	#affordance_[iteration] = affordance_num

	selected_item = ''
	if TRAIN_MODE == '10_objects':
		selected_item=str(classnr)
	elif TRAIN_MODE == 'attribute_shape':
		selected_item=str(shape_num)
	elif TRAIN_MODE == 'attribute_material':
		selected_item=str(material_num)
	elif TRAIN_MODE == 'affordance':
		selected_item=str(affordance_num)
	else:
		print "Error: TRAIN_MODE is set incorrectly!\n"

	txt.write(list+' '+selected_item+'\n')

	iteration += 1

#np.savetxt('/home/icub-niklas/Desktop/textfiles/csv-Files/test_attribute_shape.csv', attribute_shape_, fmt='%d', delimiter=',')
#np.savetxt('/home/icub-niklas/Desktop/textfiles/csv-Files/test_attribute_material.csv', attribute_material_, fmt='%d',  delimiter=',') 
#np.savetxt('/home/icub-niklas/Desktop/textfiles/csv-Files/test_affordances.csv', affordance_, fmt='%d',  delimiter=',') 

filename_txt.close()
txt.close()
classNumbers_txt.close()
attribute_shape_txt.close()
attribute_material_txt.close()
affordance_txt.close()
