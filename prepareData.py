# Custom data preparation script for iCubWorld28 dataset which prepares "train.txt" and "test.txt" for Caffe Deep Learning
# The order of folders is "hardcoded", i.e. specific for the used computer (this maybe different for other computers)

import os
from os import listdir
from os.path import isfile, join

# TRAIN-Part 1
folder = '/home/niklas/Downloads/iCubWorld28/train'
mylist = []
classes=[]
for path, subdirs, files in os.walk(folder):
	for subdir in subdirs:
		classes.append(subdir)
	for name in files:        	
		mylist.append(os.path.join(path, name))

# SYNSETS.TXT
synsets_txt = open('/home/niklas/Desktop/textfiles/synsets.txt', 'w')
synset_words_txt= open('/home/niklas/Desktop/textfiles/detsynset_words.txt', 'w')
ctr=0
for temp_class in classes:
	#print temp_class
	if temp_class!='day1' and temp_class!='day2' and temp_class!='day3' and temp_class!='day4' and temp_class!='day4_manual_crop' and temp_class!='plate' and temp_class!='laundry-detergent' and temp_class!='sprayer' and temp_class!='cup' and temp_class!='soap' and temp_class!='dishwashing-detergent' and temp_class!='sponge' and ctr<28:            	
		synsets_txt.write(temp_class+'\n')
		synset_words_txt.write(temp_class+' '+temp_class+'\n')
		ctr=ctr+1
synsets_txt.close()
synset_words_txt.close()

# TRAIN-Part 2
classnr=0
daynr=1
tag=''
prefix=''
txt = open('/home/niklas/Desktop/textfiles/train.txt', 'w')
filename_txt=open('/home/niklas/Desktop/textfiles/train_filenames.txt', 'w')
for list in mylist:
	#list = mylist.pop(0)

	dir, filename = os.path.basename(os.path.dirname(list)), os.path.basename(list)
	# Classnr according to line in synsets.txt
	if dir=='plate4':
		#if classnr!=0:
		#	daynr=daynr+1	
	     	classnr=0
		prefix='plate'
	elif dir=='plate3':
	     	classnr=1
		prefix='plate'
	elif dir=='plate2':
	     	classnr=2
		prefix='plate'
	elif dir=='plate1':
	     	classnr=3
		prefix='plate'
	elif dir=='laundry-detergent3':
	     	classnr=4
		prefix='laundry-detergent'
	elif dir=='laundry-detergent2':
	     	classnr=5
		prefix='laundry-detergent'
	elif dir=='laundry-detergent4':
	     	classnr=6
		prefix='laundry-detergent'
	elif dir=='laundry-detergent1':
	     	classnr=7
		prefix='laundry-detergent'
	elif dir=='sprayer4':
	     	classnr=8
		prefix='sprayer'
	elif dir=='sprayer1':
	     	classnr=9
		prefix='sprayer'
	elif dir=='sprayer3':
	     	classnr=10
		prefix='sprayer'
	elif dir=='sprayer2':
	     	classnr=11
		prefix='sprayer'
	elif dir=='cup3':
	     	classnr=12
		prefix='cup'
	elif dir=='cup1':
	     	classnr=13
		prefix='cup'
	elif dir=='cup2':
	     	classnr=14
		prefix='cup'
	elif dir=='cup4':
	     	classnr=15
		prefix='cup'
	elif dir=='soap3':
	     	classnr=16
		prefix='soap'
	elif dir=='soap2':
	     	classnr=17
		prefix='soap'
	elif dir=='soap4':
	     	classnr=18
		prefix='soap'
	elif dir=='soap1':
	     	classnr=19
		prefix='soap'
	elif dir=='dishwashing-detergent1':
	     	classnr=20
		prefix='dishwashing-detergent'
	elif dir=='dishwashing-detergent3':
	     	classnr=21
		prefix='dishwashing-detergent'
	elif dir=='dishwashing-detergent2':
	     	classnr=22
		prefix='dishwashing-detergent'
	elif dir=='dishwashing-detergent4':
	     	classnr=23
		prefix='dishwashing-detergent'
	elif dir=='sponge2':
	     	classnr=24
		prefix='sponge'
	elif dir=='sponge1':
	     	classnr=25
		prefix='sponge'
	elif dir=='sponge4':
	     	classnr=26
		prefix='sponge'
	elif dir=='sponge3':
	     	classnr=27
		prefix='sponge'

	# Further file preparation for right filename-matching
	# Commented days are version1
	if daynr==1:
		#tag='day1'
		tag='day3'
	elif daynr==2:
		#tag='day3'
		tag='day1'
	elif daynr==3:
		#tag='day2'
		tag='day2'
	elif daynr==4:
		#tag='day4'
		tag='day4_manual_crop'
	elif daynr==5:
		#tag='day4_manual_crop'
		tag='day4'

	#txt.write(tag+'/'+prefix+'/'+dir + '/' + filename + ' '+str(classnr)+'\n')
	txt.write(list+' '+str(classnr)+'\n')
	#filename_txt.write(tag+'/'+prefix+'/'+dir + '/' + filename+'\n')
	filename_txt.write(list+'\n')


filename_txt.close()
txt.close()

# TEST-Folder
folder = '/home/niklas/Downloads/iCubWorld28/test'
mylist = []
classes=[]

#abc_txt = open('/home/niklas/Desktop/textfiles/abc.txt', 'w')

for path, subdirs, files in os.walk(folder):
	for name in files:        	
		mylist.append(os.path.join(path, name))
		#abc_txt.write(os.path.join(path,name)+'\n')
	for subdir in subdirs:
		classes.append(subdir)

# TESTSYNSETS.TXT
TEST_synsets_txt = open('/home/niklas/Desktop/textfiles/TEST_synsets.txt', 'w')
TEST_synset_words_txt= open('/home/niklas/Desktop/textfiles/TEST_detsynset_words.txt', 'w')
ctr=0
for temp_class in classes:
	if temp_class!='day1' and temp_class!='day2' and temp_class!='day3' and temp_class!='day4' and temp_class!='day4_manual_crop' and temp_class!='plate' and temp_class!='laundry-detergent' and temp_class!='sprayer' and temp_class!='cup' and temp_class!='soap' and temp_class!='dishwashing-detergent' and temp_class!='sponge' and ctr<28:            	
		TEST_synsets_txt.write(temp_class+'\n')
		TEST_synset_words_txt.write(temp_class+' '+temp_class+'\n')
		ctr=ctr+1
TEST_synsets_txt.close()
TEST_synset_words_txt.close()

classnr=0
daynr=1
tag=''
prefix=''
txt = open('/home/niklas/Desktop/textfiles/test.txt', 'w')
filename_txt= open('/home/niklas/Desktop/textfiles/test_filenames.txt', 'w')
classNumbers_txt= open('/home/niklas/Desktop/textfiles/test_TRUE_classNumbers.txt', 'w')
#for list in mylist:
#	dir2, filename = os.path.basename(os.path.dirname(list)), os.path.basename(list)
#	txt.write(dir2 + '/' + filename + ' '+str(classnr)+'\n')
#txt.close()

for list in mylist:
	#list = mylist.pop(0)

	dir, filename = os.path.basename(os.path.dirname(list)), os.path.basename(list)
	
	if dir=='plate4':
		#if classnr!=0:
		#	daynr=daynr+1
	     	classnr=0
		prefix='plate'
	elif dir=='plate3':
	     	classnr=1
		prefix='plate'
	elif dir=='plate2':
	     	classnr=2
		prefix='plate'
	elif dir=='plate1':
	     	classnr=3
		prefix='plate'
	elif dir=='laundry-detergent3':
	     	classnr=4
		prefix='laundry-detergent'
	elif dir=='laundry-detergent2':
	     	classnr=5
		prefix='laundry-detergent'
	elif dir=='laundry-detergent4':
	     	classnr=6
		prefix='laundry-detergent'
	elif dir=='laundry-detergent1':
	     	classnr=7
		prefix='laundry-detergent'
	elif dir=='sprayer4':
	     	classnr=8
		prefix='sprayer'
	elif dir=='sprayer1':
	     	classnr=9
		prefix='sprayer'
	elif dir=='sprayer3':
	     	classnr=10
		prefix='sprayer'
	elif dir=='sprayer2':
	     	classnr=11
		prefix='sprayer'
	elif dir=='cup3':
	     	classnr=12
		prefix='cup'
	elif dir=='cup1':
	     	classnr=13
		prefix='cup'
	elif dir=='cup2':
	     	classnr=14
		prefix='cup'
	elif dir=='cup4':
	     	classnr=15
		prefix='cup'
	elif dir=='soap3':
	     	classnr=16
		prefix='soap'
	elif dir=='soap2':
	     	classnr=17
		prefix='soap'
	elif dir=='soap4':
	     	classnr=18
		prefix='soap'
	elif dir=='soap1':
	     	classnr=19
		prefix='soap'
	elif dir=='dishwashing-detergent1':
	     	classnr=20
		prefix='dishwashing-detergent'
	elif dir=='dishwashing-detergent3':
	     	classnr=21
		prefix='dishwashing-detergent'
	elif dir=='dishwashing-detergent2':
	     	classnr=22
		prefix='dishwashing-detergent'
	elif dir=='dishwashing-detergent4':
	     	classnr=23
		prefix='dishwashing-detergent'
	elif dir=='sponge2':
	     	classnr=24
		prefix='sponge'
	elif dir=='sponge1':
	     	classnr=25
		prefix='sponge'
	elif dir=='sponge4':
	     	classnr=26
		prefix='sponge'
	elif dir=='sponge3':
	     	classnr=27
		prefix='sponge'

	# Further file preparation for right filename-matching
	# Hack to define which day it is
	if daynr==1:
		tag='day3'
	elif daynr==2:
		tag='day1'
	elif daynr==3:
		tag='day2'
	elif daynr==4:
		tag='day4_manual_crop'
	elif daynr==5:
		tag='day4'

	#txt.write(tag+'/'+prefix+'/'+dir + '/' + filename + ' '+str(classnr)+'\n')
	txt.write(list+' '+str(classnr)+'\n')
	#filename_txt.write(tag+'/'+prefix+'/'+dir + '/' + filename+'\n')
	filename_txt.write(list+'\n')

	classNumbers_txt.write(str(classnr)+' ')

filename_txt.close()
classNumbers_txt.close()
txt.close()
