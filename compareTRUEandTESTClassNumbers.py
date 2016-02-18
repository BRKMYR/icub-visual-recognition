from __future__ import division

# TRUE (Ground Truth)
true_classNumbers_txt=open('/home/niklas/Desktop/textfiles/test_TRUE_classNumbers.txt','r')
true_classNumbers=true_classNumbers_txt.readlines()
true_classNumbers_list=[]
for line in true_classNumbers:
	values=map(int, line.split())
	true_classNumbers_list.append(values)
true_classNumbers_txt.close()
true_classNumbers_list=true_classNumbers_list[0]

# TEST (Prediction)
test_classNumbers_txt=open('/home/niklas/Desktop/textfiles/test_predictedNumbers.txt','r')
test_classNumbers=test_classNumbers_txt.readlines()
predicted_classNumbers_list=[]
for line in test_classNumbers:
	values=map(int, line.split())
	predicted_classNumbers_list.append(values)
test_classNumbers_txt.close()
predicted_classNumbers_list=predicted_classNumbers_list[0]

# Comparison
anzahlVector=[0]*28
detectedVector=[0]*28
classAccuracyVector=[0]*28

correctPredictions=0
overallPerformance=0.00
for num in range(0,len(predicted_classNumbers_list)-1):
	anzahlVector[true_classNumbers_list[num]]+=1
	if true_classNumbers_list[num]==predicted_classNumbers_list[num]:
		correctPredictions+=1
		detectedVector[true_classNumbers_list[num]]+=1

overallPerformance=correctPredictions/len(predicted_classNumbers_list)
print ("Overall Performance is: "+str(overallPerformance))
print ("\n")

for i in range(0,28):
	classAccuracyVector[i]=detectedVector[i]/anzahlVector[i]
	print ("Accuracy for class "+str(i+1)+" is: "+str(classAccuracyVector[i])+"\n")
	
print ("Detected objects: ")
print detectedVector
print ("\n")
print ("Total number of objects: ")
print anzahlVector
print ("\n")
