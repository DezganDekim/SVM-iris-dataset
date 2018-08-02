import csv
import sklearn
import random
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def setdata(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	del dataset[0]
	X=[]
	Y=[]
	for i in range(len(dataset)): 
		del dataset[i][0]
		if dataset[i][-1]=="Iris-setosa": dataset[i][-1]=1
		if dataset[i][-1]=="Iris-versicolor": dataset[i][-1]=2
		if dataset[i][-1]=="Iris-virginica": dataset[i][-1]=3  
		dataset[i] = [float(x) for x in dataset[i]]
		X.append(dataset[i][0:4])
		Y.append(dataset[i][-1])
	return X,Y

def splitDataset(M, N):
	trainSize = int(len(M) * 0.75)
	XT = []
	YT=[]
	copym = M
	copyn=N  
	while len(XT) < trainSize:
		index = random.randrange(len(copym))
		XT.append(copym.pop(index))
		YT.append(copyn.pop(index))
	return XT,YT,copym,copyn

def main():
	filename = 'irisdataset.csv'
	X,Y= setdata(filename)
	trainingSetX,trainingSetY, testSetX,testSetY = splitDataset(X,Y)
	gnb = GaussianNB()
	t0 = time.time()
	gnb = gnb.fit(trainingSetX,trainingSetY)
	print "Training Time:", time.time()-t0, "s"
	t0 = time.time()
	predictions=gnb.predict(testSetX)
	print "prediction Time:",time.time()-t0, "s"
	accuracy=accuracy_score(testSetY,predictions)
	print('Prediction Accuracy: {0}').format(accuracy)
 
main()

