from loadData import *
from sklearn import svm
from sklearn import cross_validation

#can be used both with all features, and a selected set of features (data is expected only to contain those)
def learnWithSVM(trainingData,trainingLabels,testData,testLabels,numFeatures):
	clf = svm.SVC(gamma=0.001,C=100.) #these are the values in some random example, idk what C is
#later set these using cross validation?
	clf.fit(trainingData,trainingLabels)
	predicted = clf.predict(testData)
	return evaluateClassifications(predicted,testLabels)
	
#now just reports accuracy,TODO maybe precision + recall?
def evaluateClassifications(predicted,testLabels):
	numWrong = 0.0
	numRight = 0.0
	fp = 0.0
	fn = 0.0
	tp = 0.0
	tn = 0.0
	
	for i in range(1,len(predicted)):
		if predicted[i]==testLabels[i] and predicted[i]: tp+=1
		elif predicted[i]==testLabels[i] and not predicted[i]: tn+=1
		elif predicted[i]!=testLabels[i] and predicted[i]: fp+=1
		else: fn+=1
	print 'fp: %f' % fp
	print 'fn: %f' % fn
	print 'tp: %f' % tp
	print 'tn: %f' % tn
	return ((tp+tn)/(len(predicted)))

if __name__=="__main__":
	"""
	#just checking
	X_train = [[1,0,-1],[0,1,-1]]
	y_train = [1,0]
	X_test = [[1,0,-1],[1,-1,0],[0,1,-1]]
	y_test = [0,0,0]
	numFeatures = 3;
	
	print learnWithSVM(X_train,y_train,X_test,y_test,numFeatures)
	#testCode()
	"""
        data = TCGAData()
        X = data.get_gene_exp_matrix()
        numFeatures = len(X[0])
	numExamples = len(X)
        Y = data.get_labels()
	print "X has %d examples, y has %d labels" % (numExamples,len(Y))

	kf = cross_validation.KFold(numExamples, k=2,shuffle=True) #TODO vary k

	for train_index, test_index in kf:
	    	print("TRAIN: %s TEST: %s" % (train_index, test_index))
	    	X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
	    	y_train, y_test = [Y[i] for i in train_index], [Y[i] for i in test_index]
		print "Xtrain has %d examples, ytrain has %d labels" % (len(X_train),len(y_train))
		
		print learnWithSVM(X_train,y_train,X_test,y_test,numFeatures)

