from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy
from loadData import Data

"""This file contains useful functions for testing/evaluating different learning algorithms"""

def print_genes_nonzero_coeff(data,coeffs):#data should be a TCGAData object
    names = data.get_gene_names()
    assert(len(coeffs)==len(names))
    nonzeroNames = []
    for i in range(0,len(names)):
        if coeffs[i]!=0:
            nonzeroNames.append(names[i])
    return nonzeroNames

def kFoldCrossValid(X,Y,learningAlgo,k=4,names=None,selection='none'):#learningAlgo is an object, not a function! and assumes that X and Y are already numpy.arrays 

        """
        Expects matrix with feature vectors, labels, a learning algorithm, and (optionally) k and a feature selection method. Currently supporting 'chi2' and 'none' and 'random'.
        The learning algorithm needs to have a "fit" method that takes matrix with feature vectors and labels
        and a predict method that takes in just one feature vector and returns a list (of length 1) with the prediction
        (The fact that a list rather than a single value is returned is just due to the fact that that's what sklearn's 
        learning algorithms' predict functions do.)

        Returns list with accuracies.
        """

	kf = cross_validation.KFold(len(X), k=k,shuffle=True) #TODO vary k
        accuracy = []
	for train_index, test_index in kf:
	    	#print("TRAIN: %s TEST: %s" % (train_index, test_index))
	    	X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
	    	y_train, y_test = [Y[i] for i in train_index], [Y[i] for i in test_index]
		print "y_train is" 
		print y_train
		#print "Xtrain has %d examples, ytrain has %d labels" % (len(X_train),len(y_train))
    		if(selection=='chi2'):
			numFeatures = len(X_train[0])
			numExamples = len(y_train)
			fs = SelectKBest(chi2,k=50)
			#fs.fit(numpy.array(X_train)*1000,y_train)
			fs.fit(numpy.array(X)*1000,Y)
			indices =  fs.get_support() #I think this gives you a bit mask of which features you want
			names =numpy.array(names)
			print names[indices]
			X_train = numpy.array(X_train)
			X_train = X_train[:,indices]
			X_test = numpy.array(X_test)
			X_test = X_test[:,indices]
    		elif selection=='random':
			print 'random!!'
			numFeatures = len(X_train[0])
			numExamples = len(y_train)
			indices = numpy.random.randint(0,numFeatures-1,50)
			names =numpy.array(names)
			print names[indices]
			X_train = numpy.array(X_train)
			X_train = X_train[:,indices]
			X_test = numpy.array(X_test)
			X_test = X_test[:,indices]
		learningAlgo.fit(X_train,y_train)
                predictions = []
                for x_vec in X_test:
                    predictions.append(learningAlgo.predict(x_vec)[0])
                accuracy.append(evaluateClassifications(predictions,y_test))
        return accuracy

def evaluateClassifications(predicted,testLabels):
        """Returns a list [accuracy,precision,recall, fp, fn, tp, tn]
           Yes, it would be nicer if I made an object with these attributes.
           Complain to me if you want me to change it to that. 
        """
	numWrong = 0.0
	numRight = 0.0
	fp = 0.0
	fn = 0.0
	tp = 0.0
	tn = 0.0
	
	for i in range(0,len(predicted)):
		if predicted[i]==testLabels[i]: numRight+=1
		else: numWrong+=1
	accuracy = numRight / (numRight + numWrong)
        return([accuracy])
