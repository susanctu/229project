from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy
from loadData import Data
import pylab
from numpy import *
import matplotlib.pyplot as plt
from pylab import *

"""This file contains useful functions for testing/evaluating different learning algorithms"""

def print_genes_nonzero_coeff(data,coeffs):#data should be a TCGAData object
    names = data.get_gene_names()
    assert(len(coeffs)==len(names))
    nonzeroNames = []
    for i in range(0,len(names)):
        if coeffs[i]!=0:
            nonzeroNames.append(names[i])
    return nonzeroNames

def kFoldCrossValid(X,Y,learningAlgo,k=210,names=None,selection='none'):#learningAlgo is an object, not a function! and assumes that X and Y are already numpy.arrays 

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
		#print "Xtrain has %d examples, ytrain has %d labels" % (len(X_train),len(y_train))
    		if(selection=='chi2'):
			numFeatures = len(X_train[0])
			numExamples = len(y_train)
			fs = SelectKBest(chi2,k=250)
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
		learningAlgo.fit(numpy.array(X_train),numpy.array(y_train))
                predictions = []
                for x_vec in X_test:
                    predictions.append(learningAlgo.predict(x_vec)[0])
		#print 'y_test:'
		#print y_test
		#print 'predictions:'
		#print predictions
                accuracy.append(evaluateClassifications(predictions,y_test))
        return accuracy

def displayConfusion(conf_arr):#taken from this stackoverflow post: http://stackoverflow.com/questions/2897826/confusion-matrix-with-number-of-classified-misclassified-instances-on-it-python
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i,0)
        for j in i:
                tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    res = ax.imshow(array(norm_conf), cmap=cm.jet, interpolation='nearest')
    for i, cas in enumerate(conf_arr):
        for j, c in enumerate(cas):
            if c>0:
                plt.text(j-.2, i+.2, c, fontsize=14)
    cb = fig.colorbar(res)
    savefig("confmat.png", format="png")
    plt.show()

def confusionMatrix(predicted,testLabels,numClasses=38):
    mat = []
    for i in range(0,numClasses):
        mat.append([0]*numClasses) 
    for i in range(0,len(predicted)):
        row = testLabels[i]
        col = predicted[i]
        mat[row][col]+=1
    return(mat)
 
def evaluateClassifications(predicted,testLabels):
        """Returns a list [accuracy,precision,recall, fp, fn, tp, tn]
           Yes, it would be nicer if I made an object with these attributes.
           Complain to me if you want me to change it to that. 
        """
	numWrong = 0.0
	numRight = 0.0
	
	for i in range(0,len(predicted)):
                print(predicted[i].__str__() + ' ' + testLabels[i].__str__())
		if predicted[i]==testLabels[i]: numRight+=1
		else: numWrong+=1
	accuracy = numRight / (numRight + numWrong)
        return([accuracy])

if __name__=="__main__":
    predicted = [21, 6, 21, 21, 8, 2, 2, 2, 2, 2, 1, 1, 1, 3, 3, 2, 2, 2, 2, 2, 2, 2,1, 2, 2, 22, 22, 22, 0, 8, 0, 0, 7, 30, 24, 24, 24, 24, 21, 21, 0, 0,21, 24, 22, 22, 18, 18, 22, 22, 22, 21, 18, 22, 18, 18, 18, 18, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 12, 13, 11, 12, 12, 12, 12, 12, 12, 18, 18, 24, 22, 34, 32, 18, 34, 34, 34, 24, 24, 24, 24, 24, 24, 24, 18, 18, 18, 18, 18, 22, 18, 18, 18, 18, 18, 18, 18, 18, 21, 21,21, 22, 22, 21, 21, 10, 21, 21, 21, 21, 18, 18, 18, 18, 22, 18, 22,22, 22, 24, 24, 21, 24, 7, 7, 7, 7, 7, 24, 24, 24, 8, 34, 34, 36, 36,34, 34, 34, 34, 37, 34, 6, 24, 34, 6, 30, 30, 6, 21, 0, 0, 0, 21, 21,34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 34, 36, 36, 36, 34, 34, 36, 36, 36, 36, 36, 36, 36, 32, 32, 32, 32, 32, 32, 32, 37, 34, 34,34, 34, 34, 37, 35, 36, 34, 36, 36, 34, 32]
    actual = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4,4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9,
9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 15,
15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37]
    displayConfusion(confusionMatrix(predicted,actual))    
    #conf_arr = [[33,2,0,0,0,0,0,0,0,1,3], [3,31,0,0,0,0,0,0,0,0,0], [0,4,41,0,0,0,0,0,0,0,1], [0,1,0,30,0,6,0,0,0,0,1], [0,0,0,0,38,10,0,0,0,0,0], [0,0,0,3,1,39,0,0,0,0,4], [0,2,2,0,4,1,31,0,0,0,2], [0,1,0,0,0,0,0,36,0,2,0], [0,0,0,0,0,0,1,5,37,5,1], [3,0,0,0,0,0,0,0,0,39,0], [0,0,0,0,0,0,0,0,0,0,38] ]
    #displayConfusion(conf_arr) 
