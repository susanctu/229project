from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy
from loadData import Data
import pylab
import matplotlib.pyplot as plt

"""This file contains useful functions for testing/evaluating different learning algorithms"""

def print_genes_nonzero_coeff(data,coeffs):#data should be a TCGAData object
    names = data.get_gene_names()
    assert(len(coeffs)==len(names))
    nonzeroNames = []
    for i in range(0,len(names)):
        if coeffs[i]!=0:
            nonzeroNames.append(names[i])
    return nonzeroNames

def trainingSetPerformance(X,y,learningAlgo,names=None,selection='none',numFeatures = 330):#learningAlgo is an object, not a function! and assumes that X and Y are already numpy.arrays 

	print 'numFeatures is:'
	print numFeatures
        """
	TODO CHANGE THIS
        Expects matrix with feature vectors, labels, a learning algorithm, and (optionally) k and a feature selection method. Currently supporting 'chi2' and 'none' and 'random'.
        The learning algorithm needs to have a "fit" method that takes matrix with feature vectors and labels
        and a predict method that takes in just one feature vector and returns a list (of length 1) with the prediction
        (The fact that a list rather than a single value is returned is just due to the fact that that's what sklearn's 
        learning algorithms' predict functions do.)

        Returns list with accuracies.
        """
#	print 'num samples??'
#	print len(X)
	numRight = 0.0
	numWrong = 0
	predictions = []
	actual = []
    	if(selection=='chi2'):
			fs = SelectKBest(chi2,k=numFeatures)
			fs.fit(numpy.array(X)*1000,y)
			indices =  fs.get_support() #I think this gives you a bit mask of which features you want
			X = numpy.array(X)
			X = X[:,indices]
    	elif selection=='random':
			print 'random!!'
			totalNumFeatures = len(X[0])
			numExamples = len(y)
			indices = numpy.random.randint(0,totalNumFeatures-1,numFeatures)
			names =numpy.array(names)
			X = numpy.array(X)
			X = X[:,indices]
	learningAlgo.fit(numpy.array(X),numpy.array(y))
	i=0
	for x_vec in X:
		p = learningAlgo.predict(x_vec)[0]
		print 'we predict:'
		print p
		print 'the actual result:'
		print y[i] 
		predictions.append(p)
		if p == y[i]: numRight+=1
		else: numWrong+=1
		i+=1
	print 'predictions:'
	print predictions
	print 'actual:'
	print y
	print zip(predictions,actual)
	accuracy = numRight / (numRight + numWrong)
	print accuracy
	#displayConfusion(confusionMatrix(predictions,actual))
        return accuracy

def leaveOneOutCrossValid(X,Y,learningAlgo,names=None,selection='none',numFeatures = 330):#learningAlgo is an object, not a function! and assumes that X and Y are already numpy.arrays 

	print 'numFeatures is:'
	print numFeatures
        """
	TODO CHANGE THIS
        Expects matrix with feature vectors, labels, a learning algorithm, and (optionally) k and a feature selection method. Currently supporting 'chi2' and 'none' and 'random'.
        The learning algorithm needs to have a "fit" method that takes matrix with feature vectors and labels
        and a predict method that takes in just one feature vector and returns a list (of length 1) with the prediction
        (The fact that a list rather than a single value is returned is just due to the fact that that's what sklearn's 
        learning algorithms' predict functions do.)

        Returns list with accuracies.
        """
#	print 'num samples??'
#	print len(X)
	l1o = cross_validation.LeaveOneOut(len(X)) 
	numRight = 0.0
	numWrong = 0
	predictions = []
	actual = []
	for train_index, test_index in l1o:
#		print 'run of leave one out'
	    	X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
	    	y_train, y_test = [Y[i] for i in train_index], [Y[i] for i in test_index]
    		if(selection=='chi2'):
			fs = SelectKBest(chi2,k=numFeatures)
			fs.fit(numpy.array(X)*1000,Y)
			indices =  fs.get_support() #I think this gives you a bit mask of which features you want
			#names =numpy.array(names)
			#print names[indices]
			X_train = numpy.array(X_train)
			X_train = X_train[:,indices]
			X_test = numpy.array(X_test)
			X_test = X_test[:,indices]
    		elif selection=='random':
			print 'random!!'
			numTotalFeatures = len(X_train[0])
			numExamples = len(y_train)
			indices = numpy.random.randint(0,numTotalFeatures-1,numFeatures)
			names =numpy.array(names)
			print names[indices]
			X_train = numpy.array(X_train)
			X_train = X_train[:,indices]
			X_test = numpy.array(X_test)
			X_test = X_test[:,indices]
		learningAlgo.fit(numpy.array(X_train),numpy.array(y_train))
		x_vec = X_test[0]
		p = learningAlgo.predict(x_vec)[0]
		print 'we predict:'
		print p
		print 'the actual result:'
		print y_test[0] 
		predictions.append(p)
		actual.append(y_test[0])
		if p == y_test[0]: numRight+=1
		else: numWrong+=1
	print 'predictions:'
	print predictions
	print 'actual:'
	print actual
	print zip(predictions,actual)
	accuracy = numRight / (numRight + numWrong)
	print accuracy
	#displayConfusion(confusionMatrix(predictions,actual))
        return accuracy

"""
conf_array should like as follows:

"""
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
    res = ax.imshow(numpy.array(norm_conf), cmap=plt.cm.jet, interpolation='nearest')
    for i, cas in enumerate(conf_arr):
        for j, c in enumerate(cas):
            if c>0:
                plt.text(j-.2, i+.2, c, fontsize=8)
    cb = fig.colorbar(res)
    plt.xticks(range(0,35),('BASO', 'BCELLa1','BCELLa2','BCELLa3','BCELLa4','CMP','DENDa1','DENDa2','EOS2','ERY1','ERY2','ERY3-5','GMP','GRAN1','GRAN2','GRAN3','HSC1','HSC2','MEGA','MEP','MONO1','MONO2','NKa1','NKa2','NKa3','NKa4','PRE_BCELL2','PRE_BCELL3','TCEL1','TCEL2','TCEL3','TCEL4','TCEL6','TCEL7','TCEL8'),rotation=90)
    plt.yticks(range(0,35),('BASO', 'BCELLa1','BCELLa2','BCELLa3','BCELLa4','CMP','DENDa1','DENDa2','EOS2','ERY1','ERY2','ERY3-5','GMP','GRAN1','GRAN2','GRAN3','HSC1','HSC2','MEGA','MEP','MONO1','MONO2','NKa1','NKa2','NKa3','NKa4','PRE_BCELL2','PRE_BCELL3','TCEL1','TCEL2','TCEL3','TCEL4','TCEL6','TCEL7','TCEL8'))
    plt.savefig("confmat.png", format="png")
    plt.show()

def confusionMatrix(predicted,testLabels,numClasses=35):
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

def showBestECOC():
    predicted = [8, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 4, 1, 3, 3, 3, 4, 4, 4, 4, 4, 22, 14, 22, 35, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 0, 8, 8, 8, 24, 9, 9, 22, 9, 22, 22, 22, 10, 10, 10, 10, 10, 10, 10, 12, 12, 11, 12, 12, 12, 11, 11, 13, 13, 12, 11, 11, 13, 12, 12, 13, 13, 12, 18, 17, 24, 22, 15, 15, 15, 15, 15, 15, 16, 16, 16, 26, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 18, 18, 19, 21, 21, 20, 9, 18, 21, 21, 21, 21, 20, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 24, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 26, 27, 27, 28, 25, 33, 28, 29, 29, 29, 30, 30, 30, 30, 30, 29, 26, 34, 34, 26, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 35, 36, 37, 37, 37, 37, 37]
    actual = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37]
    mat = confusionMatrix(predicted,actual)
    print(mat)
    displayConfusion(mat)   
 
def showBest1vsAll():
    results = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (4, 4), (3, 4), (4, 4), (4, 4), (4, 4), (5, 5), (5, 5), (19, 5), (32, 5), (6, 6), (6, 6), (6, 6), (6, 6), (6, 6), (7, 7), (7, 7), (7, 7), (7, 7), (7, 7), (8, 8), (8, 8), (8, 8), (8, 8), (0, 8), (9, 9), (5, 9), (17, 9), (9, 9), (19, 9), (19, 9), (19, 9), (9, 10), (10, 10), (10, 10), (10, 10), (10, 10), (10, 10), (10, 10), (11, 11), (11, 11), (11, 11), (11, 11), (11, 11), (11, 11), (11, 11), (11, 11), (11, 11), (11, 11), (11, 11), (11, 11), (11, 11), (11, 11), (11, 11), (11, 11), (11, 11), (11, 11), (11, 11), (16, 12), (12, 12), (21, 12), (5, 12), (13, 13), (13, 13), (13, 13), (13, 13), (13, 13), (13, 14), (14, 14), (14, 14), (14, 14), (15, 15), (15, 15), (15, 15), (15, 15), (16, 16), (16, 16), (16, 16), (16, 16), (16, 16), (16, 16), (16, 16), (16, 16), (16, 16), (16, 16), (17, 17), (16, 17), (16, 17), (17, 17), (18, 18), (18, 18), (18, 18), (18, 18), (18, 18), (18, 18), (18, 18), (18, 18), (18, 18), (18, 18), (18, 18), (18, 18), (19, 19), (19, 19), (19, 19), (19, 19), (5, 19), (19, 19), (19, 19), (19, 19), (18, 19), (20, 20), (12, 20), (20, 20), (20, 20), (21, 21), (21, 21), (21, 21), (21, 21), (8, 21), (22, 22), (22, 22), (22, 22), (22, 22), (23, 23), (23, 23), (23, 23), (23, 23), (23, 23), (24, 24), (24, 24), (24, 24), (24, 24), (24, 24), (25, 25), (22, 25), (30, 25), (25, 25), (26, 26), (26, 26), (26, 26), (27, 26), (27, 27), (0, 27), (27, 27), (27, 27), (26, 27), (23, 28), (14, 28), (30, 28), (30, 28), (29, 29), (29, 29), (29, 29), (29, 29), (29, 29), (29, 29), (29, 29), (30, 30), (30, 30), (30, 30), (30, 30), (30, 30), (30, 30), (31, 31), (31, 31), (31, 31), (31, 31), (31, 31), (31, 31), (31, 31), (32, 32), (32, 32), (32, 32), (32, 32), (32, 32), (32, 32), (32, 32), (33, 33), (33, 33), (33, 33), (33, 33), (33, 33), (33, 33), (33, 33), (34, 34), (34, 34), (34, 34), (34, 34), (34, 34), (34, 34), (34, 34)]
    pred = [elem[0] for elem in results]
    act = [elem[1] for elem in results]
    mat = confusionMatrix(pred,act)
    print(mat)
    displayConfusion(mat)    

def showBest1vs1():
    predicted=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 35, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 24, 9, 5, 22, 9, 22, 22, 22, 10, 10, 20, 10, 10, 10, 10, 11, 11, 11, 13, 12, 12, 11, 11, 11, 11, 22, 11, 11, 11, 11, 11, 11, 13, 13, 16, 16, 8, 5, 15, 15, 15, 15, 16, 15, 16, 16, 16, 24, 17, 17, 17, 18, 18, 18, 18, 18, 9, 18, 18, 18, 18, 19, 18, 19, 19, 21, 20, 20, 9, 22, 20, 21, 21, 21, 21, 21, 21, 22, 9, 22, 22, 22, 22, 9, 9, 22, 24, 14, 23, 23, 24, 24, 24, 24, 8, 25, 25, 28, 25, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 25, 28, 28, 29, 29, 29, 30, 30, 30, 30, 30, 21, 26, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37]
    actual= [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9,
9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15,15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27,27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 30, 31, 31,31, 31, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37]
    mat = confusionMatrix(predicted,actual)
    print(mat)
    displayConfusion(mat)    

if __name__=="__main__":
    showBest1vsAll()
    #showBest1vs1()
    #showBestECOC()
