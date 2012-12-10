from sklearn import svm
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from testUtils import evaluateClassifications
from loadData import Data
#from testUtils import kFoldCrossValid
from testUtils import leaveOneOutCrossValid
import numpy
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
import itertools

"""
param featureSelectionMethod: pass in a string, passes it on into the methods in testUtils (currently supports 'chi2','random') - default is 'none'
calls an svm with C =

"""

def addlists(xs,ys):
     return list(x + y for x, y in itertools.izip(xs, ys))

def svmfn(featureSelectionMethod = 'none',numFeatures = '330'):
    data = Data()
    gene_exp = data.get_gene_exp_matrix()
    labels = data.get_labels()
    names = data.get_gene_names()
    clf = svm.LinearSVC(C=125.,penalty="l1",dual=False)
    clfwrapper = OneVsRestClassifier(clf);
    accuracy = leaveOneOutCrossValid(gene_exp,labels,clfwrapper,names=names,selection=featureSelectionMethod,numFeatures=numFeatures)     
    print 'accuracy is'
    print accuracy
    estimators = clfwrapper.estimators_
    j = 0 
    for estimator in estimators:
	print 'estimator for class'
	print data.getCellName(j)
	i = range(0,len(estimator.coef_[0]))
	b = sorted(zip(estimator.coef_[0], i), reverse=True)[:80] #TODO CHANGE
	indices = data.indices_of_celltype(j)
	#print 'indices of this class:'
	#print indices
	arraysum = [0.0]*11927
	for i in indices:
		#print 'adding gene exp'
		#print gene_exp[i]
		arrayssum = addlists(arraysum,gene_exp[i])
		#print 'arraysum is now'
		#print arraysum
	#print 'arraysum is '
	#print arraysum
	arrayavg =  [x/len(indices) for x in arraysum]
	k = 0
	geneList = []
	while k<80  and  b[k][0] > 0.0000000000000001 : #TODO CHANGE
		avg_expr = arrayavg[b[k][1]]
		geneStr = names[b[k][1]] + ':' +  str(avg_expr)
		geneList = geneList + [geneStr]
		k = k+1
	j = j+1
	#print geneList
	
if __name__=="__main__":
	svmfn()
	#svmfn('chi2',numFeatures = 10)
	#svmfn('random')
	"""
	basicFeatureSelection = True #Uses all features if false, forward feature selection using chi2 if true
	#just checking
	X_train = [[1,0,-1],[0,1,-1]]
	y_train = [1,0]
	X_test = [[1,0,-1],[1,-1,0],[0,1,-1]]
	y_test = [0,0,0]
	numFeatures = 3;
	
	print learnWithSVM(X_train,y_train,X_test,y_test,numFeatures)
	#testCode()
        data = TCGAData()
	if basicFeatureSelection:
        	X = data.get_gene_exp_matrix()
                numFeatures = len(X[0])
		numExamples = len(X)
        	Y = data.get_labels()
		fs = SelectKBest(chi2)
		fs.fit(X,Y)
		print fs.get_support() #I think this gives you a bit mask of which features you want
		#TODO continue here
		
	else:
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

	"""
