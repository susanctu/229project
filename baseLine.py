from sklearn import svm
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from testUtils import evaluateClassifications
from loadData import TCGAData
from testUtils import kFoldCrossValid
import numpy

#TODO DOING FEATURE SELECTION ON ALL DATA, SO RESULTING ACCURACY DOES NOT REFLECT REAL PREDICTION POWER

def svmfn(featureSelectionMethod = 'none'):
    data = TCGAData()
    gene_exp = data.get_gene_exp_matrix()
    labels = data.get_labels()
    names = data.get_gene_names()
    clf = svm.SVC(gamma=0.001,C=100.) #these are the values in some random example, idk what C is
    accuracy = kFoldCrossValid(gene_exp,labels,clf,k=4,names=names,selection=featureSelectionMethod)     
    print(accuracy)


#can be used both with all features, and a selected set of features (data is expected only to contain those)
def learnWithSVM(trainingData,trainingLabels,testData,testLabels,numFeatures):
	clf = svm.SVC(gamma=0.001,C=100.) #these are the values in some random example, idk what C is
#TODO later set these using cross validation?
	clf.fit(trainingData,trainingLabels)
	predicted = clf.predict(testData)
	return evaluateClassifications(predicted,testLabels)[0]
	
if __name__=="__main__":
	svmfn()
	svmfn('chi2')
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
