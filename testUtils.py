from sklearn import cross_validation

"""This file contains useful functions for testing/evaluating different learning algorithms"""

def print_genes_nonzero_coeff(data,coeffs):#data should be a TCGAData object
    names = data.get_gene_names()
    assert(len(coeffs)==len(names))
    nonzeroNames = []
    for i in range(0,len(names)):
        if coeffs[i]!=0:
            nonzeroNames.append(names[i])
    return nonzeroNames


def kFoldCrossValid(X,Y,learningAlgo,k=4):#learningAlgo is an object, not a function! and assumes that X and Y are already numpy.arrays 

        """
        Expects matrix with feature vectors, labels, a learning algorithm, and (optionally) k.
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
		learningAlgo.fit(X_train,y_train)
                predictions = []
                for x_vec in X_test:
                    predictions.append(learningAlgo.predict(x_vec)[0])
                accuracy.append(evaluateClassifications(predictions,y_test))
        return(accuracy)

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
		if predicted[i]==testLabels[i] and predicted[i]: tp+=1
		elif predicted[i]==testLabels[i] and not predicted[i]: tn+=1
		elif predicted[i]!=testLabels[i] and predicted[i]: fp+=1
		else: fn+=1
	#print 'fp: %f' % fp
	#print 'fn: %f' % fn
	#print 'tp: %f' % tp
	#print 'tn: %f' % tn 
	accuracy = ((tp+tn)/(len(predicted)))
        if fp+tp==0:
            precision = float("inf")
        else:
            precision = tp/(fp+tp) 
        if tp+fn==0:
            recall = float("inf") 
        else:
            recall = tp/(tp+fn)
        return([accuracy,precision,recall,fp,fn,tp,tn])
