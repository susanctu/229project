import numpy as np
import pylab as pl
from loadData import TCGAData
from sklearn.linear_model import LogisticRegression
from testUtils import kFoldCrossValid, print_genes_nonzero_coeff
from sklearn import cross_validation
#returns a list of sparisities for the times train/test was repeated 
def kFoldGetSparsity(data,logregAlgo,k=4):
     print "--------------------------------------------"
     X = data.get_gene_exp_matrix()
     Y = data.get_labels()
     kf = cross_validation.KFold(len(X), k=k,shuffle=True)
     sparsity = []
     for train_index, test_index in kf:
         X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
         y_train, y_test = [Y[i] for i in train_index], [Y[i] for i in test_index]
         logregAlgo.fit(X_train,y_train)
         coeffs = logregAlgo.coef_.ravel()
         print print_genes_nonzero_coeff(data,coeffs) 
         sparsity.append(np.mean(coeffs==0)*100)
     return(sparsity)
   
def with_l1_penalty(data,C_list):#data should be TCGAData object
    
    """Adapted from sklearn website,  
        # Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
        #          Mathieu Blondel <mathieu@mblondel.org>
        #          Andreas Mueller <amueller@ais.uni-bonn.de>
        # License: BSD Style.
    """
    X = data.get_gene_exp_matrix()
    y = data.get_labels()
    evaluations = []
    sparsityList = []
    for C in C_list:
            l1_logReg = LogisticRegression(C=C, penalty='l1', tol=0.01)
            sparsities = kFoldGetSparsity(data,l1_logReg)
            sparsity = sum(sparsities)/len(sparsities)
            sparsityList.append(sparsity)
            print "C=%.4f" % C
            print "Sparsity with L1 penalty: %.2f%%" % sparsity
            #print "Sparsity with L2 penalty: %.2f%%" % sparsity_l2_LR
            #print "score with L2 penalty: %.4f" % l2_logReg.score(X, y)

            l1_eval = kFoldCrossValid(X,y,l1_logReg)    
            l1_eval_sum = reduce(lambda x,y: [a+b for a,b in zip(x,y)],l1_eval)
            l1_eval_avg = map(lambda x: x/len(l1_eval),l1_eval_sum)
            evaluations.append(l1_eval_avg)
            #l2_accuracy = kFoldCrossValid(X,y,l2_logReg)     
            #print(l2_accuracy)

    C_list = np.array(C_list)
    print(C_list)
    precisions = [100*evaluations[i][1] for i in range(0,len(C_list))]
    print(precisions)
    recalls = [100*evaluations[i][2] for i in range(0,len(C_list))]
    print(recalls)
    fig = pl.figure()
    precis_recall = fig.add_subplot(211)
    sparse = fig.add_subplot(212)

    precis_recall.set_title('Precision/Recall with Varying C')
    precis_recall.plot(C_list,precisions,color="red")
    precis_recall.plot(C_list,recalls,color="blue")
    precis_recall.set_xlabel("C")    
    precis_recall.set_ylabel("Precision (red)/Recall (blue)")
    
    sparse.set_title('Sparsity with Varying C')
    sparse.plot(C_list,sparsityList,color="black")
    sparse.set_xlabel("C")    
    sparse.set_ylabel("Sparsity")
    pl.show()

def ordinary_logistic(data):
    X = data.get_gene_exp_matrix()
    y = data.get_labels()
    logReg = LogisticRegression(C=1000000,tol=0.01,dual=True)#large C is less regularization, according to their docs (what is C?)
    logis_eval = kFoldCrossValid(X,y,logReg)     
    eval_sum = reduce(lambda x,y: [a+b for a,b in zip(x,y)],logis_eval)
    eval_avg = map(lambda x: x/len(logis_eval),eval_sum)
    print(eval_avg)

def main():
    data = TCGAData()
    #ordinary_logistic(data)
    with_l1_penalty(data,[0.004,0.02,0.1,0.5])    

if __name__=="__main__":
    main()
