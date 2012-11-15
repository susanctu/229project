import numpy as np
from loadData import TCGAData
from sklearn.linear_model import LogisticRegression
from testUtils import kFoldCrossValid

def with_l1_penalty(X,y):

    """From 
        # Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
        #          Mathieu Blondel <mathieu@mblondel.org>
        #          Andreas Mueller <amueller@ais.uni-bonn.de>
        # License: BSD Style.
    
        WARNING: This code runs but probably does not work correctly yet
    """
    for i, C in enumerate(1.1 ** np.arange(0, 5)):
            l1_logReg = LogisticRegression(C=C, penalty='l1', tol=0.01)
            l2_logReg = LogisticRegression(C=C, penalty='l2', tol=0.01)
            l1_logReg.fit(X, y)
            l2_logReg.fit(X, y)

            coef_l1_LR = l1_logReg.coef_.ravel()
            coef_l2_LR = l2_logReg.coef_.ravel()

            sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
            sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100

            print "C=%.2f" % C
            print "Sparsity with L1 penalty: %.2f%%" % sparsity_l1_LR
            print "score with L1 penalty: %.4f" % l1_logReg.score(X, y)
            print "Sparsity with L2 penalty: %.2f%%" % sparsity_l2_LR
            print "score with L2 penalty: %.4f" % l2_logReg.score(X, y)

            """l1 = LogisticRegression(C=C, penalty='l1', tol=0.01)
            l2 = LogisticRegression(C=C, penalty='l2', tol=0.01)
            l1_accuracy = kFoldCrossValid(X,y,l1)     
            print(l1_accuracy)
            l2_accuracy = kFoldCrossValid(X,y,l2)     
            print(l2_accuracy)"""


def ordinary_logistic(X,y):
    logReg = LogisticRegression(C=1000000,tol=0.01,dual=True)#large C is less regularization, according to their docs (what is C?)
    accuracy = kFoldCrossValid(X,y,logReg)     
    print(accuracy)

def main():
    """Try an ordinary logistic regression first"""
    data = TCGAData()
    gene_exp = data.get_gene_exp_matrix()
    labels = data.get_labels()
    with_l1_penalty(gene_exp,labels)    

if __name__=="__main__":
    main()
