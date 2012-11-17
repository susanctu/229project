import numpy as np
from loadData import TCGAData
from sklearn.linear_model import LogisticRegression
from testUtils import kFoldCrossValid, print_genes_nonzero_coeff

def with_l1_penalty(data,C_list):#data should be TCGAData object
    X = data.get_gene_exp_matrix()
    y = data.get_labels()

    """Adapted from sklearn website,  
        # Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
        #          Mathieu Blondel <mathieu@mblondel.org>
        #          Andreas Mueller <amueller@ais.uni-bonn.de>
        # License: BSD Style.
    """
    for C in C_list:
            l1_logReg = LogisticRegression(C=C, penalty='l1', tol=0.01)
            l2_logReg = LogisticRegression(C=C, penalty='l2', tol=0.01)
            l1_logReg.fit(X, y)
            l2_logReg.fit(X, y)

            coef_l1_LR = l1_logReg.coef_.ravel()
            print print_genes_nonzero_coeff(data,coef_l1_LR)
            print('\n\n')
            coef_l2_LR = l2_logReg.coef_.ravel()

            sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
            sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100

            print "C=%.2f" % C
            print "Sparsity with L1 penalty: %.2f%%" % sparsity_l1_LR
            print "score with L1 penalty: %.4f" % l1_logReg.score(X, y)
            print "Sparsity with L2 penalty: %.2f%%" % sparsity_l2_LR
            print "score with L2 penalty: %.4f" % l2_logReg.score(X, y)

            l1_accuracy = kFoldCrossValid(X,y,l1_logReg)     
            print(l1_accuracy)
            l2_accuracy = kFoldCrossValid(X,y,l2_logReg)     
            print(l2_accuracy)


def ordinary_logistic(X,y):
    logReg = LogisticRegression(C=1000000,tol=0.01,dual=True)#large C is less regularization, according to their docs (what is C?)
    accuracy = kFoldCrossValid(X,y,logReg)     
    print(accuracy)

def main():
    """Try an ordinary logistic regression first"""
    data = TCGAData()
    with_l1_penalty(data,[0.1,0.5])    

if __name__=="__main__":
    main()
