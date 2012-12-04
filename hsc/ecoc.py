from testUtils import evaluateClassifications
from loadData import Data
from testUtils import leaveOneOutCrossValid
import numpy
from sklearn import svm
from sklearn.multiclass import OutputCodeClassifier

def ecocfn(code_size=52./38, C=50, linSVC_L1=False, kernel ='linear', selection='chi2',numFeatures=330):#try different code sizes 
    """l1Reg is used only if linSVC=True"""
    data = Data()
    gene_exp = data.get_gene_exp_matrix()
    labels = data.get_labels()
    if linSVC_L1:
        clf = svm.LinearSVC(C=C,penalty='l1',dual=False)
    else:
        clf = svm.SVC(C=C,kernel=kernel)
    ecocAlgo = OutputCodeClassifier(clf, code_size=code_size, random_state=0)
    print(leaveOneOutCrossValid(gene_exp,labels,ecocAlgo,selection=selection,numFeatures=numFeatures))

if __name__=="__main__":
    #print('code length 47------------------')
    #ecocfn(code_size=47./38,C=50) 
    #print('code length 52------------------')
    #ecocfn(code_size=52./38,C=50)
    #print('code length 57------------------')
    #ecocfn(code_size=57./38,C=50)
    
    """first we try to pick C and kernel""" 
    """print "running through C=30,40,...70, kernel is rbf-----------------"
    ecocfn(C=30,kernel='rbf')
    ecocfn(C=40,kernel='rbf')
    ecocfn(C=50,kernel='rbf')
    ecocfn(C=60,kernel='rbf')
    ecocfn(C=70,kernel='rbf')
    """
    #print "running through C=30,40,...70, kernel is linear-----------------"
    #ecocfn(C=30,kernel='linear')
    #ecocfn(C=40,kernel='linear')
    #ecocfn(C=50,kernel='linear')
    #ecocfn(C=60,kernel='linear')
    #ecocfn(C=70,kernel='linear')
    """
    print "running through C=30,40,...70, kernel is linear-----------------"
    ecocfn(C=30,kernel='poly')
    ecocfn(C=40,kernel='poly')
    ecocfn(C=50,kernel='poly')
    ecocfn(C=60,kernel='poly')
    ecocfn(C=70,kernel='poly')"""

    """with linear kernel, we try l1 norm"""
    print "l1 norm, C=50-----------------------"
    ecocfn(C=50,linSVC_L1=True,selection='None')
