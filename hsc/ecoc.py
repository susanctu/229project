from testUtils import evaluateClassifications
from loadData import Data
from testUtils import leaveOneOutCrossValid
import numpy
from sklearn import svm
from sklearn.multiclass import OutputCodeClassifier

def ecocfn(code_size=50./38, C=50):#try different code sizes 
    data = Data()
    gene_exp = data.get_gene_exp_matrix()
    labels = data.get_labels()
    clf = svm.SVC(C=C)
    ecocAlgo = OutputCodeClassifier(clf, code_size=code_size, random_state=0)
    print(leaveOneOutCrossValid(gene_exp,labels,ecocAlgo,selection='chi2'))

if __name__=="__main__":
    print('code length 47------------------')
    ecocfn(code_size=47./38,C=50) 
    print('code length 52------------------')
    ecocfn(code_size=52./38,C=50)
    print('code length 57------------------')
    ecocfn(code_size=57./38,C=50)
