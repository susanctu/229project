from testUtils import evaluateClassifications
from loadData import Data
from testUtils import kFoldCrossValid
import numpy

def ecocfn(codeSize=50./38):
    data = Data()
    gene_exp = data.get_gene_exp_matrix()
    labels = data.get_labels()
    clf = svm.SVC(C=1.)
    ecocAlgo = OutputCodeClassifier(LinearSVC(), code_size=code_size, random_state=0).fit(X, y).predict(X)
    leaveOneOutCrossValid(gene_exp,labels,ecocAlgo)

if __name__="__main__":
    ecocfn()
