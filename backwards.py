from loadData import TCGAData
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from testUtils import print_genes_nonzero_coeff
from sklearn.metrics import zero_one
from sklearn.feature_selection import RFE

def rec_feature_elim(data,num_features=20):
    X = data.get_gene_exp_matrix()
    y = data.get_labels()
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=num_features, step=1)
    selector = rfe.fit(X, y)
    mask = map(lambda x: 1 if x is True else 0,selector.support_)
    print_genes_nonzero_coeff(data,mask)

def rec_feature_elim_with_KFold(data):
    """Recursive feature elimination 
    FIXME: How to pick a kernel?
    WARNING: ridiculously slow?
    """
    X = data.get_gene_exp_matrix()
    y = data.get_labels()
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, 2),loss_func=zero_one)
    selector = rfecv.fit(X, y) 
    mask = map(lambda x: 1 if x is True else 0,selector.support_)
    print_genes_nonzero_coeff(data,mask)
    print "Optimal number of features : %d" % rfecv.n_features_

def main():
    data = TCGAData()
    rec_feature_elim(data)

if __name__=="__main__":
    main()
