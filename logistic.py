from loadData import TCGAData
from sklearn.linear_model import LogisticRegression
from testUtils import kFoldCrossValid

def main():
    """Try an ordinary logistic regression first"""
    data = TCGAData()
    gene_exp = data.get_gene_exp_matrix()
    labels = data.get_labels()
    logReg = LogisticRegression(tol=0.01,dual=True)
    accuracy = kFoldCrossValid(gene_exp,labels,logReg)     
    print(accuracy)

if __name__=="__main__":
    main()
