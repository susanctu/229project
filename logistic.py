from loadData import TCGAData
from baseLine import evaluateClassifications 
from sklearn.linear_model import LogisticRegression

def main():
    """Try an ordinary logistic regression first"""
    data = TCGAData()
    gene_exp = data.get_gene_exp_matrix()
    labels = data.get_labels()
    logReg = LogisticRegression(tol=0.01,dual=True)
    logReg.fit(gene_exp,labels)
    predictions = []
    for exp in gene_exp:
        predictions.append(logReg.predict(exp)[0])
    print "Accuracy : %f" % evaluateClassifications(predictions,labels.tolist())

if __name__=="__main__":
    main()
