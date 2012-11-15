from sklearn.naive_bayes import GaussianNB
from loadData import TCGAData
from baseLine import evaluateClassifications 

def main():
    data = TCGAData()
    gene_exp = data.get_gene_exp_matrix()
    labels = data.get_labels()
    
    #print(len(labels))
    gnb = GaussianNB()
    gnb.fit(gene_exp, labels)
    predictions = []
    for exp in gene_exp:
        predictions.append(gnb.predict(exp)[0])
    #print(predictions)
    #print(len(predictions))
    print "Accuracy : %f" % evaluateClassifications(predictions,labels.tolist())

if __name__=="__main__":
    main()

