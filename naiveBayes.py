from sklearn.naive_bayes import GNB
from loadData import TCGAData

def main():
    data = TCGAData()
    gene_exp = data.get_gene_exp_matrix()
    labels = data.get_labels()

    gnb = GNB()
    gnb.fit(gene_exp, labels)
    map(lambda x: x+1 ,gene_exp)
    predictions = []
    for exp in gene_exp:
        print(predictions)
        print(exp)
        predictions = predictions + gnb.predict(exp)
    print "Number of mislabeled points : %d" % (labels != predictions).sum()

if __name__=="__main__":
    main()

