from sklearn.naive_bayes import GaussianNB
from loadData import TCGAData
from testUtils import kFoldCrossValid

def main():
    data = TCGAData()
    gene_exp = data.get_gene_exp_matrix()
    labels = data.get_labels()
    
    gnb = GaussianNB()
    accuracy = kFoldCrossValid(gene_exp,labels,gnb)     
    print(accuracy)

if __name__=="__main__":
    main()

