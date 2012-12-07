from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from loadData import RaviNormal,RaviAML,Data

def classifyWithBest1vsAll(trainData,testData):

    train_X = trainData.get_gene_exp_matrix()
    train_y = trainData.get_labels()
    test_X = testData.get_gene_exp_matrix()
    test_y = testData.get_labels()

    bestSvm = LinearSVC(C=125,penalty='l1',dual=False)
    predictions = OneVsRestClassifier(bestSvm).fit(train_X,train_y).predict(test_X)
    print("predictions:")
    print(trainData.getCellNames(predictions))
    print("actual:")
    print(testData.getCellNames(test_y))
    print("zipped:")
    print(zip(trainData.getCellNames(predictions),testData.getCellNames(test_y)))

def classifyRaviNormal():
    b = Data()
    n = RaviNormal()
    classifyWithBest1vsAll(b,n) 

def classifyRaviAML():
    b = Data()
    aml = RaviAML()
    classifyWithBest1vsAll(b,aml)
 
if __name__=="__main__":
    classifyRaviAML()
