from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier,OutputCodeClassifier
from sklearn.svm import LinearSVC
from loadData import RaviNormal,RaviAML,AML_easy,GSE_H60,GSE_K562,Data

def classifyWithBestECOC(trainData,testData):

    train_X = trainData.get_gene_exp_matrix()
    train_y = trainData.get_labels()
    test_X = testData.get_gene_exp_matrix()
    test_y = testData.get_labels()

    bestSvm = LinearSVC(C=125,penalty='l1',dual=False)
    predictions = OutputCodeClassifier(bestSvm,code_size=52).fit(train_X,train_y).predict(test_X)
    print("ecoc --------------------")
    print("predictions:")
    print(trainData.getCellNames(predictions))
    print("actual:")
    print(testData.getCellNames(test_y))
    print("zipped:")
    print(zip(trainData.getCellNames(predictions),testData.getCellNames(test_y)))

def classifyWithBest1vs1(trainData,testData):

    train_X = trainData.get_gene_exp_matrix()
    train_y = trainData.get_labels()
    test_X = testData.get_gene_exp_matrix()
    test_y = testData.get_labels()

    bestSvm = LinearSVC(C=125,penalty='l1',dual=False)
    predictions = OneVsRestClassifier(bestSvm).fit(train_X,train_y).predict(test_X)
    print("one versus one--------------------")
    print("predictions:")
    print(trainData.getCellNames(predictions))
    print("actual:")
    print(testData.getCellNames(test_y))
    print("zipped:")
    print(zip(trainData.getCellNames(predictions),testData.getCellNames(test_y)))

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
    classifyWithBest1vs1(b,n) 

def classifyRaviAML():
    b = Data()
    aml = RaviAML()
    classifyWithBest1vs1(b,aml)

def classifyAMLEasy():
    b = Data()
    amlEasy = AML_easy()
    classifyWithBestECOC(b,amlEasy) 

def classifyH60():
    b = Data()
    h = GSE_H60()
    classifyWithBestECOC(b,h) 

def classifyK562():
    b = Data()
    k = GSE_K562()
    classifyWithBestECOC(b,k) 

if __name__=="__main__":
    classifyAMLEasy()
    classifyH60()
    classifyK562()
