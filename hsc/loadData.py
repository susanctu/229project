from string import *
from sklearn import preprocessing
import numpy

class Data:

    FILE_LOC = '../../hscData/' #see dropbox folder called hscData for files
    ARRAY_ID_LEN = 9
    def __init__(self):
        self.geneNames = []
        self.cellTypes = []#numbers
        self.cellNameToCellType = {}
        self.isNormal = [] #boolean, is this array normal or not?
        self.arrayToCellType = {}
        self._make_arrayToCellType()
        self._make_cellTypes()

    def get_gene_names(self):
        if not self.geneNames:
            file = open(Data.FILE_LOC + 'genes.txt','r') 
            for line in file:
                self.geneNames.append(line.rstrip('\n'))#some of these might be ---
            file.close()
        return(self.geneNames)        

    def _make_arrayToCellType(self):
        broadFile = open(Data.FILE_LOC + 'BroadArrayList.txt')
        numTypes = 0
        for line in broadFile:
            lineParts = line.rstrip('\n').split('\t')
            if not lineParts[2] in self.cellNameToCellType:
                self.cellNameToCellType[lineParts[2]]=numTypes
                numTypes+=1
            self.arrayToCellType[lineParts[0]]=self.cellNameToCellType[lineParts[2]]
        broadFile.close()

    def _make_cellTypes(self): 
        file = open(Data.FILE_LOC + 'expression.txt','r')
        arrayNames = file.readline().rstrip('\n').split('\t')
        for arrayName in arrayNames:
            shortName = arrayName[0:Data.ARRAY_ID_LEN]
            if shortName in self.arrayToCellType:
                  self.cellTypes.append(self.arrayToCellType[shortName])
                  self.isNormal.append(True)
            else:
                self.isNormal.append(False)
        cellID = 0
        file.close()
 
    def get_labels(self):
        return(self.cellTypes)

    def get_gene_exp_matrix(self):
        expMatrix = [[0]*11927]*sum(self.isNormal)#initialize to be number of normal arrays by 11927 to avoid growing/copying lists
        file = open(Data.FILE_LOC + 'expression.txt','r')
        lineNum = -1 
        for line in file:
            if lineNum == -1:
                lineNum = lineNum + 1
                continue 
            line = line.rstrip('\n')
            lineParts = line.split('\t')
            assert(len(lineParts)==1841)#check that for this gene we have data from 1841 arrays
            normalLineParts =[]
            for i in range(0,len(lineParts)):
                if self.isNormal[i]:
                    normalLineParts.append(lineParts[i])  
            for expForCell,linePt in zip(expMatrix,normalLineParts):
                expForCell[lineNum]=float(linePt)
            lineNum = lineNum +1
        #print(expMatrix)
        return preprocessing.scale(numpy.array(expMatrix),axis=1) #normalize
         
def test():
    data = Data()
    geneExp = data.get_gene_exp_matrix()
    labels = data.get_labels()
    geneNames = data.get_gene_names()
    #print(len(geneExp))
    #print(len(geneExp[0]))
    #print(labels)

if __name__=="__main__":
    test() 
