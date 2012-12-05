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
        self.cellTypeToCellName = {}
        self.isNormal = [] #boolean, is this array normal or not?
        self.arrayToCellType = {}
        self._make_arrayToCellType()
        self._make_cellTypes()
        self._make_cellTypeToCellName()

    def _make_cellTypeToCellName(self):#only call this if cellNameToCellType is already populated!
        for name, ctype in self.cellNameToCellType.items():
            self.cellTypeToCellName[ctype] = name
   
    def getCellName(self,cellType):
        return(self.cellTypeToCellName[cellType])

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
        file.close()
 
    def get_labels(self):
        return(numpy.array(self.cellTypes))

    def get_gene_exp_matrix(self):
        numNormal = sum(self.isNormal)#initialize to be number of normal arrays by 11927 to avoid growing/copying lists
        expMatrix = []
        for i in range(0,numNormal):
                expMatrix.append([0]*11927)
        
        file = open(Data.FILE_LOC + 'expression.txt','r')
        lineNum = -1 
        for line in file:
            if lineNum == -1:
                lineNum = lineNum + 1
                continue 
            line = line.rstrip('\n')
            lineParts = line.split('\t')
            #print(len(lineParts))
            assert(len(lineParts)==1841)#check that for this gene we have data from 1841 arrays
            normalLineParts =[]
            for i in range(0,len(lineParts)):
                if self.isNormal[i]:
                    normalLineParts.append(lineParts[i]) 
            for i in range(0,len(normalLineParts)):
                expMatrix[i][lineNum]=float(normalLineParts[i])
            lineNum = lineNum +1
        file.close()
        #print(expMatrix)
        return preprocessing.scale(numpy.array(expMatrix),axis=1) #normalize
         
def test():
    data = Data()
    geneExp = data.get_gene_exp_matrix()
    labels = data.get_labels()
    geneNames = data.get_gene_names()
    #print(len(geneExp))
    #print(len(geneExp[0]))
    #print(geneExp[0].tolist())
    #print(geneExp[1])
    print([data.getCellName(label) for label in labels])



if __name__=="__main__":
    test() 
