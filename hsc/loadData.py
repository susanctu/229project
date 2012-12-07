from string import *
from sklearn import preprocessing
import numpy

class Loader: #FIXME: make the classes for the testing data inherit from this 
    
    FILE_LOC = '../../hscData/' #see dropbox folder called hscData for files
    ARRAY_ID_LEN = 9
    NUM_GENES = 11927

    """Decided to hard-code this because had to modify classes a little by hand anyways, and need to be able to this same mapping for all data sets"""
    BROAD_CELL_NAME_TO_TYPE = {0: 'BASO', 1: 'BCELLa1', 2: 'BCELLa2', 3: 'BCELLa3', 4: 'BCELLa4', 5: 'CMP', 6: 'DENDa1', 7: 'DENDa2', 8: 'EOS2', 9: 'ERY1', 10: 'ERY2', 11: 'ERY3', 11: 'ERY4', 11: 'ERY5', 12: 'GMP', 13: 'GRAN1', 14: 'GRAN2', 15: 'GRAN3', 16: 'HSC1', 17: 'HSC2', 18: 'MEGA1', 18: 'MEGA2', 18: 'MEP', 19: 'MONO1', 20: 'MONO2', 21: 'NKa1', 22: 'NKa2', 23: 'NKa3', 24: 'NKa4', 25: 'PRE_BCELL2', 26: 'PRE_BCELL3', 27: 'TCEL1', 28: 'TCEL2', 29: 'TCEL3', 30: 'TCEL4', 31: 'TCEL6', 32: 'TCEL7', 33: 'TCEL8'}

    def __init__(self,expFile,geneListFile,arrayToTypeFile,expectedNumArrays,cellTypeIdx=2):
        self.expFile = expFile
        self.geneListFile = geneListFile
        self.arrayToTypeFile = arrayToTypeFile
        self.cellTypeIdx = cellTypeIdx #which column the cell type is in in arrayToTypeFile
        self.expectedNumArrays = expectedNumArrays #we're going to use this later, in making the matrix, to check that we don't have missing vals
        
        self.geneNames = []
        self.cellTypes = []
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
            file = open(Loader.FILE_LOC + self.geneListFile,'r') 
            for line in file:
                self.geneNames.append(line.rstrip('\n'))#some of these might be ---
            file.close()
        return(self.geneNames)        

    def _make_arrayToCellType(self):#populates cellNameToCellType and arrayToCellType
        broadFile = open(Loader.FILE_LOC + self.arrayToTypeFile)
        numTypes = 0
        for line in broadFile:
            lineParts = line.rstrip('\n').split('\t')
            if not lineParts[self.cellTypeIdx] in self.cellNameToCellType:
                self.cellNameToCellType[lineParts[self.cellTypeIdx]]=numTypes
                numTypes+=1
            self.arrayToCellType[lineParts[0]]=self.cellNameToCellType[lineParts[self.cellTypeIdx]]
        broadFile.close()

    def _make_cellTypes(self): 
        file = open(Loader.FILE_LOC + self.expFile,'r')
        arrayNames = file.readline().rstrip('\n').split()
        for arrayName in arrayNames:
            shortName = arrayName[0:Loader.ARRAY_ID_LEN]
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
        print(numNormal)
        expMatrix = []
        for i in range(0,numNormal):
                expMatrix.append([0]*Loader.NUM_GENES)
        
        file = open(Loader.FILE_LOC + self.expFile,'r')
        lineNum = -1 
        for line in file:
            if lineNum == -1:
                lineNum = lineNum + 1
                continue 
            line = line.rstrip('\n')
            lineParts = line.split()
            assert(len(lineParts)==self.expectedNumArrays)#check that for this gene we have data from 1841 arrays
            normalLineParts =[]
            for i in range(0,len(lineParts)):
                if self.isNormal[i]:
                    normalLineParts.append(lineParts[i]) 
            for i in range(0,len(normalLineParts)):
                expMatrix[i][lineNum]=float(normalLineParts[i])
            lineNum = lineNum +1
        file.close()
        return preprocessing.scale(numpy.array(expMatrix),axis=1) #normalize
         
def test():
    data = Data()
    geneExp = data.get_gene_exp_matrix()
    labels = data.get_labels()
    geneNames = data.get_gene_names()

class RaviAML(Loader):
    def __init__(self):
        Loader.__init__(self,"raviArrays.txt","genes.txt","raviAMLArrayList.txt",54,cellTypeIdx=1) 

class RaviNormal(Loader):
    def __init__(self):
        Loader.__init__(self,"raviArrays.txt","genes.txt","raviNormalArrayList.txt",54,cellTypeIdx=1) 

class Data(Loader):
    def __init__(self):
        Loader.__init__(self,"expression.txt","genes.txt","BroadArrayList.txt",5897,cellTypeIdx=2) 

def testBroad():
    data = Data()
    geneExp = data.get_gene_exp_matrix()
    labels = data.get_labels()
    geneNames = data.get_gene_names()
    print(labels)
    print(geneExp[0])
    print(geneExp[5])
    print(geneExp[17])
    print(len(geneExp))
    print(len(geneExp[0]))

def testRaviAML():
    aml = RaviAML()
    geneExp = aml.get_gene_exp_matrix()
    labels = aml.get_labels()
    geneNames = aml.get_gene_names()
    print(labels)
    print(geneExp[0])
    print(geneExp[1])
    print(geneExp[2])
    print(len(geneExp))
    print(len(geneExp[0]))

def testRaviNormal():
    n = RaviNormal()
    geneExp = n.get_gene_exp_matrix()
    labels = n.get_labels()
    geneNames = n.get_gene_names()
    print(labels)
    print(geneExp[0])
    print(geneExp[1])
    print(geneExp[2])
    print(len(geneExp))
    print(len(geneExp[0]))

if __name__=="__main__":
    testBroad() 
    #testRaviAML()
    #testRaviNormal()
