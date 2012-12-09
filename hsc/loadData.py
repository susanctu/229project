from string import *
from sklearn import preprocessing
import numpy

class Loader: #FIXME: make the classes for the testing data inherit from this 
   
    """
        This is class helps you load data and get gene expression matrices and labels, but is not meant to be directly used. 
        See its subclasses. (Scroll down) 
    
        Sorry, the naming is not so great here (e.g., isNormal is left over from when this method was only used for picking out the normal arrays from the Broad data set) because this was previously only used for Broad data.
    """ 

    FILE_LOC = '../../hscData/' #see dropbox folder called hscData for files
    ARRAY_ID_LEN = 9
    NUM_GENES = 11927

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

    def getCellName(self,cellType):
        return(self.cellTypeToCellName[cellType])   

    def indices_of_celltype(self,celltype_id):
  	value = celltype_id
	qlist = self.get_labels().tolist()
    	indices = []
    	idx = -1
    	while True:
        	try:
            		idx = qlist.index(value, idx+1)
            		indices.append(idx)
        	except ValueError:
            		break
    	return(indices)

    def getCellNames(self,cellTypeList):
        """Pass your predicted cell type lists to this method to get back the cell type names"""
        return([self.cellTypeToCellName[cellType] for cellType in cellTypeList])
    
    def _make_cellTypeToCellName(self):#only call this if cellNameToCellType is already populated! overwritten by class for broad data
        """sets things up for getCellName and getCellNames"""
        for name, ctype in self.cellNameToCellType.items():
            self.cellTypeToCellName[ctype] = name
 
    def get_gene_names(self):
        """returns gene names, same order they are in in the feature vector"""
        if not self.geneNames:
            file = open(Loader.FILE_LOC + self.geneListFile,'r') 
            for line in file:
                self.geneNames.append(line.rstrip('\n'))#some of these might be ---
            file.close()
        return(self.geneNames)        

    def _make_arrayToCellType(self):
        """populates cellNameToCellType and arrayToCellType"""
        broadFile = open(Loader.FILE_LOC + self.arrayToTypeFile)
        numTypes = 0
        for line in broadFile:
            lineParts = line.rstrip('\n').split('\t')
            if not lineParts[self.cellTypeIdx] in self.cellNameToCellType:
                self.cellNameToCellType[lineParts[self.cellTypeIdx]]=numTypes
                numTypes+=1
            self.arrayToCellType[lineParts[0].rstrip(' ')]=self.cellNameToCellType[lineParts[self.cellTypeIdx]]#the rstrip here may seem weird, but if if I don't use it I can get extra space after the array name for ravi normal
        broadFile.close()

    def _make_cellTypes(self):
        """figures out which are the arrays that we want (i.e., the ones we have labels for) and makes the label vector"""
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
	#self.cellTypes = self.cellTypes[1:28]+self.cellTypes[30:]	
        return(numpy.array(self.cellTypes))

    def get_gene_exp_matrix(self):
        numNormal = sum(self.isNormal)#initialize to be number of normal arrays by 11927 to avoid growing/copying lists
        expMatrix = []
        for i in range(0,numNormal):
                expMatrix.append([0]*Loader.NUM_GENES)
        
        file = open(Loader.FILE_LOC + self.expFile,'r')
        lineNum = -1 
        for line in file:
            if lineNum == -1:#this block corresponds to skipping the column names
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
	#expMatrix = expMatrix[1:28] + expMatrix[30:]
        return preprocessing.scale(numpy.array(expMatrix),axis=1) #normalize
         
def test():
    data = Data()
    geneExp = data.get_gene_exp_matrix()
    labels = data.get_labels()
    geneNames = data.get_gene_names()

"""
    WARNING!: The three subclasses below all keep track of their own int -> cell type mappings (i.e., for RaviAML, 0 might be BM_CMP_BM_NBM07, but for Broad, 0 is BASO), so if you train with dataset A and try to classify with dataset B, you should use A's getCellNames method to get back the names of your predicted classes.
"""

class RaviAML(Loader):
    def __init__(self):
        Loader.__init__(self,"raviArrays.txt","genes.txt","raviAMLArrayList.txt",54,cellTypeIdx=1) 

class RaviNormal(Loader):
    def __init__(self):
        Loader.__init__(self,"raviArrays.txt","genes.txt","raviNormalArrayList.txt",54,cellTypeIdx=1) 

class Data(Loader):
    def __init__(self):
        Loader.__init__(self,"expression.txt","genes.txt","BroadArrayList.txt",5897,cellTypeIdx=2) 

    def _make_arrayToCellType(self):
        self.cellNameToCellType = {'DENDa2': 7, 'NKa2': 22, 'DENDa1': 6, 'TCEL1': 27, 'BCELLa2': 2, 'BCELLa3': 3, 'TCEL3': 29, 'ERY3-5': 11, 'ERY2': 10, 'TCEL4': 30, 'BCELLa4': 4, 'ERY1': 9, 'GRAN3': 15, 'GRAN2': 14, 'GRAN1': 13, 'BCELLa1': 1, 'NKa1': 21, 'TCEL7': 32, 'CMP': 5, 'PRE_BCELL2': 25, 'PRE_BCELL3': 26, 'EOS2': 8, 'HSC2': 17, 'HSC1': 16, 'TCEL6': 31, 'BASO': 0, 'GMP': 12, 'TCEL2': 28, 'TCEL8': 33, 'MONO2': 20, 'NKa3': 23, 'MONO1': 19, 'MEGA_MEP': 18, 'NKa4': 24}
        
        file = open(Loader.FILE_LOC + self.arrayToTypeFile)
        for line in file:
            lineParts = line.rstrip('\n').split('\t')
            cellName = lineParts[self.cellTypeIdx]
            if cellName in ['ERY3','ERY4','ERY5']:
                cellName = 'ERY3-5'
            elif cellName in ['MEGA1','MEGA2','MEP']:
                cellName = 'MEGA_MEP'
            self.arrayToCellType[lineParts[0]]=self.cellNameToCellType[cellName]
        file.close()

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
    print(n.getCellNames(labels))
    print(geneExp[0])
    print(geneExp[1])
    print(geneExp[2])
    print(len(geneExp))
    print(len(geneExp[0]))

if __name__=="__main__":
    #testBroad() 
    #testRaviAML()
    testRaviNormal()