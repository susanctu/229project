from string import *
from sklearn import preprocessing
import numpy

class Data:

    FILE_LOC = '../../hscData/' #see dropbox folder called hscData for files
    def __init__(self):
        self.geneNames = []
        self.cellTypes = []

    def get_gene_names(self):
        if not self.geneNames:
            file = open(Data.FILE_LOC + 'genes.txt','r') 
            for line in file:
                self.geneNames.append(line.rstrip('\n'))#some of these might be ---
        return(self.geneNames)        

    def get_labels(self):
        if not self.cellTypes:
            file = open(Data.FILE_LOC + 'meanExp.txt','r')
            self.cellTypes =file.readline().rstrip('\n').split(' ')
        return(self.cellTypes)

    def get_gene_exp_matrix(self):
        expMatrix = [[0]*11927]*38#initialize to be 38 by 11927 to avoid growing/copying lists
        file = open(Data.FILE_LOC + 'meanExp.txt','r')
        lineNum = -1 
        for line in file:
            if lineNum == -1:
                lineNum = lineNum + 1
                continue 
            line = line.rstrip('\n')
            lineParts = line.split(' ')
            assert(len(lineParts)==38)#check that for this gene we have expression data for all 38 types of cells
            for expForCell,linePt in zip(expMatrix,lineParts):
                expForCell[lineNum]=float(linePt)
            lineNum = lineNum +1
        return preprocessing.scale(numpy.array(expMatrix),axis=1) #normalize
         
def test():
    data = Data()
    geneExp = data.get_gene_exp_matrix()
    labels = data.get_labels()
    geneNames = data.get_gene_names()
    #print(geneExp)
    #print(labels)

if __name__=="__main__":
    test() 
