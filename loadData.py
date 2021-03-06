from string import *
import glob
import os
from sklearn import preprocessing
import numpy
os.chdir("../projectFiles/")

class geneSignatures():

    def __init__(self):
        self.sourceToGenes = {}
        self._populateDict()

    def _populateDict(self):
        pass #FIXME:read in the file and populate sourceToGenes appropriately

    def get_genes(self):
        return self.sourceToGenes
    
    def get_gene(self,source):
        return self.sourceToGenes[source]        

class TCGAData():

    """for making matrix"""
    EXPRESSION_IDX = 2
    FILE_SUFFIX = "*gene_expression_analysis.txt"

    """for making array"""
    TUMOR = 1
    NORMAL = 0
    BARCODE_IDX =1
    SAMPLE_TYPE_IDX=6

    """for getting gene names"""
    GENE_NAME_IDX = 1
 
    def __init__(self):
        self.barcodeToIdx = {}
        self.geneNames=[] #should be in same order as in original files from Dill

    def _nonzero(self, num):
        if num == 0: 
            return 0
        else:
            return 1
    # gives a list of strings, each of which is the name of a gene
    def get_gene_names(self):
        if not self.geneNames: #if empty
            filename = glob.glob(TCGAData.FILE_SUFFIX)[0] #get any of the files in the glob FIXME: make this more efficient?   
            print(filename)
            file = open(filename,'r')
            lineNum = -1
            for line in file:    
                if lineNum == -1:
                    lineNum = lineNum +1
                    continue
                line = line.rstrip('\n')
                lineParts = line.split('\t')
                self.geneNames.append(lineParts[TCGAData.GENE_NAME_IDX])
                lineNum = lineNum + 1
            file.close()
        return self.geneNames

    def _nonzero(self, num):
        if num == 0:
            return 0
        else:
            return 1

    def get_gene_exp_matrix(self):
        expMatrix = []
        #genesMissingData = [0]*17814
        #peopleMissingData = [False]*599
        personNum = 0
        for filename in sorted(glob.glob(TCGAData.FILE_SUFFIX)):#
            file = open(filename,'r')
            perPersonCol = []#gene expression data for 1 person, should be in same order as in original files from Dill
            lineNum = -1
            for line in file:
                if lineNum == -1:
                    lineNum = lineNum +1
                    continue
                line = line.rstrip('\n')
                lineParts = line.split('\t')
                if (lineParts[TCGAData.EXPRESSION_IDX]=="null"):
                    #peopleMissingData[personNum]=True
                    #genesMissingData[lineNum] = genesMissingData[lineNum] + 1
                    perPersonCol.append('null')
                else:
                    perPersonCol.append(lineParts[TCGAData.EXPRESSION_IDX])
                lineNum = lineNum +1
            expMatrix.append(perPersonCol)
            file.close()
            personNum = personNum + 1
        #print(sum(genesMissingData))
        #genesMissingData[0]=self._nonzero(genesMissingData[0])
        #numGenesSometimesMissing = reduce(lambda x, y: x+self._nonzero(y), genesMissingData)
        #print(numGenesSometimesMissing)
        #print(sum(map(lambda x: 1 if x is True else 0,peopleMissingData)))#number of people with missing data
        geneAvgs = [0]*len(expMatrix[0])
        for i in range(0,len(expMatrix[0])):#loop over genes and average all the non-null values 
            expForGene = [person[i] for person in expMatrix]
            geneAvgs[i]=self._getAvgNonNull(expForGene)

        #now convert everything to floats
        for personExp in expMatrix:
            for i in range(0,len(personExp)):
                if personExp[i]=='null':
                    personExp[i]=geneAvgs[i]
                else:
                    personExp[i]=float(personExp[i])
        return preprocessing.scale(numpy.array(expMatrix),axis=1) #normalize

    def _getAvgNonNull(self,exp):
        sumNonNull = sum(map(lambda x: 0.0 if x=='null' else float(x),exp))
        numNonNull = sum(map(lambda x: 0.0 if x=='null' else 1.0,exp)) 
        avg = sumNonNull/numNonNull      
        return avg

    def get_labels(self): #as a list/array
        barcodeToLabels = {} #dict from barcode to label FIXME:do we want to keep track of this?
        file = open("UUID_NormalOrTumor.txt")
        for line in file:
            line = line.replace('"','')
            line=line.rstrip('\n')
            lineParts = line.split(',')
            if lineParts[TCGAData.SAMPLE_TYPE_IDX]=="Primary solid Tumor":
                barcodeToLabels[lineParts[TCGAData.BARCODE_IDX]]=TCGAData.TUMOR
            else:
                barcodeToLabels[lineParts[TCGAData.BARCODE_IDX]]=TCGAData.NORMAL
        file.close()
        labelList = []
        index = 0
        for barcode in sorted(barcodeToLabels.keys()):
            labelList.append(barcodeToLabels[barcode])
            self.barcodeToIdx[barcode]=index
            index = index + 1
        return numpy.array(labelList)
    
    def get_index(self,barcode):#FIXME: this only works if get_labels was called first
        return self.barcodeToIdx[barcode]

def testCode():
        data = TCGAData()
        gene_exp = data.get_gene_exp_matrix()
        labels = data.get_labels()
        length = len(gene_exp[0])
        #print(length)
        for person_exp in gene_exp:
            if len(person_exp)!=length:
                print(len(person_exp))
            assert(len(person_exp)==length)
        print(len(labels))
        assert(len(labels)==len(gene_exp))
        index = data.get_index('TCGA-E2-A1BD-01A-11R-A12P-07')
        print(gene_exp[index])
        print(len(gene_exp[index]))
        #print(labels[index])
        #index = data.get_index('TCGA-E2-A15E-01A-11R-A12D-07')
        #print(gene_exp[index])
        #print(labels[index])
        index = data.get_index('TCGA-BH-A0DP-11A-12R-A089-07')
        #print(gene_exp[index])
        #print(labels[index])

        print(labels)
