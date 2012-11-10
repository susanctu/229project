from string import *
import glob
import os
os.chdir("../projectFiles/")

class TCGAData():

    """for making matrix"""
    EXPRESSION_IDX = 2
    FILE_SUFFIX = "*gene_expression_analysis.txt"

    """for making array"""
    TUMOR = 1
    NORMAL = 0
    BARCODE_IDX =1
    SAMPLE_TYPE_IDX=6     

    """TODO: check that list of genes for each person are exactly same (only checked same length for now), and normalize gene expression numbers"""

    def __init__(self):
        self.barcodeToIdx = {}

    def get_gene_exp_matrix(self):
        matrix = []
        for filename in sorted(glob.glob(TCGAData.FILE_SUFFIX)):#
            file = open(filename,'r')
            perPersonCol = []#gene expression data for 1 person, should be in same order as in original files from Dill 
            for line in file:
                line = line.rstrip('\n')
                lineParts = line.split('\t')   
                perPersonCol.append(lineParts[TCGAData.EXPRESSION_IDX])
            matrix.append(perPersonCol)
            file.close()
        return matrix

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
        return labelList 
    
    def get_index(self,barcode):#FIXME: this only works if get_labels was called first
        return self.barcodeToIdx[barcode]

def testCode():
        data = TCGAData()
        gene_exp = data.get_gene_exp_matrix()
        labels = data.get_labels()
        length = len(gene_exp[0])
        print(length)
        for person_exp in gene_exp:
            if len(person_exp)!=length:
                print(len(person_exp))
            assert(len(person_exp)==length)
        print(len(labels))
        print(len(gene_exp))
        assert(len(labels)==len(gene_exp))
        index = data.get_index('TCGA-E2-A1BD-01A-11R-A12P-07')        
        print(gene_exp[index])
        print(labels[index])
        #index = data.get_index('TCGA-E2-A15E-01A-11R-A12D-07')        
        #print(gene_exp[index])
        #print(labels[index])
        index = data.get_index('TCGA-BH-A0DP-11A-12R-A089-07')        
        print(gene_exp[index])
        print(labels[index])

        print(labels)
