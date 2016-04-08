
# coding: utf-8

# In[7]:

mywd="C:/Users/pcs/s/knn"
get_ipython().magic(u'cd {mywd}')


# In[8]:

from numpy import *
import operator
from os import listdir

from os.path import dirname


def createDataSet():       #create sample dataset
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):        #sample input ([0,0],group,labels,3)  
    dataSetSize=dataSet.shape[0]            # shape[0] : 4  shape[1] : 2 -> shape:array axis length
                                            # print shape : (4L,2L)
    
    diffMat=tile(inX,(dataSetSize,1))-dataSet    #tile(inX,(dataSetSize,1) : [[0 0][0 0][0 0][0 0]]
                                                #(tile(inX,(dataSetSize,1) inX*1 extend by dataSetSize
                                                 #(dataSetSize,1) : (4L, 1)
    sqDiffMat=diffMat**2               #pow Matirx
    sqDistances=sqDiffMat.sum(axis=1)#axis=1: x direction sum, axis=0: y direction sum
    distances=sqDistances**0.5   #pow(1/2)=sqrt(2) Matrix
    sortedDistIndicies=distances.argsort()   # return rank order
    classCount={}   #dictionary struct
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1   #classCount.get(voteIlabel,0) : if return val is null-> return 0
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    #classCount->sortedClassCount 
    #classCount.iteritems()  get iterator
    #operator.itemgetter(1) on 'key':val,  return val
    
    
    return sortedClassCount[0][0]  # return key of max Value

def file2matrix(filename):
    fr=open(filename)
    numberOfLines=len(fr.readlines())
    returnMat=zeros((numberOfLines,3))    # create matrix [ [0 0 0] [0 0 0] , .... ,[ 0 0 0] ]
    
    classLabelVector=[]  #create list
    
    fr=open(filename)
    index=0
    for line in fr.readlines():
        line=line.strip()    #remove whitespace two ends of str        
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        #classLabelVector.append(listFromLine[-1])   # =listFromLine[3]
        if (listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))    
        else:
            classLabelVector.append(listFromLine[-1])

        index+=1
    return returnMat,classLabelVector

def autoNorm(dataSet):        #(val-min) / (max-min)
    minVals=dataSet.min(0)   #arr
    maxVals=dataSet.max(0)   #arr
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]    
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        # normMat[i,:] only i-th row 
        # normMat[numTestVecs:m,:]  101~999 th row 
        print "the classifier came back with:%d, the real answer is: %d" % (classifierResult,datingLabels[i])
        if(classifierResult!=datingLabels[i]): errorCount+=1.0
    print "the total error rate is : %f" %(errorCount/float(numTestVecs))
    
def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(raw_input("percentage of time spent playing video games?"))
    ffMiles=float(raw_input("frequent flier miles earned per year?"))
    iceCream=float(raw_input("listers of ice cream consumed per year?"))
    datingDataMat,datingLabels=file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person: ",resultList[int(classifierResult)-1]
    
    
    
def img2vector(filename):
    returnVect=zeros((1,1024))    # 1024=32*32
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2Vector('trainingDigits/%s' % fileNameStr)
    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s' % fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print "the classifier came back with: %d, the real answer is: %d" %(classifierResult,classNumStr)
        if(classifierResult!=classNumStr): errorCount+=1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
    
    
#refernce 
# Copyright (c) 2007 David Cournapeau <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
#               2010 Olivier Grisel <olivier.grisel@ensta.org>

def loadIris(filename):    # iris: flower species
    module_path = dirname(__file__)
    with open(join(module_path, '', filename)) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = array(temp[2:])
        data = empty((n_samples, n_features))     
        target = empty((n_samples,), dtype=int)

        for i, ir in enumerate(data_file):
            data[i] = asarray(ir[:-1], dtype=float)   #points
            target[i] = asarray(ir[-1], dtype=int)    #label
            
    return data,target

def irisClassifier():
    trainingGroup,labels=loadIris("iris2.csv")    # 135 iris dataSet
    testGroup,solutions=loadIris("iristest.csv")  # 15 iris dataset for test
    errCount=0.0
    for i in range(len(testGroup)):
        result=classify0(testGroup[i,:],trainingGroup,labels,3)
        print "the classifier came back with: %d, the real answer is: %d" %(result,solutions[i])
        if(result!=solutions[i]): errorCount+=1.0
            
    print "\nthe total number of errors is: %d" % errCount
    print "\nthe total error rate is: %f" % (errCount/float(len(testGroup)))
    