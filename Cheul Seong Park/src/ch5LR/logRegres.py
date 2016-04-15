
# coding: utf-8

# In[ ]:

from numpy import *

def loadDataSet():
    dataMat=[];labelMat=[]   
    fr=open('testSet.txt')    # txt file having X1,X2,label line by line 
    #length of line=100
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])   #dataMat+=[X0,X1,X2]
        labelMat.append(int(lineArr[2]))  #add label
    return dataMat,labelMat    

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)  #m=100, n=3 in this example
    #shape's return values are length of col and row in matrix
    alpha=0.001
    maxCycles=500
    weights=ones((n,1)) #return nx1 matrix having value 1
    
    for k in range(maxCycles):   #calc for 500
        h=sigmoid(dataMatrix*weights)   #input matrix, output matrix
        
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error #weights calc for 300(=matrix length )
        #3x100 matrix and 100x1 error matrix give rise to 3x1 matrix
       
    return weights

def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights=wei
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[]; ycord1=[]
    xcord2=[]; ycord2=[]
    for i in range(n):   #to classify
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)   #range for first f(x)
    y=(-weights[0]-weights[1]*x)/weights[2]   #f(x) depends on weights
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()
    
def stocGradAscent0(dataMatrix,classLabels):
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix,classLabels,numIter=150):   #parameter defualt
    m,n=shape(dataMatrix)
    
    weights=ones(n)
    for j in range(numIter):
        dataIndex=range(m)
        for i in range(m):
            
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex))) #one of 0~100
            h=sigmoid(sum(dataMatrix[i]*weights))
            error=classLabels[i]-h
            weights=weights+alpha*error*dataMatrix[i]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:return 1.0
    return 0.0

def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[]; trainingLabels=[]
    #training
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights=stocGradAscent1(array(trainingSet),trainingLabels,500)
    #training
    #test
    errorCount=0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    #test
    print "the error rate of this test is: %f" %errorRate
    
    return errorRate

def multiTest():
    numTests=10;errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print "after %d iterations the average error rate is: %f"%(numTests,errorSum/float(numTests))
def loadDataFromTxt(filename):
    frTrain=open(filename)
    trainingSet=[]; trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[len(currLine)-1]))
    return trainingSet,trainingLabels

def logRegresTest(weights,filename):
    frTest=open(filename)
    errorCount=0;numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(len(currLine)-1):
            #if(float(currLine[i]).isdigit()):
            lineArr.append(float(currLine[i]))
            
        if int(classifyVector(array(lineArr),weights))!=int(currLine[len(currLine)-1]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" %errorRate
    
    return errorRate





        
        
        
        
        
                   

