
# coding: utf-8

# In[ ]:



from math import log
import operator

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts={}  # dict struct
    for featVec in dataSet:    #counting loop to dictinory
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt =0.0
    #sample {'yes': 2, 'no': 3}

    for key in labelCounts:  #entropy calc loop
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)     # log_2 prob
    return shannonEnt



def createDataSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels


# append a=[1,2,3] b=[4,5,6] a.append(b) -> a=[1,2,3,[4,5,6]]
# extend a=[1,2,3] b=[4,5,6] a.extend(b) -> a=[1,2,3,4,5,6]

def splitDataSet(dataSet,axis,value):
    retDataSet=[]                # due to call-by reference in python, new create list
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]  #list add 0~axis
          
            reducedFeatVec.extend(featVec[axis+1:])  #list add asix+1~
            retDataSet.append(reducedFeatVec)  #list add list
    return retDataSet   #return dataSet without axis element

def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1     # return val 0 or 1 in sample
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0; bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)   # struct set : not allow to duplicate element
        
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)     # i=value split
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy      # the larger infoGain, the smaller newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeatures=i
    return bestFeatures #return index of the smallest entropy


def majorityCnt(classList): #return the element having maximum number of element in classList
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
             
    
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    
    if classList.count(classList[0])==len(classList): # all elements in list are same
        return classList[0]
    if len(dataSet[0])==1:     # length of list ==1        
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]    # bestFeatLabel = no surfacing or flippers in sample
    myTree={bestFeatLabel:{}}    # format : json 
    
    del(labels[bestFeat])    # remove element in labels
    
    featValues=[example[bestFeat] for example in dataSet]
    
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)  #recusive - call
    return myTree    # format : tree(json)

def classify(inputTree,featLabels,testVec):
    firstStr=inputTree.keys()[0]
    secondDict=inputTree[firstStr]
    
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':     #if not leaf node -> recusive
                classLabel=classify(secondDict[key],featLabels,testVec)
            else: classLabel=secondDict[key]     # if leaf node-> return 
    return classLabel


def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr=open(filename)
    return pickle.load(fr)