import sys
import numpy as np
import operator
import matplotlib.pyplot as plt
from scipy.stats import spearmanr 
import random
import math
from operator import itemgetter

######################
### Handling files ###
######################

def getObservations(fileName):
    """ Retrieves the data inputs from the provided file """
    observations = []
    with open(fileName) as f:
        for line in f:
           conf, RealClass = line.split()
           observations.append([float(conf),int(RealClass),0]) 
    return observations

def getFileNameFromArguments():
    """ Returns the file names of TrainingDataInputs, TrainingDataOutputs 
        from from command-line arguments passed to the script at locations 1 and 2 """
    return str(sys.argv[1])

############################
###   Helper Functions   ###
############################

def intDiv(num1,num2):
    return math.floor(num1/num2)

def getLastObsIndexWithConfidanceLessThan(obs,threshold):
    if(threshold<=obs[0][0]):
        return -1

    if(threshold>=obs[-1][0]):
        return len(obs)-1

    low = 0
    high = len(obs)-1
    index = intDiv(low+high,2)
    count = 0
    while(not(obs[index][0]<threshold and obs[index+1][0]>=threshold) and count<10000):
        print("@",index,"Obs[index]=",obs[index][0],"Obs[index+1]=",obs[index+1][0],"Threshold=",threshold,"Condition:",(obs[index][0]<threshold and obs[index+1][0]>=threshold))
        if(obs[index][0]>=threshold):
            high = index
        else:
            low = index
        index = intDiv((low+high),2)
        count += 1

    return index

############################
### Calculate Parameters ###
############################

def trueClassCounts(obs):
    TruePositives = sum(map(lambda x : x[1]==1, obs))
    TrueNegatives = len(obs)-TruePositives
    return TruePositives,TrueNegatives

def setPredictions(obs,threshold):
    lastNegativeIndex = getLastObsIndexWithConfidanceLessThan(obs,threshold)
    for i in range(0,len(obs)):
        obs[i][2] = 1 if i>lastNegativeIndex else 0

    return obs

def calcParameters(obs):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(0,len(obs)):
        trueClass = obs[i][1]
        predictedClass = obs[i][2]
        if predictedClass == 1:
            if trueClass == 1: 
                TP += 1
            else:
                FP +=1
        else:
            if trueClass == 1:
                FN += 1
            else:
                TN += 1

    return TP, FP, FN, TN

def calcPrecisionRecallPairs(obs,thresholdStep):
    threshold = 0
    recallPrecisionThresholdList = []
    while threshold<1:
        obs = setPredictions(obs,threshold)
        TP, FP, FN, TN = calcParameters(obs)
        precision = np.float64(TP) / (TP + FP)
        recall = np.float64(TP) / (TP + FN)
        recallPrecisionThresholdList.append((precision,recall,threshold))
        threshold += thresholdStep

    return recallPrecisionThresholdList
        

################################################
###   Start Precision-Recall Curve Script    ###
################################################

# get file names of inputs and outputs files
observationsFile = getFileNameFromArguments()

obs = getObservations(observationsFile)

obs.sort(key=itemgetter(0))

print(len(obs),trueClassCounts(obs))

threshold = 0.5

setPredictions(obs,threshold)
#print("@",index,"Obs[index]=",obs[index][0],"Obs[index+1]=",obs[index+1][0],"Threshold=",threshold,"Condition:",(obs[index][0]<threshold and obs[index+1][0]>=threshold))


print(calcParameters(obs),sum(calcParameters(obs)))

print(calcPrecisionRecallPairs(obs,0.05))