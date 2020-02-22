import sys
import numpy as np
import operator
import matplotlib.pyplot as plt
from scipy.stats import spearmanr 
import random
import math
from operator import itemgetter
import matplotlib.pyplot as plt

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
    """ Returns the file name passed as argument at locations 1 """
    return str(sys.argv[1])

############################
###   Helper Functions   ###
############################

def intDiv(num1,num2):
    """ Integer Division OF num1 by num2 """
    return math.floor(num1/num2)

def getLastObsIndexWithConfidanceLessThan(obs,threshold):
    """ Returns the index of the last element that has a confidance levele less than the provided threshold """
    if(threshold<=obs[0][0]):
        return -1

    if(threshold>=obs[-1][0]):
        return len(obs)-1

    low = 0
    high = len(obs)-1
    index = intDiv(low+high,2)
    count = 0
    while(not(obs[index][0]<threshold and obs[index+1][0]>=threshold) and count<10000):
        #print("@",index,"Obs[index]=",obs[index][0],"Obs[index+1]=",obs[index+1][0],"Threshold=",threshold,"Condition:",(obs[index][0]<threshold and obs[index+1][0]>=threshold))
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
    """ Returns the count of the observations that are from calss 1 and class 0 """
    TruePositives = sum(map(lambda x : x[1]==1, obs))
    TrueNegatives = len(obs)-TruePositives
    return TruePositives,TrueNegatives

def setPredictions(obs,threshold):
    """ Sets the prediction value of the observations according to the provided threshold"""
    lastNegativeIndex = getLastObsIndexWithConfidanceLessThan(obs,threshold)
    for i in range(0,len(obs)):
        obs[i][2] = 1 if i>lastNegativeIndex else 0

    return obs

def calcParameters(obs):
    """ Calculates the True/False Positives/Negatives parameters of the provided observations (must have their prediction values set) """
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
    """ Calculates the precision/recall pairs for the provided observations and threshold step """
    threshold = thresholdStep
    precisionRecallThresholdList = []
    while threshold<1:
        obs = setPredictions(obs,threshold)
        TP, FP, FN, TN = calcParameters(obs)
        precision = np.float64(TP) / (TP + FP)
        recall = np.float64(TP) / (TP + FN)
        precisionRecallThresholdList.append((precision,recall,threshold))
        threshold += thresholdStep

    return precisionRecallThresholdList
        

################################################
###   Start Precision-Recall Curve Script    ###
################################################

# get file names of inputs and outputs files
observationsFile = getFileNameFromArguments()

# Extract Observations from file
obs = getObservations(observationsFile)

# Sort the observations according to their confidance levels
obs.sort(key=itemgetter(0))

# Calculate the precision/recall pairs
prPairs = calcPrecisionRecallPairs(obs,0.001)

# Get the current Axes instance on the current figure
ax = plt.gca()
#ax.set_xticks(range(0,1,0.05))
#ax.set_yticks(range(0,1,0.05))

# Plot the precision/recall pairs
Xs = list(map(lambda tup: tup[1],prPairs))
Ys = list(map(lambda tup: tup[0],prPairs))
plt.plot(Xs, Ys)
plt.savefig("PRC.png")

