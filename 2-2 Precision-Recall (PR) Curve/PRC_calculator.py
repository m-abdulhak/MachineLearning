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
           key, val = line.split()
           observations.append((key,val)) 
    return observations

def getFileNameFromArguments():
    """ Returns the file names of TrainingDataInputs, TrainingDataOutputs 
        from from command-line arguments passed to the script at locations 1 and 2 """
    return str(sys.argv[1])

############################
### Calculate Parameters ###
############################

def trueClassCounts(obs):
    TruePositives = sum(map(lambda x : x[1]=='1', obs))
    TrueNegatives = len(obs)-TruePositives
    return TruePositives,TrueNegatives

################################################
###   Start Precision-Recall Curve Script    ###
################################################

# get file names of inputs and outputs files
observationsFile = getFileNameFromArguments()

observations = getObservations(observationsFile)

observations.sort(key=itemgetter(0))

print(trueClassCounts(observations))