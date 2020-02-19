import sys
import numpy as np
import operator
import matplotlib.pyplot as plt
from scipy.stats import spearmanr 

######################
### Handling files ###
######################

def getDataInputs(fileName):
    """ Retrieves the data inputs from the provided file """
    dataInputs = {}
    with open(fileName) as f:
        for line in f:
           (key, val) = line.split()
           dataInputs[key] = val
    return dataInputs

def getTrainingDataOutputs(fileName):
    """ Retrieves the data outputs from the provided file"""

    trainingDataOutputs = {}
    headerArray = []
    with open(fileName) as f:
        for lineIndx,line in enumerate(f):
            if(lineIndx == 0):
                headerArray = line.split()
                headerArray =  cleanSeqNames(headerArray)
            else:
                lineArray = line.split()
                for colIndx,value in enumerate(lineArray):
                    if(not(headerArray[colIndx] in trainingDataOutputs.keys())):
                        trainingDataOutputs[headerArray[colIndx]] = []
                    trainingDataOutputs[headerArray[colIndx]].append(float(value))
    return trainingDataOutputs

def cleanSeqNames(seqNamesArray):
    """ Removes ' and \" characters from the start and end of a sequence name """
    cleanedSeqNames = []
    for seqName in seqNamesArray:
        cleanedSeqName = seqName[1:] if seqName[0]=="'" or seqName[0]=="\"" else seqName
        cleanedSeqName = cleanedSeqName[:-1] if cleanedSeqName[-1]=="'" or cleanedSeqName[-1]=="\"" else cleanedSeqName
        cleanedSeqNames.append(cleanedSeqName)
    return cleanedSeqNames

def getFileNamesFromArguments():
    """ Returns the file names of TrainingDataInputs, TrainingDataOutputs, and UnseesSequences 
        from from command-line arguments passed to the script at locations 1,2, and 3 """
    return str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])

def getTrainingSet(inputsFileName,outputsFileName):
    """ Returns the training data set containing the trainig inputs and outputs extracted from the selected files
        in the form of a dictionary object with 'inputs' and 'outputs' keys """

    # Input data is a dictionry with key, values pairs of sequence names and sequence values
    # Example: {'Alx3': 'RRNRTTFSTFQLEELEKVFQKTHYPDVYAREQLALRTDLTEARVQVWFQNRRAKWRK', ...}
    trainingDataInputs = getDataInputs(inputsFileName)

    # Output data is a dictionry with key, values pairs of sequence names and output vectors
    # Example: {'Alx3': [1.1,3.01,2,0.44,..........................], ...}
    trainingDataOutputs = getTrainingDataOutputs(outputsFileName)

    return {'inputs':trainingDataInputs,'outputs':trainingDataOutputs}

def writeToFile(outputDataSet):
    """ Write the output data set to a file """
    # Set output file name
    outputFileName = "output.txt"

    # Open output file to start writing
    with open(outputFileName, 'w') as f:
        # Get list of the sequence names of all output instances
        outputDataSetList = list(outputDataSet)

        # Write sequence names of all output instances in output file header row
        f.write('\t'.join(outputDataSetList))
        # End of header line
        f.write('\n')

        # Get length of output vector generated for each instance - 1 for rows loop indexing 
        lastIndexInOutputVector =  len(outputDataSet[outputDataSetList[0]])

        # For each row in an output vector
        for i in range(0,lastIndexInOutputVector):     
            # On line i: for each sequence in sequence list 
            for j in outputDataSetList:
                # Write the i-th value in its output vector + tab seperator
                f.write(str(np.around(outputDataSet[j] [i], decimals=4)) + '\t')

            # End of line
            f.write('\n')

######################
### KNN Functions  ###
######################

def performKNNUsingFiles(k,inputsFileName,outputsFileName, unseenInstancesFileName):
    """ Performs the K Nearest Neighbor algorithm using the provided K and the training data provided in the inputs and outputs files
        on the instances provided in the unseen instances file, and saves the resulting output vectors in an 'output.txt' file """
    
    # Only used when trying to find the best K
    allCoef = []
    allP = []

    # Get the training data set from the files
    # DataSet is a dictionary with 'inputs', 'outputs' keys
    trainingObservations = getTrainingSet(inputsFileName,outputsFileName)

    # Get the unseen instances from the files
    # Example: {'Alx3': 'RRNRTTFSTFQLEELEKVFQKTHYPDVYAREQLALRTDLTEARVQVWFQNRRAKWRK', ...}
    unseenInstances = getDataInputs(unseenInstancesFileName)

    # Define output dataset containing the names and calculated output vectors for all unseen instances 
    # Example: {'Alx3': [1.1,3.01,2,0.44,..........................], ...}
    outputDataSet = {}

    # Calculate output for each unseen instance
    for instanceName in unseenInstances:
        outputDataSet[instanceName] = calculateOutputVectorUsingKNN(k,unseenInstances[instanceName],trainingObservations)
        
        # For performance testing, when trying to find the best k 
        # Real output values of unseen instances must exist in training dataset for comparison (but wont be used during training)
        if(testPerformanceOverKRange==1):
            iCoef,iP = spearmanr(outputDataSet[instanceName],trainingObservations['outputs'][instanceName])
            allCoef.append(iCoef)
            allP.append(iP)

    writeToFile(outputDataSet)

    if(testPerformanceOverKRange == 1):
        return np.mean(allCoef), np.mean(allP)
    else:
        return -1,-1

def calculateOutputVectorUsingKNN(k,newInstance,trainingObservations):
    # Calculate the distances between the new instance and all instances in the training set
    distances = calcInstanceObsDistances(newInstance,trainingObservations)

    # Find the k Nearest Neighbors corresponding to the k minimum distances 
    kNearestNeighbors = getSeqOfKMinDistances(k,distances)
    displayDistances(distances,kNearestNeighbors)
    
    # Generate the predicted output for the new instance as the mean of the output vector of the 
    # k nearest neighbors corresponding to the k minimum distances
    instanceOutput = generateInstanceOutput(kNearestNeighbors,trainingObservations['outputs'])
    displayOutputVectors(instanceOutput,trainingObservations['outputs'],kNearestNeighbors)

    return instanceOutput

def calcInstanceObsDistances(xi,obs):
    """ Calculate the distances between the new instance Xi and all instances in the training set defined in 'obs'
        Returns the a dictionary(sequence_name,distance) of the calculated sequences  """
    distances = {}

    for indx, o in enumerate(obs['inputs']):
        distances[o] = calcSpecialDistance(xi,obs['inputs'][o])

    return distances

def calcDistance(x1,x2):
    """ Calculates the Hamming distance between 2 strings as the Number of non-identical characters in the 2 sequences """
    length = np.minimum(len(x1),len(x2))

    distance = 0

    for i in range(0,length):
        distance += 1 if x1[i] != x2[i] else 0
    
    return distance

residuesList = [3, 5, 6, 25, 31, 44, 46, 47, 48, 50, 51, 53, 54, 55, 57]

def calcSpecialDistance(x1,x2):
    """ Calculates the distance between 2 sequences as the Number of non-identical characters over a set of 15 DNA-contacting amino acids.
        These 15 residues are @ 3, 5, 6, 25, 31, 44, 46, 47, 48, 50, 51, 53, 54, 55 and 57 """

    length = np.minimum(len(x1),len(x2))

    distance = 0

    for i in range(0,length):
        distance += 1 if x1[i] != x2[i] and i+1 in residuesList else 0
    
    return distance

def getSeqOfKMinDistances(k,distanceDict):
    """ Returns the a dictionary(sequence_name,distance) corresponding to the k minimum distances found in 'distanceDict' """

    # the k minimum distances found in distanceDict
    kMinDistances = {}
    # the maximum distance within the list 
    maxDistInKMin = 0
    # the key of the maximum distance within the list
    keyOfMaxDistInKMin = 0
    
    for indx,x in enumerate(distanceDict):
        if(len(kMinDistances)<k):
            if(len(kMinDistances)==0 or distanceDict[x]>maxDistInKMin):
                maxDistInKMin = distanceDict[x]
                keyOfMaxDistInKMin = x
            kMinDistances[x] = distanceDict[x]
        else:
            if(distanceDict[x]<maxDistInKMin):
                kMinDistances.pop(keyOfMaxDistInKMin)
                kMinDistances[x] = distanceDict[x]
                keyOfMaxDistInKMin = max(kMinDistances.items(), key=operator.itemgetter(1))[0]
                maxDistInKMin = distanceDict[keyOfMaxDistInKMin]
    return kMinDistances

def generateInstanceOutput(kNearestNeighbors,trainingDataOutputs):
    """ Calculate the predicted output as the mean of the output vectors of the k nearest neighbors"""
    output = [0] * len(trainingDataOutputs[list(kNearestNeighbors)[0]])
    k = len(kNearestNeighbors)

    for seq in kNearestNeighbors:
        for i in range(0,len(trainingDataOutputs[seq])):
            output[i] += trainingDataOutputs[seq][i] / k
            #print(trainingDataOutputs[seq][i])

    return output

################################
### Visulaization Functions  ###
################################

def displayDistances(distances,kNearestNeighbors):
    """ Displays all distances in gray and the selected k nearest distances in red"""
    if(displayGraphs):
        # Get the current Axes instance on the current figure
        ax = plt.gca()

        for distance in distances:
            plt.plot(0,distances[distance], marker='o', markersize=2, color = 'lightgrey')
        
        for distance in kNearestNeighbors:
            plt.plot(0,kNearestNeighbors[distance], marker='o', markersize=2, color = 'red')

        plt.show()

def displayOutputVectors(outputVector,trainingOutputs,kNearestNeighbors):
    """ Dispalys the full training set output vectors (in grey), the k nearest neighbors output vectors (in green)
        and the predicted new instance output vector (in red) """
    if(displayGraphs):
        # Get the current Axes instance on the current figure
        ax = plt.gca()

        # number of output points to show
        n = 1000
        
        if(displayGraphs):
            for seq in trainingOutputs:
                plt.plot(trainingOutputs[seq][0:n],'lightgrey')
            for seq in kNearestNeighbors:
                plt.plot(trainingOutputs[seq][0:n],'green')
            plt.plot(outputVector[0:n],'red')
            plt.show()


################################
###     Start of Script      ###
################################

# Set whether to display graphs showing distances and output vectors for each unseen instance
displayGraphs = 0

# set whether to do performance testing over a range of k instead of using 1 k 
testPerformanceOverKRange = 0

if ( testPerformanceOverKRange == 0 ):
    # Set the number of nearest neighbors to use
    k = 8

    # Get the file names from command-line arguments passed to the script
    inputsFileName,outputsFileName, unseenInstancesFileName = getFileNamesFromArguments()

    # Perform KNN algorithm
    performKNNUsingFiles(k,inputsFileName,outputsFileName, unseenInstancesFileName)

else:
    coef = []
    p = []

    for i in range(3,25):
        # Set the number of nearest neighbors to use
        k = i

        # get the file names from command-line arguments passed to the script
        inputsFileName,outputsFileName, unseenInstancesFileName = getFileNamesFromArguments()

        coefMean,pMean = performKNNUsingFiles(k,inputsFileName,outputsFileName, unseenInstancesFileName)

        coef.append(coefMean)
        p.append(pMean)

    # Get the current Axes instance on the current figure
    ax = plt.gca()
    print(coef,p)
    plt.plot(coef,'lightgrey')
    plt.plot(p,'lightgreen')
    plt.show()