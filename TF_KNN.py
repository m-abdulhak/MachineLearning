import sys
import numpy as np
import operator
import matplotlib.pyplot as plt

print ("This is the name of the script: ", sys.argv[0])
print ("Number of arguments: ", len(sys.argv))
print ("The arguments are: " , str(sys.argv))

######################
### Handling files ###
######################

def getDataInputs(fileName):
    dataInputs = {}
    with open(fileName) as f:
        for line in f:
           (key, val) = line.split()
           dataInputs[key] = val
    return dataInputs

def getTrainingDataOutputs(fileName):
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
    outputFileName = "out.txt"

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

def calculateOutputVectorUsingKNN(newInstance,trainingObservations):
    distances = calcInstanceObsDistances(newInstance,trainingObservations)
    kNearestNeighbors = getKMinDistances(k,distances)
    displayDistances(distances,kNearestNeighbors)
    
    instanceOutput = generateInstanceOutput(kNearestNeighbors,trainingObservations['outputs'])
    displayOutputVectors(instanceOutput,trainingObservations['outputs'],kNearestNeighbors)

    return instanceOutput

def calcInstanceObsDistances(xi,obs):
    distances = {}
    for indx, o in enumerate(obs['inputs']):
        distances[o] = calcDistance(xi,obs['inputs'][o])

    return distances

def calcDistance(x1,x2):
    length = np.minimum(len(x1),len(x2))

    distance = 0

    for i in range(0,length):
        distance += 1 if x1[i] != x2[i] else 0
    
    return distance

def getKMinDistances(k,dict):
    # the k minimum distances found in dict
    kMinDistances = {}
    # the maximum distance within the list 
    maxDistInKMin = 0
    # the key of the maximum distance within the list
    keyOfMaxDistInKMin = 0
    
    for indx,x in enumerate(dict):
        if(len(kMinDistances)<k):
            if(len(kMinDistances)==0 or dict[x]>maxDistInKMin):
                maxDistInKMin = dict[x]
                keyOfMaxDistInKMin = x
            kMinDistances[x] = dict[x]
        else:
            if(dict[x]<maxDistInKMin):
                kMinDistances.pop(keyOfMaxDistInKMin)
                kMinDistances[x] = dict[x]
                keyOfMaxDistInKMin = max(kMinDistances.items(), key=operator.itemgetter(1))[0]
                maxDistInKMin = dict[keyOfMaxDistInKMin]
    return kMinDistances

def generateInstanceOutput(kNearestNeighbors,trainingDataOutputs):
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
    if(displayGraphs):
        # Get the current Axes instance on the current figure
        ax = plt.gca()

        for distance in distances:
            plt.plot(0,distances[distance], marker='o', markersize=2, color = 'lightgrey')
        
        for distance in kNearestNeighbors:
            plt.plot(0,kNearestNeighbors[distance], marker='o', markersize=2, color = 'red')

        plt.show()

def displayOutputVectors(outputVector,trainingOutputs,kNearestNeighbors):
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

# Set the number of nearest neighbors to use
k = 100

# get the file names from command-line arguments passed to the script
inputsFileName,outputsFileName, unseenInstancesFileName = getFileNamesFromArguments()

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
    outputDataSet[instanceName] = calculateOutputVectorUsingKNN(unseenInstances[instanceName],trainingObservations)

writeToFile(outputDataSet)