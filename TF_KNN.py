import sys
print ("This is the name of the script: ", sys.argv[0])
print ("Number of arguments: ", len(sys.argv))
print ("The arguments are: " , str(sys.argv))

def setInputVector(fileName):
    inputVector = {}
    with open(fileName) as f:
        for line in f:
           (key, val) = line.split()
           inputVector[key] = val
    return inputVector

def setOutputVector(fileName):
    outputVector = {}
    headerArray = []
    with open(fileName) as f:
        for lineIndx,line in enumerate(f):
            if(lineIndx == 0):
                headerArray = line.split()
                headerArray =  cleanSeqNames(headerArray)
            else:
                lineArray = line.split()
                for colIndx,value in enumerate(lineArray):
                    if(not(headerArray[colIndx] in outputVector.keys())):
                        outputVector[headerArray[colIndx]] = []
                    outputVector[headerArray[colIndx]].append(value)
    return outputVector

def cleanSeqNames(seqNamesArray):
    cleanedSeqNames = []
    for seqName in seqNamesArray:
        cleanedSeqName = seqName[1:] if seqName[0]=="'" or seqName[0]=="\"" else seqName
        cleanedSeqName = cleanedSeqName[:-1] if cleanedSeqName[-1]=="'" or cleanedSeqName[-1]=="\"" else cleanedSeqName
        cleanedSeqNames.append(cleanedSeqName)
    return cleanedSeqNames

def initializeOutputVector(outputVector,headerArray):
    for key in headerArray:
        outputVector[key] = []


# Script:
inputsFileName = str(sys.argv[1])
outputsFileName = str(sys.argv[2])

inputVector = setInputVector(inputsFileName)
outputVector = setOutputVector(outputsFileName)

for input in inputVector:
    print(input,inputVector[input],len(outputVector[input]),outputVector[input][0],outputVector[input][-1])