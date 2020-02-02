import sys
print ("This is the name of the script: ", sys.argv[0])
print ("Number of arguments: ", len(sys.argv))
print ("The arguments are: " , str(sys.argv))

def setInputVector(fileName):
    inputVector = {}
    with open(inputsFileName) as f:
        for line in f:
           (key, val) = line.split()
           inputVector[key] = val
    return inputVector

inputsFileName = str(sys.argv[1])
outputsFileName = str(sys.argv[2])

inputVector = setInputVector(inputsFileName)
for input in inputVector:
    print(input,inputVector[input])    