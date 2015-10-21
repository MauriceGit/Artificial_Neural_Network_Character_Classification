
import re
import numpy

def arrayToNum(a):
    index = 0
    while index < len(a):
        if float(a[index]) > 0.5:
            return index
        index += 1
    return -1
    
def string2floatArray(a):
    b = []
    for i in a:
        b.append(float(i))
    return b

def parseTrainingData(filename):
    
    fileArray = open(filename).read().split('\n')
    inputLines   = int(fileArray[0])
    inputColumns = int(fileArray[1])
    outputLines  = int(fileArray[2])
    outputColumns= int(fileArray[3])

    sampleCount = 0
    samples = []

    while sampleCount*(inputLines+1)+(inputLines+1) < len(fileArray):
        singleSample = []
        index = 0
        for i in range(4+sampleCount*(inputLines+1), 3+(sampleCount+1)*(inputLines+1)):
            tmp = filter(bool, re.split(" +", fileArray[i].rstrip()))
            singleSample.append(string2floatArray(tmp))
            index = i
        singleOutput = filter(bool, re.split(" +", fileArray[index+1].rstrip()))
        samples.append(tuple([numpy.array(singleSample), singleOutput]))
        sampleCount += 1
        
    return samples
    
def loadAllTrainingData(filenames, prefix):
	samples = []
	for f in filenames:
		samples += parseTrainingData(prefix + f)
	return samples

def parseImagesFromData(imAndLab):
	res = []
	for t in imAndLab:
		res.append(t[0])
	return res

def parseTargetsFromData(imAndLab):
	res = []
	for t in imAndLab:
		asFloat = []
		for i in t[1]:
			asFloat += [float(i)]
		res.append(asFloat)
	return res
