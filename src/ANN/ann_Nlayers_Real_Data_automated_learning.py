import numpy as np
import time
from parseFiles import loadAllTrainingData
from parseFiles import parseImagesFromData
from parseFiles import parseTargetsFromData


def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
   
# Initialisiert alle Schichten des Networks mit zufaelligen Matrizen der
# entsprechenden Groesse.
def initNetwork(inputArraySize, outputArraySize, hiddenLayerCount, hiddenLayerSize):
    np.random.seed(1)
    syns = []
    syns.append(2*np.random.random((inputArraySize, hiddenLayerSize)) - 1)
    for i in range(hiddenLayerCount):
        syns.append(2*np.random.random((hiddenLayerSize,hiddenLayerSize)) - 1)
    syns.append(2*np.random.random((hiddenLayerSize,outputArraySize)) - 1)
    return syns

# Trainiert das generierte zufaellige Netz - bis ein bestimmter Error-Wert unterschritten ist 
# und gibt das trainierte Netz zurueck.
def train(inputData, expectedData, syns, maxError, maxRounds, alpha, printError):
    
    dataSize = len(syns)
    
    ls = []
    l_errors = []
    l_deltas = []
    roundCount = 0
	
    # init:
    for i in xrange(dataSize):
        l_errors.append([])
        l_deltas.append([])
    
    while (roundCount < 1 or np.mean(np.abs(l_errors[dataSize-1])) > maxError) and roundCount < maxRounds:
        ls = []
        ls.append(inputData)
        for atLayer in xrange(dataSize):
            ls.append(sigmoid(np.dot(ls[len(ls)-1], syns[atLayer])))
        
        l_errors[dataSize-1] = ls[dataSize] - expectedData
        l_deltas[dataSize-1] = l_errors[dataSize-1] * sigmoid(ls[dataSize], deriv=True)
        
        if printError and roundCount % 3000 == 0:
            print "Error:", np.mean(np.abs(l_errors[dataSize-1]))
        
        for i in xrange(dataSize-2, -1, -1):
            l_errors[i] = l_deltas[i+1].dot(syns[i+1].T)
            l_deltas[i] = l_errors[i] * sigmoid(ls[i+1], deriv=True)
            
        for i in xrange(dataSize-1, -1, -1):   
            syns[i] -= alpha * np.dot(ls[i].T, l_deltas[i])
            
        roundCount += 1
    
    if printError:
        if roundCount < maxRounds:
            print "-----> finished because of very small error values."
        else:
            print "-----> finished because round count was exceeded."
    
    return syns

# basicTrainAlgorithm ist der Lernalgorithmus mit Parametern. 
# Zusaetzlich werden Lern-Epochen und Batch-Size spezifiziert.
def trainAndPredictParametrized(epochCount, batchSize, inputCount, X, y, testData, testTargets, syns, maxErrorRate, maxRounds, alpha, debugPrints):
    
    # Das hier muesste die Anzahl der Epochen zum Lernen sein?!
    for blubb in range(epochCount):
        # Das hier dann die Batch-Groesse!
        for i in range(inputCount/batchSize):
            # Neuronales Netz mit Backpropagation trainieren mit X --> y.
            syns = train(np.array(X[i:i+batchSize]),  np.array(y[i:i+batchSize]), syns, maxErrorRate, maxRounds, alpha, debugPrints)
            
        correctPredicted = testAllData(testData, testTargets, syns, False)
        if debugPrints:
            print "correct predicted:", correctPredicted, " --> ", str(100*correctPredicted/len(testData)) + "%"
    return correctPredicted
    

# Wendet das trainierte neuronale Netz auf uebergebene Testdaten an.     
def predict(data, syns):
    ls = sigmoid(np.dot(data, syns[0]))
    for syn in syns[1:]:
        ls = sigmoid(np.dot(ls, syn))
    return ls
    
# Gibt den Index (also Buchstaben) des groessten Wertes zurueck.
def getBiggestIndex(field):
    maxI = -1
    index = -1
    for i in range(len(field)):
        if field[i] >= maxI:
            maxI = field[i]
            index = i
    return index

# Testet ein Array an Datensaetzen.
def testAllData(testData, testTargets, syns, debug):
    correctPredicted = 0
    for i in range(len(testData)):
        # Trainiertes Netz anwenden mit Testdaten.
        result = predict(testData[i], syns)
        #print result, "<-- should be like -->", testTargets[i]
        if debug:
            print "should be:", chr(getBiggestIndex(result)+65), "but is: ", chr(getBiggestIndex(testTargets[i])+65)
        if getBiggestIndex(result) == getBiggestIndex(testTargets[i]):
            correctPredicted += 1
    return correctPredicted

# Macht aus einem 2D-Feld ein Array der gleichen Reihenfolge.
def flattenFieldToArray(field):
    res = []
    for f in field:
        for a in f:
            res += [a]
    return res

# Nimmt im speziellen Fall ein 3D-Feld und macht ein 2D-Feld draus, in dem
# jede Dimension reduziert wird. Das innerste Element ist dann ein Array und
# nicht jedes einzelne Element!
def reduceDimensionByOne(field):
    res = []
    for f in field:
        res += [flattenFieldToArray(f)]
    return res
    


    
                                        #
                                    #########
                                #################
                            #########################
                        #################################
                    #########################################
                #################################################
            #########################################################
        #################################################################
    #########################################################################
################################__AWESOME_SHIT__#################################
    #########################################################################
        #################################################################
            #########################################################
                #################################################
                    #########################################
                        #################################
                            #########################
                                #################
                                    #########
                                        #
                     
                     
########################################################################
# Trainings- und Testdaten einlesen!
########################################################################

#trainingFiles = ["Trainingsdaten.pat"]
trainingFiles = ["Trainingsdaten.pat", "Handschriften-Sommersemester-2015_01.pat", "Handschriften-Sommersemester-2015_02.pat", "TestHandschriften.pat"]
#testFiles     = ["Handschriften-Sommersemester-2015_02.pat"] 
testFiles     = ["TestWindowsSchrift.pat"] 

images_and_labels = loadAllTrainingData(trainingFiles, "/home/maurice/Uni/Master/Learning_Softcomputing/src/")
images  = parseImagesFromData(images_and_labels)
targets = parseTargetsFromData(images_and_labels)

test_images_and_labels = loadAllTrainingData(testFiles, "/home/maurice/Uni/Master/Learning_Softcomputing/src/")
test_images  = parseImagesFromData(test_images_and_labels)
test_targets = parseTargetsFromData(test_images_and_labels)

X = np.array(reduceDimensionByOne(images[:260]))
y = np.array(targets[:260])

testData = []
for i in range(len(test_images)):
    testData.append(flattenFieldToArray(test_images[i]))
    
########################################################################
# Trainings-Parameter einstellen
########################################################################

inputArraySize   = len(X[0]) 
inputCount       = len(X)    
outputArraySize  = len(y[0]) 
hiddenLayerCount = 2    
hiddenLayerSize  = 25   
maxErrorRate     = 0.0005 
maxRounds        = 25000 
alpha            = 0.02 

epochCount       = 1
batchSize        = 260

bestResult = 0

# Tests, dass zumindest das Datenformat und die jew. Groessen korrekt sind:
if  inputArraySize != len(testData[0]) or inputCount != len(y):
    print "Daten nicht korrekt!."
    exit(0)

for test_epochCount in range(1, 4):                                     # 3
    for test_hiddenLayerCount in range(1, 5):                           # 4
        for test_hiddenLayerSize in range(20, 200, 20):                 # 10
            for test_maxRounds in range(30000, 100000, 10000):          # 10
                for test_alpha in range(1, 120, 20):                    # 10
                    test_test_alpha = test_alpha / 1000.0               
                    for test_batchSize in [65, 130, 195, 260]:          # 4        === 48.000 Durchlaeufe... shit.  
                        t0 = time.clock()
                        ########################################################################
                        # Lernen und testen der Eingabe/Ausgabe-Daten
                        ########################################################################
                        # Initialisieren der Input-Hidden- und Output-Neuronen zu einem Netz.
                        syns = initNetwork(inputArraySize, outputArraySize, test_hiddenLayerCount, test_hiddenLayerSize)
                        # Trainiert das Netz epochenweise mit spezifizierter Batch-Size.
                        # Nach jeder Epoche wird gegen die Testdaten getestet und ein Zwischenwert ausgegeben.
                        tmpResult = trainAndPredictParametrized(test_epochCount, test_batchSize, inputCount, X, y, testData, test_targets, syns, maxErrorRate, test_maxRounds, test_test_alpha, False)
                        if tmpResult >= bestResult:
                            bestResult = tmpResult
                            ########################################################################
                            # Info-Ausgaben zum Nachvollziehen der Parameter
                            ########################################################################
                            print "Input        -->", "Size =", inputArraySize, "- Count =", inputCount
                            print "Output       -->", "Size =", outputArraySize
                            print "Testdata     -->", "Count =", len(testData)
                            print "Hidden Layer -->", "Count =", test_hiddenLayerCount, "- Size =", test_hiddenLayerSize
                            print "Parameters   -->", "Min Error =", maxErrorRate, "- Max Rounds =", test_maxRounds, "- Alpha =", test_test_alpha
                            print "Methods      -->", "Epoch Count =", test_epochCount, "- Batch Size =", test_batchSize
                            print "Time:", (time.clock() - t0) / 60, "Minuten" 
                            print "correct predicted:", bestResult, " --> ", str(100*bestResult/len(testData)) + "%"
                            print "\n===========================================================\n"




















