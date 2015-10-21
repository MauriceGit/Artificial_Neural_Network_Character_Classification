import numpy as np
import time
from parseFiles import loadAllTrainingData
from parseFiles import parseImagesFromData
from parseFiles import parseTargetsFromData
import matplotlib.pyplot as plt
from numpy.random import normal
import string

#                    .....  #
#                .          #
#             .             #
#           .               #
#          .                #
#         .                 #
#       .                   #
#.....                      #  
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
        
        # Forward Propagation
        ls.append(inputData)
        for atLayer in xrange(dataSize):
            ls.append(sigmoid(np.dot(ls[len(ls)-1], syns[atLayer])))
        
        # Wie weit sind wir vom Perfekten Wert, den wir haben wollen noch weg?
        l_errors[dataSize-1] = ls[dataSize] - expectedData
        # In welche Richtung haben wir uns im letzten Schritt denn vertan? In die Richtung anpassen!
        # Wenn wir gut waren, nicht viel aendern!
        l_deltas[dataSize-1] = l_errors[dataSize-1] * sigmoid(ls[dataSize], deriv=True)
        
        if printError and roundCount % 3000 == 0:
            print "Error:", np.mean(np.abs(l_errors[dataSize-1]))
        
        for i in xrange(dataSize-2, -1, -1):
			# Wieviel hat denn der letzte Delta-Wert abhaengig der Gewichte zum jetzigen Errorwert beigetragen?
            l_errors[i] = l_deltas[i+1].dot(syns[i+1].T)
            # In welche Richtung liegt denn unser lokales Ziel ls[i+1]?
            # Wenn wir gut waren, nicht viel aendern!
            l_deltas[i] = l_errors[i] * sigmoid(ls[i+1], deriv=True)
            
        for i in xrange(dataSize-1, -1, -1):   
			# Gewichte updaten und aufbessern.
			# Zusaetzlich Gradientenverfahren ueber alpha.
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
def trainAndPredictParametrized(epochCount, batchSize, inputCount, X, y, testData, testTargets, syns, maxErrorRate, maxRounds, alpha, rejectionRate, debugPrints):
    
    # Das hier muesste die Anzahl der Epochen zum Lernen sein?!
    for blubb in range(epochCount):
        # Das hier dann die Batch-Groesse!
        for i in range(inputCount/batchSize):
            # Neuronales Netz mit Backpropagation trainieren mit X --> y.
            syns = train(np.array(X[i:i+batchSize]),  np.array(y[i:i+batchSize]), syns, maxErrorRate, maxRounds, alpha, debugPrints)
            
        correctPredicted = testAllData(testData, testTargets, syns, rejectionRate, False)
        print "correct predicted:", correctPredicted, " --> ", str(100*correctPredicted/len(testData)) + "%"
    
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
    return index, maxI

# Erstellt ein Histogramm der Verteilung der erkannten Buchstaben.
# Also welcher Buchstabe wie oft korrekt erkannt wurde.
def plotData(data):
	plt.hist(data, bins=26)
	plt.title("Histogramm")
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	plt.xticks(range(26), list(string.ascii_uppercase), rotation=45)	
	plt.show()

# Plottet 16 Beispielbuchstaben aus den Handschriften 2015.
# Die ersten 2 Zeilen sind Negativbeispiele, die naechsten 2 Zeilen Positivbeispiele.
def plotAllChars(chars):
	# negativ
	MsN = [38, 90, 220, 350]
	# negativ
	IsN = [34, 346]
	# negativ
	ZsN = [51, 363]
	# positiv
	IsP = [60, 112, 268, 372]
	# positiv
	XsP = [179, 257, 309, 387]
	
	# 1. Zeile mit Ms
	for i in range(4):
		ax1 = plt.subplot2grid((4,4), (0,i))
		ax1.imshow(chars[MsN[i]], cmap=plt.cm.gray_r, interpolation='nearest')
		plt.title('Buchstabe: M')
		ax1.axes.get_xaxis().set_visible(False)
		ax1.axes.get_yaxis().set_visible(False)
		plt.tight_layout()
	
	# 2. Zeile mit Is
	for i in range(2):
		ax1 = plt.subplot2grid((4,4), (1,i))
		ax1.imshow(chars[IsN[i]], cmap=plt.cm.gray_r, interpolation='nearest')
		plt.title('Buchstabe: I')
		ax1.axes.get_xaxis().set_visible(False)
		ax1.axes.get_yaxis().set_visible(False)
		plt.tight_layout()
		
	# 2. Zeile mit Zs
	for i in range(2, 4):
		ax1 = plt.subplot2grid((4,4), (1,i))
		ax1.imshow(chars[ZsN[i-2]], cmap=plt.cm.gray_r, interpolation='nearest')
		plt.title('Buchstabe: Z')
		ax1.axes.get_xaxis().set_visible(False)
		ax1.axes.get_yaxis().set_visible(False)
		plt.tight_layout()
	
	# 1. Zeile mit Is
	for i in range(4):
		ax1 = plt.subplot2grid((4,4), (2,i))
		ax1.imshow(chars[IsP[i]], cmap=plt.cm.gray_r, interpolation='nearest')
		plt.title('Buchstabe: I')
		ax1.axes.get_xaxis().set_visible(False)
		ax1.axes.get_yaxis().set_visible(False)
		plt.tight_layout()
	
	# 1. Zeile mit Xs
	for i in range(4):
		ax1 = plt.subplot2grid((4,4), (3,i))
		ax1.imshow(chars[XsP[i]], cmap=plt.cm.gray_r, interpolation='nearest')
		plt.title('Buchstabe: X')
		ax1.axes.get_xaxis().set_visible(False)
		ax1.axes.get_yaxis().set_visible(False)
	
	plt.show()

# Testet ein Array an Datensaetzen.
def testAllData(testData, testTargets, syns, rejectionRate, debug):
    correctPredicted = 0
    predictedValues = []
    for i in range(len(testData)):
        # Trainiertes Netz anwenden mit Testdaten.
        result = predict(testData[i], syns)
        biggestIndex, rate = getBiggestIndex(result)
        biggestIndexTarget, tmp = getBiggestIndex(testTargets[i])
        
        if debug:
            print "should be:", chr(biggestIndex+65), "but is: ", chr(biggestIndexTarget+65)
            
        if biggestIndex == biggestIndexTarget and rate >= rejectionRate:
            correctPredicted += 1
            predictedValues += [biggestIndex]
    
    #plotData(predictedValues)
    
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

basePath	  = "../"
trainingFiles = ["Trainingsdaten.pat"]
testFiles     = ["Handschriften-Sommersemester-2015.pat"] 
#testFiles     = ["TestWindowsSchrift.pat"] 
testDataCount = 260

images_and_labels = loadAllTrainingData(trainingFiles, basePath)
images  = parseImagesFromData(images_and_labels)
targets = parseTargetsFromData(images_and_labels)

test_images_and_labels = loadAllTrainingData(testFiles, basePath)
test_images  = parseImagesFromData(test_images_and_labels)
test_targets = parseTargetsFromData(test_images_and_labels)

X = np.array(reduceDimensionByOne(images[:testDataCount]))
y = np.array(targets[:testDataCount])

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
maxRounds        = 30000 
alpha            = 0.02 

epochCount       = 1
batchSize        = testDataCount

rejectionRate	 = 0.4

########################################################################
# Info-Ausgaben zum Nachvollziehen der Parameter
########################################################################

print "Input        -->", "Size =", inputArraySize, "- Count =", inputCount
print "Output       -->", "Size =", outputArraySize
print "Testdata     -->", "Count =", len(testData)
print "Hidden Layer -->", "Count =", hiddenLayerCount, "- Size =", hiddenLayerSize
print "Parameters   -->", "Min Error =", maxErrorRate, "- Max Rounds =", maxRounds, "- Alpha =", alpha
print "Methods      -->", "Epoch Count =", epochCount, "- Batch Size =", batchSize

# Tests, dass zumindest das Datenformat und die jew. Groessen korrekt sind:
if  inputArraySize != len(testData[0]) or inputCount != len(y):
    print "Daten nicht korrekt!."
    exit(0)

########################################################################
# Lernen und testen der Eingabe/Ausgabe-Daten
########################################################################

t0 = time.clock()
# Initialisieren der Input-Hidden- und Output-Neuronen zu einem Netz.
syns = initNetwork(inputArraySize, outputArraySize, hiddenLayerCount, hiddenLayerSize)

# Trainiert das Netz epochenweise mit spezifizierter Batch-Size.
# Nach jeder Epoche wird gegen die Testdaten getestet und ein Zwischenwert ausgegeben.
#trainAndPredictParametrized(epochCount, batchSize, inputCount, X, y, testData, test_targets, syns, maxErrorRate, maxRounds, alpha, rejectionRate, True)
print "Time:", (time.clock() - t0) / 60, "Minuten"  

plotData([3, 10, 11, 8, 23, 8, 9, 0, 1, 2, 9, 18, 23, 8, 18, 8, 9, 11, 21, 3, 8, 9, 11, 0, 8, 11, 17, 23, 8, 11, 23, 0, 2, 3, 8, 21, 23, 0, 3, 7, 10, 11, 23, 0, 2, 8, 9, 11, 16, 18, 20, 23, 5, 8, 9, 11, 24]);
plotAllChars(test_images)










