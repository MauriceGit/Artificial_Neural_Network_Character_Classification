import numpy as np

# Sigmoid Funktion mit Ableitung
def sigmoid(x, deriv = False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# Aus- und Eingabearrays
layer1_output = np.array([[0,0,1,1]]).T
layer0_input = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

# Gewichte initialisieren
np.random.seed(1)
weights = 2*np.random.random((3,1)) - 1

for iter in xrange(1000):
    layer0 = layer0_input
    layer1 = sigmoid(np.dot(layer0, weights))
    layer1_error = layer1_output - layer1
    layer1_delta = layer1_error * sigmoid(layer1, True)
    weights += np.dot(layer0.T, layer1_delta)

    if iter % 20 == 0:
        print "Ergebnis nach ", iter, "Iterationen:"
        print "Fehler"
        print layer1_error
        print "Ausgabe:"
        print layer1
        print "------------"

print ""
print "Endergebnis"
print layer1
