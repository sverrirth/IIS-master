from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

class NNetwork:
	def __init__(self):
		self.ds = ClassificationDataSet(7, 1, nb_classes=8)  #8 since we have 8 gestures, 7 since we have 7 features
		
	def add_data(self, training_data):
		for gesture in training_data:
			self.ds.addSample(gesture[1], gesture[0])  #a method to add all the training data we have
			
	def newData(self, training_data):   #a method for replacing the data already existing and adding data from scratch
		self.ds = ClassificationDataSet(7, 1, nb_classes=8)
		for gesture in training_data:
			self.ds.addSample(gesture[1], gesture[0])
	
	def train(self, shouldPrint):
		tstdata, trndata = self.ds.splitWithProportion(0.2)  #splits the data into training and verification data
		trndata._convertToOneOfMany()
		tstdata._convertToOneOfMany()
		self.fnn = buildNetwork(trndata.indim, 64, trndata.outdim, outclass=SoftmaxLayer) #builds a network with 64 hidden neurons
		self.trainer = BackpropTrainer(self.fnn, dataset=trndata, momentum=0.1, learningrate=0.01, verbose=True, weightdecay=0.1)
		#uses the backpropagation algorithm
		self.trainer.trainUntilConvergence(dataset=trndata, maxEpochs=100, verbose=True, continueEpochs=10, validationProportion=0.20) #early stopping with 20% as testing data
		trnresult = percentError( self.trainer.testOnClassData(), trndata['class'] )
		tstresult = percentError( self.trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )
		
		if shouldPrint:
			print "epoch: %4d" % self.trainer.totalepochs, "  train error: %5.2f%%" % trnresult, "  test error: %5.2f%%" % tstresult
	def activate(self, data): #tests a particular data point (feature vector)
	    return self.fnn.activate(data)
