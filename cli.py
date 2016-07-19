from __future__ import division
from parser import DatasetParser
from nnetwork import NNetwork
from os import listdir
from os.path import isfile, join, walk
from string import digits
from pybrain.utilities import percentError
from pybrain.datasets            import ClassificationDataSet
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

import sys
sys.path.insert(0, "/Users/pablo/Downloads/LeapDeveloperKit_2.3.1+31549_mac/LeapSDK/lib")
import Leap, sys, thread, time
from Leap import CircleGesture, KeyTapGesture, ScreenTapGesture, SwipeGesture
import json
import pprint
import math
import copy
import cPickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import manifold,svm
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict

class App:
	
    nGestures = 8
    nSamples = 10
    nUsers = 8
    gestureNames = ["circle.txt", "come.txt", "pinky.txt", "pistol.txt", "rock.txt", "roll.txt", "scissors.txt",
                     "stop.txt"]

    def __init__(self):
        self.nn = NNetwork()
        self.dp = DatasetParser()
        self.listener = SampleListener()
        self.controller = Leap.Controller()
        self.tags = []
    def menu(self):

        allresults = [] #for storing the feature vectors from the training data
        features_gathered = False #a flag to see if the feature vectors have been stored so far

        while True:
            print '\n|||||||||||||||||'
            print 'Choose an option:'
            print '|||||||||||||||||\n'
            print '1. Feed from folder'
            print '2. Train NN'
            print '3. Test from file'
            print '4. Record'
            print '5. Test from input'
            print '6. Show and save all features vectors'
            print '7. TensorFlow'
            print '8. Cross-verification with (W)(K)NN'
            print '9. Test input AUTO'
            print '10. Train with NN & save'
            print '11. Plot'
            print '\n'

            option = int(raw_input('> '))
            if option == 1:
                path = raw_input('Folder: ')
                self.feedFolder(path)
            if option == 2:
                self.train()
            if option == 3:
                file = raw_input('File path: ')
                self.testFile(file)
            if option == 4:
                gesture = raw_input('Gesture key: ')
                self.recordGesture(gesture)
            if option == 5:
                self.detectGesture()
            if option == 6:
                self.showAllFeatures()
            if option == 7:
                self.tensorFlow()
            if option == 8:
                print 'Select KNN (1) or WKNN (2) or NN (3)'
                option2 = int(raw_input('> '))
                if option2 == 1:
                    self.cross_verify(1)
                if option2 == 2:
                    self.cross_verify(2)
                if option2 == 3:
                    self.cross_verify(3)
            if option == 9:
                print 'Select KNN (1) or WKNN (2) or NN (3)'
                option2 = int(raw_input('> '))
                if option2 == 1:
                    self.detectGestureAuto(1)
                if option2 == 2:
                    self.detectGestureAuto(2)
                if option2 == 3:
                    self.detectGestureAuto(3)
            if option == 10:
                self.neuronalNetwork()
            if option == 11:
            	self.doPlot()
    
	    
    def doPlot(self): #plot the training data using t-MSE
		BACKGROUND = False
		path = './train'
		plot_colors = "ymcrgbwk"  #the letters correspond to clors
		input_data=[]
		folders = [f for f in listdir(path) if not isfile(join(path, f))]
		for index,folder in enumerate(folders):
			subFolders_users = [f for f in listdir(path+'/'+folder) if not isfile(join(path+'/'+folder, f))]
			for index2,folder2 in enumerate(subFolders_users):
				files = [f for f in listdir(path+'/'+folder+'/'+folder2) if isfile(join(path+'/'+folder+'/'+folder2, f)) and '.txt' in f]
				for file in files:
					input_data.append([index] + DatasetParser().parseFile(path+'/'+folder+'/'+folder2+'/'+file))
		
		n_classes = len(folders)
		######################### READ FILES : END #############################
		
		
		
		
		##################### DATA MODIFICATION : START ########################
		
		# Transform the list into a numpy array
		numpy_array = np.array(input_data)
		
		# Choose which column is the label and which are the data
		data = numpy_array[:,1:8]
		y = numpy_array[:,0]
		
		# To apply a classifier on this data, we need to flatten the image, to
		# turn the data in a (samples, feature) matrix:
		n_samples = len(data)
		
		#Manifold Embedding
		tsne = manifold.TSNE(n_components=2, random_state=1)
		X_trans = tsne.fit_transform(data)
		
		# We only take the two corresponding features
		X = X_trans
		
		###################### DATA MODIFICATION : END #########################
		
		
		
		######################## PLOTTING PART: START ##########################
		
		if BACKGROUND == True:
			# Parameters
			plot_step = 0.01
			
			# Shuffle
			idx = np.arange(X.shape[0])
			np.random.seed(13)
			np.random.shuffle(idx)
		
			X = X[idx]
			y = y[idx]
		
			# Standardize
			mean = X.mean(axis=0)
			std = X.std(axis=0)
			X = (X - mean) / std
		
			# Train
			clf = DecisionTreeClassifier().fit(X, y)
		
			x_min, x_max = round(X[:, 0].min()-0.1,1), round(X[:, 0].max()+0.1,1)
			y_min, y_max = round(X[:, 1].min()-0.1,1), round(X[:, 1].max()+0.1,1)
			xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
								 np.arange(y_min, y_max, plot_step))
		
			Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
			Z = Z.reshape(xx.shape)
			cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
		
			for i in idx:
				plt.scatter(X[i, 0], X[i, 1], label=y[i], c=plot_colors[int(y[i])], cmap=plt.cm.Paired)
		else:	
			
			x_min = int(round(X[:, 0].min()-10,-1))
			x_max = int(round(X[:, 0].max()+10,-1))
			y_min = int(round(X[:, 1].min()-10,-1))
			y_max = int(round(X[:, 1].max()+10,-1))	
			
			plt.xlim(x_min,x_max)
			plt.ylim(y_min,y_max)
			
			plt.xticks(range(x_min,x_max,10))
			plt.yticks(range(y_min,y_max,10))
			
			
			for i in range(0,len(X)):
				plt.scatter(X[i, 0], X[i, 1], label=y[i], c=plot_colors[int(y[i])], cmap=plt.cm.Paired)
		
		handles, labels = plt.gca().get_legend_handles_labels()
		by_label = OrderedDict(zip(labels, handles))
		
		plt.legend(by_label.values(), folders, scatterpoints=1,borderpad=1,bbox_to_anchor=[1.15,1.11])
		
		plt.savefig('points.png')
		
		
		plt.show()
		
		plt.close()	    
    def neuronalNetwork(self):
        allresults = cPickle.load(open('features.p')) 
        self.nn.add_data(allresults)
        self.nn.train(True)
        fileObject = open('nn', 'w')
        cPickle.dump(self.nn, fileObject)
        fileObject.close()
    
    def multilayer_perceptron(self, _X, _weights, _biases):
        #Hidden layer with RELU activation
        layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
        #Hidden layer with RELU activation
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
        return tf.matmul(layer_2, _weights['out']) + _biases['out']

    def showAllFeatures(self):
        allresults = []  #for storing the feature vectors
        path = './train'
        outer_folders = [f for f in listdir(path) if not isfile(join(path, f))]
        folders = []  #the folders in which each gesture is stored
        for i in range(1, self.nUsers+1):
            for j in outer_folders:
                folders.append(j + "/user" + str(i))  #opening each user-file in each gesture folder
        print folders
        self.tags = folders
        for id, folder in enumerate(folders):
            files = [f for f in listdir(path + '/' + folder) if isfile(join(path + '/' + folder, f)) and '.txt' in f]
            for file in files:
                filename = self.dp.parseFile(path + '/' + folder + '/' + file)
                allresults.append([self.gestureNames.index(file.translate(None, digits)), filename])
                print file
                print self.dp.parseFile(path + '/' + folder + '/' + file)

        print len(allresults)

        cPickle.dump(allresults, open('features.p', 'wb'))

    def detectGestureAuto(self, selection):
        print "PLACE HAND"
        sys.stdout.flush()
        self.controller.add_listener(self.listener)

        last = self.listener.last
        time.sleep(0.1)
        now = self.listener.last

        while self.checkPause(False, last, now, 0.5) == False:
            last = now
            time.sleep(0.1)
            now = self.listener.last

        print "START GESTURE"
        sys.stdout.flush()

        first = last

        last = self.listener.last
        time.sleep(0.05)
        now = self.listener.last
        stop = False
        while self.checkPause(True, last, now, 3.0) == False and not stop:
            last = now
            time.sleep(0.05)
            now = self.listener.last
            if now['hands']==[]: #if the hand is no longer being detected by the leap motion, the method should not return a gesture
               print "NOTHING DETECTED"
               sys.stdout.flush()
               stop = True
               self.detectGestureAuto(selection)
               return 

        print "MOVEMENT DETECTED"
        sys.stdout.flush()
        stop = False
        last = self.listener.last
        time.sleep(0.1)
        now = self.listener.last
        while self.checkPause(False, last, now, 0.5) == False and not stop:
            if now['hands']==[]:
               print "NOTHING DETECTED"
               sys.stdout.flush()
               stop = True
               self.detectGestureAuto(selection)
               return 
            sys.stdout.flush()
            last = now
            time.sleep(0.1)
            now = self.listener.last
     
            
        data = self.dp.parse([first, last])   #the last and the first frame are used to create a feature vector
  
        allresults = cPickle.load(open('features.p'))
        print self.gestureNames[self.returnBest(selection, allresults, data, 16)].replace(".txt", "") #16 is a default value for k that seems to work reasonably
        sys.stdout.flush()
        time.sleep(2)
        self.detectGestureAuto(selection) #this method is always on loop
    def checkPause(self, movement, first, second, amount):

        if first==None or second==None:
            return False

        if first['hands'] == [] or second['hands'] == []:
            return False

        data = self.dp.parse([first, second])

        if(movement):

            return any(i>amount for i in data)

        else:

            return all(i<amount for i in data)

    def cv_nn(self): #a help method for doing cross-validation

        all_results = cPickle.load(open('features.p'))

        split_no = self.nGestures * self.nSamples

        split = [all_results[i:i + split_no] for i in range(0, len(all_results), split_no)]

        split_results = []

        for i in range(0, self.nUsers):

            this_results = []

            for j in range(0, self.nUsers):

                if j != i:

                    this_results += (split[j])

            split_results.append(this_results)

        return (split, split_results) #the first list is the data belong to a specific user by index, the second is all data except by user specified by index

    def cross_verify(self, selection): #selection is an integer which specifies which of the three algorithms is being tested

        (split, split_results) = self.cv_nn()

        if selection == 3: #neural network
        
            percErrors = [0] * self.nUsers

            for i in range(0, self.nUsers):
            
                verificationSet = ClassificationDataSet(7, 1, nb_classes=8)
                for gesture in split[i]:
                
                   
                
                    verificationSet.addSample(gesture[1], gesture[0]) 
            	
            	trainingSet = ClassificationDataSet(7, 1, nb_classes=8)
            	            	
            	for gesture in split_results[i]:
            	                
                    trainingSet.addSample(gesture[1], gesture[0]) #the first is the class, the second is the feature vector
                    
                verificationSet._convertToOneOfMany()
                trainingSet._convertToOneOfMany()
                    
                neural = buildNetwork(trainingSet.indim, 10, trainingSet.outdim, outclass=SoftmaxLayer)
                
                trainer = BackpropTrainer(neural, dataset=trainingSet, momentum=0.1, learningrate=0.01, verbose=True, weightdecay=0.1)
                trainer.trainUntilConvergence(dataset=trainingSet, maxEpochs=100, verbose=True, continueEpochs=10, validationProportion=0.20)
            	#splits the dataset 80-20 and uses the 20 percent for early stopping
            	vrfresult = percentError( trainer.testOnClassData(dataset=verificationSet ), verificationSet['class'])
            	
            	percErrors[i] = vrfresult
            	
            	print "The error for ", str(i), " was %5.2f%%" % vrfresult
            
            #print "The average error for the nn is", sum([x for x in percErrors])/self.nUsers
            
        if selection == 1 or selection == 2: #knn and wknn here

            for k in range(1, self.nUsers * self.nGestures * self.nSamples + 1):

                all_users = [0] * self.nUsers

                for i in range(0, self.nUsers):

                    successes = 0
                    failures = 0

                    for element in split[i]:

                        correct_class = element[0]
                        features = element[1]

                        if element in split_results[i]:
                            print "error"

                        result = self.returnBest(selection, split_results[i], features, k)

                        if result == correct_class:
                            successes += 1
                        else:
                            failures += 1
                    all_users[i] = (successes/(successes+failures))

                print "k", k, "all_users", all_users

                average = sum([x for x in all_users])/self.nUsers

                if selection==1:
                    print "k:", k, "average_knn:", str(average)
                if selection==2:
                    print "k:", k, "average_wknn:", str(average)

    def returnBest(self, selection, all_results, element, k): #return the class expected according to the selected algorithm (selection)

        distances = [[x[0], self.eucl(element, x[1])] for x in all_results]
        distances.sort(key=lambda x: x[1])

        if selection == 1: #knn algorithm

            return self.best(k, [x[0] for x in distances])

        if selection == 2: #wknn algorithm

            distances = [[x[0], self.eucl(element, x[1])] for x in all_results]
            distances.sort(key=lambda x: x[1])

            weight_sums = [0] * self.nGestures
            shortest = distances[0][1]
            furthest = distances[k-1][1]

            diff = furthest - shortest

            weight_sums[distances[0][0]] = 1 #closest element has the weight 1

            for i in range(1,k-1):
                element = distances[i]
                weight = (furthest - element[1]) / diff
                weight_sums[element[0]] += weight

            return weight_sums.index(max(weight_sums))  #returns the normal knn and weighted knn result

        if selection == 3:
            fileObject = open('nn','r')
            self.nn = cPickle.load(fileObject)
            a = self.nn.activate(element)
            n = a.tolist().index(max(a))
            return n

    def most_common(self, listy):
        return max(set(listy), key=listy.count)

    def best(self, k, distances):
        return self.most_common([distances[i] for i in range(0, k)])

    def eucl(self, f, s):
        dist = 0
        for i in range(0, len(f)):
            dist += math.pow(f[i] - s[i], 2)
        return math.sqrt(dist)

    def detectGesture(self): #detcts the gesture using the neural network
        self.controller.add_listener(self.listener)
        raw_input("Position 1...")
        first = self.listener.last
        raw_input("Position 2...")
        last = self.listener.last
        data = self.dp.parse([first, last])
        a = self.nn.activate(data)
        print "\n -> Results:"
        for id, tag in enumerate(self.tags):
            print "     " + tag + ": " + str(int(a[id] * 100)) + "%"

        n = a.tolist().index(max(a))
        print "\n -> Match: " + self.tags[n]


    def recordGesture(self, gesture): #records the gesture and stores it in the data directory
        raw_input("Start...")
        self.controller.add_listener(self.listener)
        #first = self.listener.last
        raw_input("End...")
        #last = self.listener.last
        path = 'train/' + gesture + '/'
        print self.dp.parse(self.listener.recording)
        files = [f for f in listdir(path) if isfile(join(path, f)) and '.txt' in f]
        file_ = open(path + gesture + str(len(files) + 1) + ".txt", 'w')
        file_.write(json.dumps(self.listener.recording))
        file_.close()
        self.listener.recording = []
        self.controller.remove_listener(self.listener)


    def testFile(self, file): #tests a file using a neural network
        data = self.dp.parseFile(file)
        a = self.nn.activate(data)
        print "\n -> Results:"
        for id, tag in enumerate(self.tags):
            print "     " + tag + ": " + str(int(a[id] * 100)) + "%"

        n = a.tolist().index(max(a))
        print "\n -> Match: " + self.tags[n]


    def train(self):
        self.nn.train()
        print "\n -> Neuronal network trained!"


    def feedFolder(self, path):
        data = []
        path = './' + path
        folders = [f for f in listdir(path) if not isfile(join(path, f))]
        self.tags = folders
        for id, folder in enumerate(folders):
            files = [f for f in listdir(path + '/' + folder) if isfile(join(path + '/' + folder, f)) and '.txt' in f]
            for file in files:
                data.append([id, self.dp.parseFile(path + '/' + folder + '/' + file)])
        self.nn.add_data(data)
        print "\n -> Folder added! Total sets: " + str(len(data))


    def main(self):
        print '|| IIS Dataset Trainer ||'

        self.menu()



class SampleListener(Leap.Listener):
    last = None
    recording = []
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
    state_names = ['STATE_INVALID', 'STATE_START', 'STATE_UPDATE', 'STATE_END']
    def get_last(self):
    	return last
    def on_init(self, controller):
	    pass

    def on_connect(self, controller):
	    controller.set_policy_flags(Leap.Controller.POLICY_BACKGROUND_FRAMES);

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print "Disconnected"

    def on_exit(self, controller):
	    pass

    def on_frame(self, controller):
        frame = controller.frame()
        data = {"frame":frame.id, "hands":[]}
        #print "Frame id: %d, timestamp: %d, hands: %d, fingers: %d, tools: %d, gestures: %d" % (
        #      frame.id, frame.timestamp, len(frame.hands), len(frame.fingers), len(frame.tools), len(frame.gestures()))

        # Get hands
        for hand in frame.hands:
            handType = "Left hand" if hand.is_left else "Right hand"


            # Get the hand's normal vector and direction
            normal = hand.palm_normal
            direction = hand.direction

            # Calculate the hand's pitch, roll, and yaw angles


            # Get arm bone
            arm = hand.arm

            fingers = []
            # Get fingers
            for finger in hand.fingers:

                bones = []
                # Get bones
                for b in range(0, 4):
                    bone = finger.bone(b)

                    bones.append({"name":self.bone_names[bone.type], "start":str(bone.prev_joint), "end":str(bone.next_joint), "direction":str(bone.direction)})
                fingers.append({"name":self.finger_names[finger.type], "id":finger.id, "bones":bones})
                
            data["hands"].append({"type":handType, "id":hand.id, "palm_position":str(hand.palm_position), "normal":str(normal), "pitch":direction.pitch*Leap.RAD_TO_DEG, "normal_roll":normal.roll * Leap.RAD_TO_DEG, "direction_yaw":direction.yaw * Leap.RAD_TO_DEG, "fingers":fingers})
        self.recording.append(data)
        self.last = data
        
if __name__ == "__main__":
	app = App()
	app.main()
