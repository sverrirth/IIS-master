#!/usr/bin/python
from parser import DatasetParser
from os import listdir
from os.path import isfile, join, walk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import manifold,svm
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict

##### MAIN VARIABLES : START #####
path = './train'
BACKGROUND = False
plot_colors = "ymcrgbwk" # The library only allows to use 8 different colors
###### MAIN VARIABLES : END ###### 



######################## READ FILES : START ############################

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

######################### PLOTTING PART: END ###########################
