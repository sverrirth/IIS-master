from parser import DatasetParser
from nnetwork import NNetwork
from os import listdir
from os.path import isfile, join

class App:

	def __init__(self):
		self.nn = NNetwork()
		self.dp = DatasetParser()
		self.tags = []
	def menu(self):
		while True:
			print '\n|||||||||||||||||'
			print 'Choose an option:'
			print '|||||||||||||||||\n'
			print '1. Feed from folder'
			print '2. Train'
			print '3. Test from file'
			print '\n'
			
			option = input('> ')
			if option == 1:
				path = raw_input('Folder: ')
				self.feedFolder(path)
			if option == 2:
				self.train()
			if option == 3:
				file = raw_input('File path: ')
				self.testFile(file)

	def testFile(self, file):
		data = self.dp.parse(file)	
		a = self.nn.activate(data)
		print "\n -> Results:"
		for id,tag in enumerate(self.tags):
			print "     "+tag+": "+str(int(a[id]*100))+"%"
			
		n =  a.tolist().index(max(a))
		print "\n -> Match: "+self.tags[n]
	
	def train(self):
		self.nn.train()
		print "\n -> Neuronal network trained!"
	
	def feedFolder(self, path):
		data = []
		path = './'+path
		folders = [f for f in listdir(path) if not isfile(join(path, f))]
		self.tags = folders
		for id,folder in enumerate(folders):
			files = [f for f in listdir(path+'/'+folder) if isfile(join(path+'/'+folder, f))]
			for file in files:
				data.append([id, self.dp.parse(path+'/'+folder+'/'+file)])
		self.nn.add_data(data)
		print "\n -> Folder added! Total sets: "+str(len(data))
		
			
	
	def main(self):
		
		print '|| IIS Dataset Trainer ||'
	
		self.menu()



if __name__ == "__main__":
	app = App()
	app.main()
