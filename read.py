import os, json, math
from pprint import pprint

###### Configuration : Start ######
DIR = 'train/'
ONLY_FIRST_LAST = True #In case of 'True' only the first line and the last one will be treated. Else, the whole document is readed
VERBOSE = True
######  Configuration : End  ######

def main():
	for filename in os.listdir(DIR):
		with open(DIR+filename,"r") as data_file:
			
			hands_names = ["Right","Left"]
			fingers_names = ["Thumb","Index","Middle","Ring","Pinky"]
			
			all_file = data_file.readlines()
			total_lines = len(all_file)
			
			for line_num,text in enumerate(all_file):
				
				if ONLY_FIRST_LAST:
					if not (line_num==0 or line_num==total_lines-1):
						continue
				
				try:
					data=json.loads(text)
					#~ pprint(data) #To print the JSON with a better view for the human eye :)
					
					# Filter to create the new JSON with the data to be used in the train process
					new_json = {}
					new_json['frame']=data['frame']

					new_json['palm_position']=data['hands'][0]['palm_position']
					new_json['normal_roll']=data['hands'][0]['normal_roll']
					new_json['fingers'] = {}
					
					ed_start = new_json['palm_position']
					
					for idx,fin in enumerate(fingers_names):
						new_json['fingers'][fin] = {}
						ed_end	 = data['hands'][0]['fingers'][idx]['bones'][3]['end']
						new_json['fingers'][fin]['length']	= euclidean_distance(ed_start,ed_end)
						new_json['fingers'][fin]['distal_dir'] = data['hands'][0]['fingers'][idx]['bones'][3]['direction']
					
					json_data = json.dumps(new_json)
					if VERBOSE:
						pprint(json_data)

				except:
					if VERBOSE:
						print "ERROR in line: ", line_num # Enumeration beguins in 0. It can be used as a index.
					continue
					
				data_file.close()

def str_coor_to_list(data):
	return data.replace(" ","").replace('(',"").replace(')',"").split(',')
	
def euclidean_distance(start, end):
	start = str_coor_to_list(start)
	end   = str_coor_to_list(end)
	return math.sqrt(sum(math.pow(float(start[i])-float(end[i]),2) for i in range(0,3)))

if __name__ == "__main__":
	main()
