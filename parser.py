import json
import math
class DatasetParser:
	def parseFile(self, filePath):
		with open(filePath) as data_file:
			return self.parse(json.loads(data_file.read()))
			
	def parse(self, dataR):
			json_all = []
			hands_names = ["Right","Left"]
			fingers_names = ["Thumb","Index","Middle","Ring","Pinky"]
			first = None
			for data in dataR:
				if len(data['hands'])>0:
					if first == None:
						first = 0
					#Filter to create the new JSON with the data to be used in the train process
					new_json = {}
					new_json['frame']=data['frame']
					new_json['palm_position']=data['hands'][0]['palm_position']
					new_json['type'] = 1 if data['hands'][0]['type']=='Left hand' else 0
					new_json['normal_roll']=data['hands'][0]['normal_roll']
					new_json['pitch']=data['hands'][0]['pitch']
					new_json['fingers'] = {}
					ed_start = new_json['palm_position']
					
					for idx,fin in enumerate(fingers_names):
						new_json['fingers'][fin] = {}
						ed_end	 = data['hands'][0]['fingers'][idx]['bones'][3]['end']
						new_json['fingers'][fin]['length']	= self.euclidean_distance(ed_start,ed_end)
						new_json['fingers'][fin]['distal_dir'] = data['hands'][0]['fingers'][idx]['bones'][3]['direction']
					json_all.append(new_json)

			results =  [abs(json_all[-1]['normal_roll'] - json_all[first]['normal_roll']), json_all[-1]['pitch'] - json_all[first]['pitch']]
			for x in range(0, 5):
				results.append(json_all[-1]['fingers'][fingers_names[x]]['length'] - json_all[first]['fingers'][fingers_names[x]]['length'])
			return results

	def str_coor_to_list(self, data):
		return data.replace(" ","").replace('(',"").replace(')',"").split(',')
		
	def euclidean_distance(self, start, end):
		start = self.str_coor_to_list(start)
		end   = self.str_coor_to_list(end)
		return math.sqrt(sum(math.pow(float(start[i])-float(end[i]),2) for i in range(0,3)))
    