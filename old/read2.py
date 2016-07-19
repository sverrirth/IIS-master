import json
import math
#from pybrain.datasets import ClassificationDataSet
from pprint import pprint

###### Configuration : Start ######

ALL_FRAMES = False #In case of 'False' only the first line and the last one will be treated  <- TO BE INPLEMENTED ******

######  Configuration : End  ######

def main():

    training_data = []

    data_size = 10
    types = 5

    string_names = ["pistol", "yaw", "rnr", "rock", "scissors"]

    for i in range(0,5):

        for j in range(1,11):

            with open('train/' + string_names[i] + str(j) + '.txt') as data_file:
            #~ with open('test.txt') as data_file:

                hands_names = ["Right","Left"]
                fingers_names = ["Thumb","Index","Middle","Ring","Pinky"]

                json_all = []

                for line,text in enumerate(data_file):
                    try:
                        data=json.loads(text)
                        #~ pprint(data) #To print the JSON with a better view for the human eye :)

                        # Filter to create the new JSON with the data to be used in the train process
                        new_json = {}
                        new_json['frame']=data['frame']

                        new_json['palm_position']=data['hands'][0]['palm_position']
                        new_json['type'] = 1 if data['hands'][0]['type'] else 0
                        new_json['normal_roll']=data['hands'][0]['normal_roll']
                        new_json['fingers'] = {}

                        ed_start = new_json['palm_position']

                        for idx,fin in enumerate(fingers_names):
                            new_json['fingers'][fin] = {}

                            ed_end	 = data['hands'][0]['fingers'][idx]['bones'][3]['end']
                            new_json['fingers'][fin]['length']	= euclidean_distance(ed_start,ed_end)
                            new_json['fingers'][fin]['distal_dir'] = data['hands'][0]['fingers'][idx]['bones'][3]['direction']

                        json_all.append(new_json)

                    except:
                        print "ERROR in line: ", line
                        continue
                data_file.close()
                results =  [json_all[0]['type'], json_all[-1]['normal_roll'] - json_all[0]['normal_roll']]
                for x in range(0, 5):
                  results.append(json_all[-1]['fingers'][fingers_names[x]]['length'] - json_all[0]['fingers'][fingers_names[x]]['length'])

                training_data.append([i, results])

    print training_data

def str_coor_to_list(data):
    return data.replace(" ","").replace('(',"").replace(')',"").split(',')

def euclidean_distance(start, end):
    start = str_coor_to_list(start)
    end   = str_coor_to_list(end)
    return math.sqrt(sum(math.pow(float(start[i])-float(end[i]),2) for i in range(0,3)))

if __name__ == "__main__":
    main()
