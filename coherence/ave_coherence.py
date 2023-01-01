import os
import json


project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__))) 
source_dir = os.path.join(project_dir,"results")
#record coherence at different time to output list for visualisation
chosen_type = "UMass"
chosen_type_coherence = {}


for foldername in os.listdir(source_dir):
    file_dir = os.path.join(source_dir,foldername)
    print("=======================================")
    print(foldername)
    for filename in os.listdir(file_dir):
        try:
            file_info = filename[:-4]
            current_type = file_info[:-len(foldername)]
            print(current_type)
            
            with open(os.path.join(file_dir, filename), "r") as f:
                data = f.readlines()
                f.close()
            score = 0.
            for item in data[1:]:
                item = item.strip()
                item = item.split("\t")
                #print(item)
                score += float(item[1])
            score = score/len(data[1:])
            print("coherence score", "%1.4f" % (score) )
            if current_type == chosen_type: 
                chosen_type_coherence[int(foldername)] = "%1.4f" % (score)
        except:
            print(filename)

if chosen_type != None:
    list_coherence = []
    for value in sorted(chosen_type_coherence): list_coherence.append(chosen_type_coherence[value])
    print(chosen_type, list_coherence[::-1])

