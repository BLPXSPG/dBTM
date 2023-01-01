import os

project_dir = os.path.abspath(os.path.dirname(__file__))
source_dir = os.path.join(project_dir, "results")

coherence_type = ["C_A", "C_P", "C_V", "NPMI", "UCI", "UMass"] 

input_filename_list = [filename for filename in os.listdir(
        os.path.join(project_dir, "input_words"))]

for input_filename in input_filename_list:
    for type_ in coherence_type:
        directory = os.path.join(source_dir, input_filename[:-4])
        if not os.path.exists(directory):
            os.makedirs(directory)
        commend = "java -jar palmetto-0.1.0-jar-with-dependencies.jar ./wikipedia_bd " + type_ + " ./input_words/" + input_filename + " > ./results/" + input_filename[:-4] + "/" + type_ + input_filename
        os.system(commend)