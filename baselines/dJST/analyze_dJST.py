import os
import json
from collections import Counter
from scipy.stats import spearmanr, pearsonr
import numpy as np



def update(time_slice, topic_idx, word, collection_dictionary, topic_saving_path):
    word = [word_i for word_i in word if len(word_i) > 2]
    print(time_slice, topic_idx, word)

    if topic_idx in collection_dictionary:
        current_record = collection_dictionary[topic_idx]
        if time_slice in current_record:
            current_record[time_slice] = current_record[time_slice] + [word[:10]]
        else:
            current_record[time_slice] = [word[:10]]
        collection_dictionary[topic_idx] = current_record
    else:
        collection_dictionary[topic_idx] = {time_slice: [word[:10]]}

    word_string = " ".join(word)
    word_string = word_string.split()
    word_string = list(set(word_string))
    word_string = " ".join(word_string[:10])
    with open(topic_saving_path, "a") as f:
        f.write(word_string)
        f.write('\n')
        f.close()
    return collection_dictionary


def elementwise_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def get_brand_score_real(score_indices, brand_indices, brand_map):
    # Get average of brand score
    brand_score = {}
    for i in range(len(score_indices)):
        if brand_indices[i] in brand_score:
            brand_score[brand_indices[i]
                         ] = brand_score[brand_indices[i]] + [score_indices[i]]
        else:
            brand_score[brand_indices[i]] = [score_indices[i]]
    brand_score_real = {}
    for brand_i in brand_score:
        #brand_i = int(brand_i)
        brand_score_real[brand_map[brand_i]] = sum(
            brand_score[brand_i])/len(brand_score[brand_i])
    print(brand_score_real)
    return brand_score_real


def load_score_data(source_dir, time):
    data_dir = os.path.join(source_dir, str(time))
    brand_indices = np.load(
        os.path.join(data_dir, "brand_indices.npy")).astype(np.int32)
    score_indices = np.load(
        os.path.join(data_dir, "score_indices.npy")).astype(np.int32)
    return score_indices, brand_indices


def normalise(input_vec):
    output_vec = (input_vec - np.mean(input_vec[1:]))/np.std(input_vec[1:])
    return output_vec


if __name__ == "__main__":
    project_dir = os.path.abspath(os.path.dirname(__file__))
    save_dir = os.path.join(project_dir, "out", "top_words")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    time_dir = os.path.join(project_dir, "data", "time")
    time_list = [int(filename) for filename in os.listdir(time_dir)]
    time_list.sort()

    #Restructure words in .twords to topics.json for sentence representation and top_words for coherence
    time_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    collection_dictionary = {}
    for time_slice in time_list:
        with open(os.path.join(project_dir, "out", "test", "epoch_" + str(int(time_slice-1)) + "-final.twords"), 'r') as f:
            data = f.readlines()
            f.close()
        label_idx = 0
        #topic_idx = 0
        word = []

        topic_saving_path = os.path.join(save_dir, str(time_slice)+".txt")
        with open(os.path.join(save_dir, str(time_slice)+".txt"), "w") as f:
            f.close()

        for i in range(len(data)):
            word_info = data[i]
            if word_info[:5] == "Label":
                if len(word) != 0:
                    collection_dictionary = update(time_slice, topic_idx, word, collection_dictionary, topic_saving_path)
                    word = []
                    label_idx += 1
                topic_idx = 0
                continue
            if word_info[:5] == "Topic":
                if len(word) != 0:
                    collection_dictionary = update(time_slice, topic_idx, word, collection_dictionary, topic_saving_path)
                    word = []
                    topic_idx += 1
            else:
                word_info = word_info.split(' ')
                word.append(word_info[0].strip())
        collection_dictionary = update(time_slice, topic_idx, word, collection_dictionary, topic_saving_path)
    with open(os.path.join(project_dir, "topics.json"), "w") as json_file:
        json.dump(collection_dictionary, json_file)
        json_file.close()



    #Calculate score of each brands
    label_type = [-1, 0, 1]
    with open(os.path.join(project_dir, "data", "brand_map.txt"), "r") as f:
        brand_map = f.readlines()
        brand_map = [word.strip() for word in brand_map]
        f.close()


    correlation = {}
    generated_score = {}
    for time_slice in time_list:

        score_indices, brand_indices = load_score_data(time_dir, time_slice)
        brand_score_real = get_brand_score_real(score_indices, brand_indices, brand_map)

        with open(os.path.join(project_dir, "out", "test", "epoch_" + str(int(time_slice-1)) + "-final.pi"), 'r') as f:
            data = f.readlines()
            f.close()
        load_file_path = os.path.join(time_dir, str(time_slice))
            
        brand_label = {}
        brand_count = {}

        print(len(brand_indices), len(data[:]), data[-1], data[0])
        for i in range(len(data[1:])):
            doc_info = data[i]
            doc_info = doc_info.split(' ')
            doc_info = doc_info[2:5]
            doc_info = [float(item) for item in doc_info]
            brand = brand_map[brand_indices[i]]

            if brand in brand_label:
                brand_label[brand] = elementwise_add(brand_label[brand], doc_info)
                brand_count[brand] = brand_count[brand] + 1
            else:
                brand_label[brand] = doc_info
                brand_count[brand] = 1

            if i%10000 == 0:
                print(i, brand, brand_count[brand], brand_label[brand])
        
        rating_real = []
        rating_generated = []
        for brand in brand_label:
            brand_label_key = brand_label[brand]
            brand_count_key = brand_count[brand]
            
            brand_label_distri = [item/brand_count_key for item in brand_label_key]
            final_score = sum(label_type[i]*brand_label_distri[i] for i in range(len(label_type)))
            #print(brand, brand_label_key, brand_count_key, brand_label_distri, final_score)
            print((brand + ": "), final_score)
            if brand in generated_score:
                generated_score[brand] += [final_score]
            else:
                generated_score[brand] = [final_score]
            rating_generated.append(final_score)
            rating_real.append(brand_score_real[brand])
        correlation[time_slice] = {"spearman":[spearmanr(rating_real, rating_generated)], "pearsonr":[pearsonr(rating_real, rating_generated)]}


    print()
    table_save = []
    for brand in generated_score:
        generated_score[brand] = normalise(generated_score[brand])
        print("{name: 'dJST-" + brand + "', type: 'line', data: " + str(generated_score[brand]) + ",},")
        table_save.append([brand]+generated_score[brand])
    print()

    import csv
    with open(os.path.join(project_dir, "out", "rating_table.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(table_save)
        

    print(correlation)
    print(time_list)