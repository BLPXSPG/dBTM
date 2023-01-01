from collections import Counter
import os


def uniqueness(words_all):
    word_count = dict(Counter(words_all))
    L = len(word_count.keys())
    cnt = 0
    for word in word_count:
        cnt += 1/word_count[word]
    return cnt/L


project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__))) 
source_dir = os.path.join(project_dir,"input_words")

for filename in os.listdir(source_dir):
    file_dir = os.path.join(source_dir, filename)
    print("=======================================")
    print(filename)
    words_across_topics = []

    with open(file_dir, "r") as f:
        data = f.readlines()
        f.close()
    for i in range(len(data)):
        item = data[i]
        item = item.strip()
        item = item.split()
        if i%3 == 0:
            words_all = item
        else:
            words_all += item            
        if i%3 == 2:
            words_all = list(set(words_all))
            words_across_topics += words_all[:15]

            
    #print(words_across_topics)
    print("uniqueness x topics", uniqueness(words_across_topics))
