import os
import numpy as np
import analysis_utils as utils
from scipy.stats import spearmanr, pearsonr
#from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau
import math


def _spearmanr(data1, data2, alpha=0.05, polarize = 1):
    coef, p = spearmanr(data1, data2)
    coef = coef*polarize
    print('Spearmans correlation coefficient: %.3f' % coef)
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)
    return coef, p


def _pearsonr(data1, data2, alpha=0.05):
    coef, p = pearsonr(data1, data2)
    print('Pearson correlation coefficient: %.3f' % coef)
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)
    return coef, p


def _kendalltau(data1, data2, alpha=0.05):
    coef, p = kendalltau(data1, data2)
    print('kendalltau correlation coefficient: %.3f' % coef)
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)
    return coef, p


def sort_dic(input_dic):
    return {k: v for k, v in sorted(input_dic.items(), key=lambda item: item[1])}


def load_text_data(data_dir):
    """
    Args:
      data_dir: Path to directory where data is stored.

    Returns:
      counts: A sparse matrix with shape [num_documents, num_words], representing
        the documents in a bag-of-words format.
      vocabulary: An array of strings with shape [num_words].
      author_indices: An array of integeres with shape [num_documents], where
        each entry represents the author who wrote the document.
      author_map: An array of strings with shape [num_authors], containing the
        names of each author.
    """
    vocabulary = np.loadtxt(
        os.path.join(data_dir, "vocabulary.txt"),
        dtype=str,
        delimiter="\n",
        comments="<!-")
    author_map = np.loadtxt(
        os.path.join(data_dir, "author_map.txt"),
        dtype=str,
        delimiter="\n")
    return vocabulary, author_map


def get_author_score_real(score_indices, author_indices, author_map):
    # Get average of brand score
    author_score = {}
    for i in range(len(score_indices)):
        if author_indices[i] in author_score:
            author_score[author_indices[i]
                         ] = author_score[author_indices[i]] + [score_indices[i]]
        else:
            author_score[author_indices[i]] = [score_indices[i]]
    author_score_real = {}
    for author_i in author_score:
        author_score_real[author_map[author_i]] = sum(
            author_score[author_i])/len(author_score[author_i])
    return author_score_real


def load_score_data(source_dir, time):
    data_dir = os.path.join(source_dir, "time", str(time))
    author_indices = np.load(
        os.path.join(data_dir, "author_indices.npy")).astype(np.int32)
    score_indices = np.load(
        os.path.join(data_dir, "score_indices.npy")).astype(np.int32)
    score_initial = np.load(
        os.path.join(data_dir, "score_indices_initial.npy")).astype(np.int32)
    return score_indices, author_indices, score_initial


def get_score_trend(rating_difference, rating_difference_real):
    if rating_difference*rating_difference_real>= 0:
        return 1
    else:
        return 0


def normalise(input_vec):
    output_vec = (input_vec - np.mean(input_vec[1:]))/np.std(input_vec[1:])
    return output_vec

    
if __name__ == "__main__":
    project_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir))
    source_dir = os.path.join(
        project_dir, "TBIP", "data", "beauty_makeupalley")

    # Load data.
    data_dir = os.path.join(source_dir, "clean")
    coherence_input_dir = os.path.join(source_dir, "input_words")
    if not os.path.exists(coherence_input_dir):
        os.makedirs(coherence_input_dir)

    (vocabulary, author_map) = load_text_data(data_dir)

    time_dir = os.path.join(source_dir, "time")
    time_choice = [int(filename) for filename in os.listdir(time_dir)]
    time_choice.sort(reverse=True)
    polarize = 1
    sentiment_label = ["negative: ", "neutral: ", "positive: "]
    output_trend = True


    collection_dictionary = {}
    brand_score_dictionary = {}
    brand_score_plots = {}
    rating_real_last = []
    rating_generated_last = None
    #real_brand_score_all = {}
    time_slice_index = 0

    for time in time_choice:
        print("time", time_slice_index)

        # Load author_indices, score_indices, product_indices
        score_indices, author_indices, score_initial = load_score_data(source_dir, time)
        author_score_real = get_author_score_real(score_indices, author_indices, author_map)

        # Load TBIP parameters.
        param_dir = os.path.join(source_dir, "tbip-fits", str(time), "params")

        objective_topic_loc = np.load(
            os.path.join(param_dir, "objective_topic_loc.npy"))
        objective_topic_scale = np.load(
            os.path.join(param_dir, "objective_topic_scale.npy"))
        ideological_topic_loc = np.load(
            os.path.join(param_dir, "ideological_topic_loc.npy"))
        ideological_topic_scale = np.load(
            os.path.join(param_dir, "ideological_topic_scale.npy"))
        ideal_point_loc = np.load(
            os.path.join(param_dir, "ideal_point_loc.npy"))
        ideal_point_scale = np.load(
            os.path.join(param_dir, "ideal_point_scale.npy"))

        save_topic_dir = os.path.join(coherence_input_dir, str(time_slice_index)+".txt")
        time_slice_index += 1        

        # Print ideal point orderings.
        author_score_generated = {}
        ideal_point_loc = ideal_point_loc
        rating_real = list([author_score_real[author] for author in author_map])
        for i in range(len(author_map)):
            author_score_real[author_map[i]] = rating_real[i]
        
        for i in range(len(author_map)):
            index = np.argsort(ideal_point_loc)[i]
            author_score_generated[author_map[index]] = ideal_point_loc[index]
        
        rating_generated = [author_score_generated[author] for author in author_map]

        try:
            if output_trend:
                if time == max(time_choice):
                    coef, _ = _spearmanr(rating_real, rating_generated)
                    polarize = math.copysign(1, coef)
                    if polarize < 0:
                        sentiment_label = sentiment_label[::-1]
                    initial_rating_generated = rating_generated
                brand_score_dictionary[time] = {"spearman": [_spearmanr(rating_real, rating_generated, polarize = polarize)], "pearson": [
                    _pearsonr(rating_real, rating_generated)], "kendalltau": [_kendalltau(rating_real, rating_generated)], "rating_real":rating_real,  "rating_generated": rating_generated}
        except:
            print("nan score error")
        
        collection_dictionary = utils.save_topic_words(objective_topic_loc,
                                                       objective_topic_scale,
                                                       ideological_topic_loc,
                                                       ideological_topic_scale,
                                                       vocabulary,
                                                       save_topic_dir,
                                                       collection_dictionary,
                                                       time,
                                                       sentiment_label = sentiment_label,
                                                       cut=1)

        brand_score_plots[time] = {"real":author_score_real, "generated":author_score_generated}
        print()

    with open(os.path.join(project_dir, "TBIP", "data", "beauty_makeupalley", "topic_words.txt"), "w") as f:
        f.close()
    brand_score_plots_sort = {}
    for time in brand_score_plots:
        brand_score_at_time = brand_score_plots[time]
        real_data = brand_score_at_time["real"]
        generated_data = brand_score_at_time["generated"]
        for brand in generated_data.keys():
            if brand in brand_score_plots_sort:
                brand_score_plots_sort[brand] += [round(generated_data[brand],5)]
                brand_score_plots_sort[brand + "real"] += [round(real_data[brand],5)]
            else:
                brand_score_plots_sort[brand] = [round(generated_data[brand],5)]
                brand_score_plots_sort[brand + "real"] = [round(real_data[brand],5)]
    print()

    table_save = []
    for brand in brand_score_plots_sort:
        brand_score_plots_sort[brand] = normalise(brand_score_plots_sort[brand])
        print("{name: 'TBIP-" + brand + "', type: 'line', data: " + str(brand_score_plots_sort[brand]) + ",},")
        table_save.append([brand]+list(brand_score_plots_sort[brand]))
    print()

    ACC_all = []
    MAE_all = []
    test_time_list = [i for i in range(len(time_choice))][1:]
    for brand in author_map:        
        brand_score_plots_sort[brand] = list(brand_score_plots_sort[brand])
        brand_score_plots_sort[brand+"real"] = list(brand_score_plots_sort[brand+"real"])
        score_difference = [abs(brand_score_plots_sort[brand][i] - brand_score_plots_sort[brand + "real"][i]) for i in test_time_list]
        MAE_all.append(sum(score_difference)/len(score_difference))
        correct_trend = [get_score_trend(brand_score_plots_sort[brand][i] - brand_score_plots_sort[brand][int(i-1)], brand_score_plots_sort[brand + "real"][i] - brand_score_plots_sort[brand + "real"][int(i-1)]) for i in test_time_list]
        brand_polarity = [get_score_trend(brand_score_plots_sort[brand][i], brand_score_plots_sort[brand + "real"][i]) for i in test_time_list]
        ACC_all.append(sum(brand_polarity)/len(brand_polarity))

    import csv
    with open(os.path.join(project_dir, "TBIP", "data", "beauty_makeupalley", "rating_table.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(table_save)

    for time in brand_score_dictionary:
        print("++++++++++ time", int(max(time_choice) - time), "++++++++++++++")
        print(brand_score_dictionary[time])
    print("ACC", sum(ACC_all)/len(ACC_all))
    print("MAE", sum(MAE_all)/len(MAE_all))

