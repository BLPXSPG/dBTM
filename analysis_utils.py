"""Helpful functions for analysis."""

import numpy as np
import os
import scipy.sparse as sparse
from scipy.stats import bernoulli, poisson


def load_text_data(data_dir):
  """Load text data used to train the model.

  Args:
    data_dir: Path to directory where data is stored.

  Returns:
    counts: A sparse matrix with shape [num_documents, num_words], representing
      the documents in a bag-of-words format.
    vocabulary: An array of strings with shape [num_words].
    brand_indices: An array of integeres with shape [num_documents], where
      each entry represents the brand who wrote the document.
    brand_map: An array of strings with shape [num_brands], containing the
      names of each brand.
    raw_documents: A string vector with shape [num_documents] containing the
      raw documents.
  """
  counts = sparse.load_npz(
      os.path.join(data_dir, "counts.npz"))
  vocabulary = np.loadtxt(
      os.path.join(data_dir, "vocabulary.txt"),
      dtype=str,
      delimiter="\n",
      comments="<!-")
  brand_indices = np.load(
      os.path.join(data_dir, "brand_indices.npy")).astype(np.int32)
  brand_map = np.loadtxt(
      os.path.join(data_dir, "brand_map.txt"),
      dtype=str,
      delimiter="\n")
  raw_documents = np.loadtxt(
      os.path.join(data_dir, "raw_documents.txt"),
      dtype=str,
      delimiter="\n",
      comments="<!-")
  return counts, vocabulary, brand_indices, brand_map, raw_documents


def get_ideological_topic_means(objective_topic_loc,
                                objective_topic_scale,
                                ideological_topic_loc,
                                ideological_topic_scale):
  """Returns neutral and ideological topics from variational parameters.

  For each (k,v), we want to evaluate E[beta_kv], E[beta_kv * exp(eta_kv)],
  and E[beta_kv * exp(-eta_kv)], where the expectations are with respect to the
  variational distributions. Like the paper, beta refers to the obective topic
  and eta refers to the ideological topic.

  Dropping the indices and denoting by mu_b the objective topic location and
  sigma_b the objective topic scale, we have E[beta] = exp(mu + sigma_b^2 / 2),
  using the mean of a lognormal distribution.

  Denoting by mu_e the ideological topic location and sigma_e the ideological
  topic scale, we have E[beta * exp(eta)] = E[beta]E[exp(eta)] by the
  mean-field assumption. exp(eta) is lognormal distributed, so E[exp(eta)] =
  exp(mu_e + sigma_e^2 / 2). Thus, E[beta * exp(eta)] =
  exp(mu_b + mu_e + (sigma_b^2 + sigma_e^2) / 2).

  Finally, E[beta * exp(-eta)] =
  exp(mu_b - mu_e + (sigma_b^2 + sigma_e^2) / 2).

  Because we only care about the orderings of topics, we can drop the exponents
  from the means.

  Args:
    objective_topic_loc: Variational lognormal location parameter for the
      objective topic (beta). Should be shape [num_topics, num_words].
    objective_topic_scale: Variational lognormal scale parameter for the
      objective topic (beta). Should be positive, with shape
      [num_topics, num_words].
    ideological_topic_loc: Variational Gaussian location parameter for the
      ideological topic (eta). Should be shape [num_topics, num_words].
    ideological_topic_scale: Variational Gaussian scale parameter for the
      ideological topic (eta). Should be positive, with shape
      [num_topics, num_words].

  Returns:
    neutral_mean: A matrix with shape [num_topics, num_words] denoting the
      variational mean for the neutral topics.
    positive_mean: A matrix with shape [num_topics, num_words], denoting the
      variational mean for the ideological topics with an ideal point of +1.
    negative_mean: A matrix with shape [num_topics, num_words], denoting the
      variational mean for the ideological topics with an ideal point of -1.
  """
  neutral_mean = objective_topic_loc + objective_topic_scale ** 2 / 2
  positive_mean = (objective_topic_loc +
                   ideological_topic_loc +
                   (objective_topic_scale ** 2 +
                    ideological_topic_scale ** 2) / 2)
  negative_mean = (objective_topic_loc -
                   ideological_topic_loc +
                   (objective_topic_scale ** 2 +
                    ideological_topic_scale ** 2) / 2)
  return neutral_mean, positive_mean, negative_mean


def print_topics(objective_topic_loc,
                 objective_topic_scale,
                 ideological_topic_loc,
                 ideological_topic_scale,
                 vocabulary,
                 words_per_topic=10):
  """Prints neutral and ideological topics from variational parameters.

  Args:
    objective_topic_loc: Variational lognormal location parameter for the
      objective topic (beta). Should be shape [num_topics, num_words].
    objective_topic_scale: Variational lognormal scale parameter for the
      objective topic (beta). Should be positive, with shape
      [num_topics, num_words].
    ideological_topic_loc: Variational Gaussian location parameter for the
      ideological topic (eta). Should be shape [num_topics, num_words].
    ideological_topic_scale: Variational Gaussian scale parameter for the
      ideological topic (eta). Should be positive, with shape
      [num_topics, num_words].
    vocabulary: A list of strings with shape [num_words].
    words_per_topic: The number of words to print for each topic.
  """

  neutral_mean, positive_mean, negative_mean = get_ideological_topic_means(
      objective_topic_loc,
      objective_topic_scale,
      ideological_topic_loc,
      ideological_topic_scale)
  num_topics, num_words = neutral_mean.shape

  top_neutral_words = np.argsort(-neutral_mean, axis=1)
  top_negative_words = np.argsort(-negative_mean, axis=1)
  top_positive_words = np.argsort(-positive_mean, axis=1)
  topic_strings = []
  for topic_idx in range(num_topics):
    neutral_start_string = "Neutral {}:".format(topic_idx)
    neutral_row = [vocabulary[word] for word in
                   top_neutral_words[topic_idx, :words_per_topic]]
    neutral_row_string = ", ".join(neutral_row)
    neutral_string = " ".join([neutral_start_string, neutral_row_string])

    positive_start_string = "Positive {}:".format(topic_idx)
    positive_row = [vocabulary[word] for word in
                    top_positive_words[topic_idx, :words_per_topic]]
    positive_row_string = ", ".join(positive_row)
    positive_string = " ".join([positive_start_string, positive_row_string])

    negative_start_string = "Negative {}:".format(topic_idx)
    negative_row = [vocabulary[word] for word in
                    top_negative_words[topic_idx, :words_per_topic]]
    negative_row_string = ", ".join(negative_row)
    negative_string = " ".join([negative_start_string, negative_row_string])

    topic_strings.append(negative_string)
    topic_strings.append(neutral_string)
    topic_strings.append(positive_string)
    topic_strings.append("==========")

  print("{}\n".format(np.array(topic_strings)))


def get_expected_word_count(word,
                            ideal_point,
                            document_mean,
                            objective_topic_mean,
                            ideological_topic_mean,
                            vocabulary):
  """Gets expected count for a word and ideal point using fitted topics.

  Args:
    word: The word we want to query, a string.
    ideal_point: The ideal point to compute the expectation.
    document_mean: A vector with shape [num_topics], representing the
      document intensities for the word count we want to evaluate.
    objective_topic_mean: A matrix with shape [num_topics, num_words],
      representing the fitted objective topics (beta).
    ideological_topic_mean: A matrix with shape [num_topics, num_words],
      representing the fitted ideological topics (eta).
    vocabulary: [vocabulary: An array of strings with shape [num_words].

  Returns:
    expected_word_count: A scalar representing the expected word count for
      the queried word, ideal point, and document intensities.
  """
  word_index = np.where(vocabulary == word)[0][0]
  expected_word_count = np.dot(
      document_mean,
      objective_topic_mean[:, word_index] *
      np.exp(ideal_point * ideological_topic_mean[:, word_index]))
  return expected_word_count


def get_term_score(topic_means):
  import math
  #topic_means = np.array(topic_means)
  topic_means = (topic_means - np.min(topic_means)) / (np.max(topic_means) - np.min(topic_means))
  term_score = np.zeros((len(topic_means),len(topic_means[0])))
  
  for v in range(len(topic_means[0])):
    normalize_term = [topic_means[j][v] for j in range(len(topic_means))]
    term_prob = np.prod(np.array(normalize_term))**(1/len(topic_means))
    #for k in range(len(topic_means)):
    #  normalize_term.append(math.log(topic_means[k][v]))
    #term_prob = sum([math.log(topic_means[k][v]) for k in range(len(topic_means))])
    #term_prob = term_prob/len(topic_means)
    for k in range(len(topic_means)):
      term_score[k][v] = topic_means[k][v]*(math.log(topic_means[k][v]/term_prob))
  return term_score


def cut_topics(positive_mean, neutral_mean, negative_mean, cut):
  topic_means = []
  cut_slice_p = (positive_mean - neutral_mean)/cut
  cut_slice_n = (neutral_mean - negative_mean)/cut
  topic_means.append(negative_mean)
  for i in range(cut):
    topic_means.append(negative_mean + cut_slice_n*(i+1))
  for i in range(cut):
    topic_means.append(neutral_mean + cut_slice_p*(i+1))    
  return topic_means


def save_topic_words(objective_topic_loc, 
                 objective_topic_scale,
                 ideological_topic_loc, 
                 ideological_topic_scale, 
                 vocabulary,
                 save_dir,
                 collection_dictionary,
                 time_slice,
                 rescore_filter_num = 50,
                 cut = 1,
                 words_per_topic=10,
                 sentiment_label = ["negative: ", "neutral: ", "positive: "],
                 use_term_score = False,
                 print_topic = True):
  """Prints neutral and ideological topics from variational parameters.
  
  Args:
    objective_topic_loc: Variational lognormal location parameter for the 
      objective topic (beta). Should be shape [num_topics, num_words].
    objective_topic_scale: Variational lognormal scale parameter for the 
      objective topic (beta). Should be positive, with shape 
      [num_topics, num_words].
    ideological_topic_loc: Variational Gaussian location parameter for the 
      ideological topic (eta). Should be shape [num_topics, num_words].
    ideological_topic_scale: Variational Gaussian scale parameter for the 
      ideological topic (eta). Should be positive, with shape 
      [num_topics, num_words].
    vocabulary: A list of strings with shape [num_words].
    words_per_topic: The number of words to print for each topic.
    rescore_filter_num: number of top words chosen to rescore by tfidf
  """
  
  neutral_mean, positive_mean, negative_mean = get_ideological_topic_means(
      objective_topic_loc, 
      objective_topic_scale,
      ideological_topic_loc, 
      ideological_topic_scale)
  num_topics, num_words = neutral_mean.shape
  topic_means = cut_topics(positive_mean, neutral_mean, negative_mean, cut)
  if use_term_score:
    neutral_mean_rescore = get_term_score(neutral_mean)
    positive_mean_rescore = get_term_score(positive_mean)
    negative_mean_rescore = get_term_score(negative_mean)
    topic_means_rescore = cut_topics(positive_mean_rescore, neutral_mean_rescore, negative_mean_rescore, cut)

  # print(topic_means)
  with open(save_dir, "w") as f:
    for topic_idx in range(num_topics):
      if print_topic: print("#################", topic_idx, "##################")
      word_string_all = []
      for i in range(len(topic_means)):
        top_words = np.argsort(-topic_means[i], axis=1)
        if use_term_score:
          top_words = top_words[topic_idx, :rescore_filter_num]
          top_words_rescore = np.argsort(-topic_means_rescore[i], axis=1)
          top_words_rescore = top_words_rescore[topic_idx, :]
          #print("top_words_rescore_num", len(top_words_rescore))
          words_row = [vocabulary[word] for word in top_words_rescore if word in top_words]
          words_row = words_row[:words_per_topic]
        else: 
          words_row = [vocabulary[word] for word in 
                      top_words[topic_idx, :words_per_topic]]

        if print_topic: print(sentiment_label[i], words_row)

        word_string = " ".join(words_row)
        word_string = word_string.split()
        word_string = list(set(word_string))
        word_string = " ".join(word_string[:10])
        f.write(word_string)
        f.write('\n')

        word_string_all.append(words_row)

      if topic_idx in collection_dictionary:
        current_record = collection_dictionary[topic_idx]
        current_record[time_slice] = word_string_all
        collection_dictionary[topic_idx] = current_record
      else:
        collection_dictionary[topic_idx] = {time_slice: word_string_all}
    f.close()
  return collection_dictionary

