from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import functools
import os
import time

from absl import flags
import numpy as np
import scipy.sparse as sparse
import tensorflow as tf

import tensorflow_probability as tfp
from collections import Counter


flags.DEFINE_float("learning_rate",
                   default=0.01,
                   help="Adam learning rate.")
flags.DEFINE_integer("max_steps",
                     default=50000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("add_steps",
                     default=50000,
                     help="Number of fine tuning steps to run based on classification loss.")
flags.DEFINE_integer("num_topics",
                     default=30,
                     help="Number of topics.")
flags.DEFINE_integer("batch_size",
                     default=256,
                     help="Batch size.")
flags.DEFINE_boolean("test",
                     default=True,
                     help="Whether to split data into train and test set for testing accuracy")
flags.DEFINE_integer("num_samples",
                     default=1,
                     help="Number of samples to use for ELBO approximation.")
flags.DEFINE_enum("counts_transformation",
                  default="nothing",
                  enum_values=["nothing", "binary", "sqrt", "log"],
                  help="Transformation used on counts data.")
flags.DEFINE_boolean("pre_initialize_parameters",
                     default=True,
                     help="Whether to use pre-initialized document and topic "
                          "intensities (with Poisson factorization).")
flags.DEFINE_string("data",
                    default="beauty_makeupalley",
                    help="Data source being used.")
flags.DEFINE_integer("senate_session",
                     default=113,
                     help="Senate session (used only when data is "
                          "'senate-speech-comparisons'.")
flags.DEFINE_integer("print_steps",
                     default=500,
                     help="Number of steps to print and save results.")
flags.DEFINE_integer("check_steps",
                     default=5000,
                     help="Number of steps to check whether stop training and save parameters.")
flags.DEFINE_integer("seed",
                     default=123,
                     help="Random seed to be used.")
flags.DEFINE_float("dev_ratio",
                   default=0.1,
                   help="Split rate of data to dev and train")

FLAGS = flags.FLAGS


def build_database(random_state, num_documents, counts_transformation, counts, author_indices, score_indices, batch_size, dev_ratio, dev=True, balance=True):
    # Shuffle data.
    print(counts.dtype)
    documents = random_state.permutation(num_documents)
    shuffled_author_indices = author_indices[documents]
    shuffled_score_indices = score_indices[documents]
    shuffled_counts = counts[documents].astype(np.float32)
    print(shuffled_counts.dtype)

    # Apply counts transformation.
    if counts_transformation == "nothing":
        count_values = shuffled_counts.data
    elif counts_transformation == "binary":
        count_values = np.int32(shuffled_counts.data > 0)
    elif counts_transformation == "log":
        count_values = np.round(np.log(1 + shuffled_counts.data))
    elif counts_transformation == "sqrt":
        count_values = np.round(np.sqrt(shuffled_counts.data))
    else:
        raise ValueError("Unrecognized counts transformation.")
    # Store counts as sparse tensor so it occupies less memory.
    print(shuffled_counts.dtype)
    shuffled_counts = tf.SparseTensor(
        indices=np.array(shuffled_counts.nonzero()).T,
        values=count_values,
        dense_shape=shuffled_counts.shape)
    print(shuffled_counts.dtype)
    dataset = tf.data.Dataset.from_tensor_slices(
        (documents, shuffled_counts, shuffled_author_indices, shuffled_score_indices))

    if dev:
        dev_num = int(num_documents*dev_ratio)
        dev_dataset = dataset.take(dev_num)
        dataset = dataset.skip(dev_num)
        dev_batches = dev_dataset.repeat().batch(batch_size).prefetch(batch_size)
        dev_iterator = dev_batches.make_one_shot_iterator()
    else:
        dev_iterator = 0

    if balance:
        # Here we balance the pos/neg reviews in mini-batch
        ds_pos = dataset.filter(lambda documents, shuffled_counts, shuffled_author_indices,
                                shuffled_score_indices: tf.reshape(tf.equal(shuffled_score_indices, 2), []))
        ds_neu = dataset.filter(lambda documents, shuffled_counts, shuffled_author_indices,
                                shuffled_score_indices: tf.reshape(tf.equal(shuffled_score_indices, 1), []))
        ds_neg = dataset.filter(lambda documents, shuffled_counts, shuffled_author_indices,
                                shuffled_score_indices: tf.reshape(tf.equal(shuffled_score_indices, 0), []))
        # ds_neg = ds_neg.concatenate(ds_neu)
        ds_neg = ds_neg.repeat()
        ds_pos = ds_pos.concatenate(ds_neu)
        dataset_new = tf.data.Dataset.zip((ds_pos, ds_neg))
        dataset = dataset_new.flat_map(
            lambda ex_pos, ex_neg: tf.data.Dataset.from_tensors(ex_pos).concatenate(
                tf.data.Dataset.from_tensors(ex_neg)))

    batches = dataset.repeat().batch(batch_size).prefetch(batch_size)
    iterator = batches.make_one_shot_iterator()

    return iterator, dev_iterator


def load_data(data_dir):
    counts = sparse.load_npz(os.path.join(data_dir, "counts.npz"))
    num_documents, num_words = counts.shape
    print(num_documents, num_words)
    author_indices = np.load(
        os.path.join(data_dir, "author_indices.npy")).astype(np.int32)
    score_indices = np.load(
        os.path.join(data_dir, "score_indices.npy")).astype(np.int32)
    num_authors = np.max(author_indices + 1)

    return num_documents, num_words, counts, author_indices, score_indices, num_authors


def build_input_pipeline(source_dir,
                         batch_size,
                         random_state,
                         time_slice,
                         dev_ratio,
                         counts_transformation="nothing",
                         test=True):
    """Load data and build iterator for minibatches.
    Args:
      data_dir: The directory where the data is located. There must be four
        files inside the rep: `counts.npz`, `author_indices.npy`,
        `author_map.txt`, and `vocabulary.txt`.
      batch_size: The batch size to use for training.
      random_state: A NumPy `RandomState` object, used to shuffle the data.
      counts_transformation: A string indicating how to transform the counts.
        One of "nothing", "binary", "log", or "sqrt".
    """
    data_dir = os.path.join(source_dir, "clean")
    if test:
        train_dir = os.path.join(source_dir, "time", str(time_slice))
        num_documents, num_words, counts, author_indices, score_indices, num_authors = load_data(
            train_dir)
        iterator, dev_iterator = build_database(random_state, num_documents, counts_transformation,
                                                counts, author_indices, score_indices, batch_size, dev_ratio)

        test_iterator, _ = build_database(random_state, num_documents, counts_transformation,
                                          counts, author_indices, score_indices, batch_size, dev_ratio, dev=False, balance=False)
    else:
        num_documents, num_words, counts, author_indices, score_indices, num_authors = load_data(
            data_dir)
        iterator, dev_iterator = build_database(random_state, num_documents, counts_transformation,
                                                counts, author_indices, score_indices, batch_size, dev_ratio)
        test_iterator = 0

    author_map = np.loadtxt(os.path.join(data_dir, "author_map.txt"),
                            dtype=str,
                            delimiter="\n")
    vocabulary = np.loadtxt(os.path.join(data_dir, "vocabulary.txt"),
                            dtype=str,
                            delimiter="\n",
                            comments="<!-")

    total_counts_per_author = np.bincount(
        author_indices,
        weights=np.array(np.sum(counts, axis=1)).flatten())
    counts_per_document_per_author = (
        total_counts_per_author / np.bincount(author_indices))
    # Author weights is how much lengthy each author's opinion over average is.
    author_weights = (counts_per_document_per_author /
                      np.mean(np.sum(counts, axis=1))).astype(np.float32)
    return (iterator, dev_iterator, test_iterator, author_weights, vocabulary, author_map,
            num_documents, num_words, num_authors)


def build_lognormal_variational_parameters(initial_document_loc,
                                           initial_objective_topic_loc,
                                           num_documents,
                                           num_words,
                                           num_topics):
    """
    Build document and objective topic lognormal variational parameters.

    Args:
      initial_document_loc: A [num_documents, num_topics] NumPy array containing
        the initial document intensity means.
      initial_objective_topic_loc: A [num_topics, num_words] NumPy array
        containing the initial objective topic means.
      num_documents: Number of documents in the data set.
      num_words: Number of words in the data set.
      num_topics: Number of topics.

    Returns:
      document_loc: A Variable object with shape [num_documents, num_topics].
      document_scale: A positive Variable object with shape [num_documents,
        num_topics].
      objective_topic_loc: A Variable object with shape [num_topics, num_words].
      objective_topic_scale: A positive Variable object with shape [num_topics,
        num_words].
    """
    print(num_words, num_documents)
    document_loc = tf.get_variable(
        "document_loc",
        initializer=tf.constant(np.log(initial_document_loc)))
    objective_topic_loc = tf.get_variable(
        "objective_topic_loc",
        initializer=tf.constant(np.log(initial_objective_topic_loc)))
    document_scale_logit = tf.get_variable(
        "document_scale_logit",
        shape=[num_documents, num_topics],
        initializer=tf.initializers.random_normal(mean=0, stddev=1.),
        dtype=tf.float32)
    objective_topic_scale_logit = tf.get_variable(
        "objective_topic_scale_logit",
        shape=[num_topics, num_words],
        initializer=tf.initializers.random_normal(mean=0, stddev=1.),
        dtype=tf.float32)
    document_scale = tf.nn.softplus(document_scale_logit)
    objective_topic_scale = tf.nn.softplus(objective_topic_scale_logit)

    tf.summary.histogram("params/document_loc", document_loc)
    tf.summary.histogram("params/objective_topic_loc", objective_topic_loc)
    tf.summary.histogram("params/document_scale", document_scale)
    tf.summary.histogram("params/objective_topic_scale", objective_topic_scale)

    return (document_loc, document_scale,
            objective_topic_loc, objective_topic_scale)


def print_topics(neutral_mean, negative_mean, positive_mean, vocabulary):
    """Get neutral and ideological topics to be used for Tensorboard.

    Args:
      neutral_mean: The mean of the neutral topics, a NumPy matrix with shape
        [num_topics, num_words].
      negative_mean: The mean of the negative topics, a NumPy matrix with shape
        [num_topics, num_words].
      positive_mean: The mean of the positive topics, a NumPy matrix with shape
        [num_topics, num_words].
      vocabulary: A list of the vocabulary with shape [num_words].

    Returns:
      topic_strings: A list of the negative, neutral, and positive topics.
    """
    num_topics, num_words = neutral_mean.shape
    words_per_topic = 10
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
        positive_string = " ".join(
            [positive_start_string, positive_row_string])

        negative_start_string = "Negative {}:".format(topic_idx)
        negative_row = [vocabulary[word] for word in
                        top_negative_words[topic_idx, :words_per_topic]]
        negative_row_string = ", ".join(negative_row)
        negative_string = " ".join(
            [negative_start_string, negative_row_string])

        topic_strings.append("  \n".join(
            [negative_string, neutral_string, positive_string]))
    return np.array(topic_strings)


def print_ideal_points(ideal_point_loc, author_map):
    """Print ideal point ordering for Tensorboard."""
    return ", ".join(author_map[np.argsort(ideal_point_loc)])


def get_log_prior(samples, prior):
    """Return log prior of sampled Gaussians.

    Args:
      samples: A `Tensor` with shape `[num_samples, :, :]`.
      prior: String representing prior distribution.

    Returns:
      log_prior: A `Tensor` with shape `[num_samples]`, with the log priors
        summed across latent dimensions.
    """
    if prior == 'normal':
        prior_distribution = tfp.distributions.Normal(loc=0., scale=1.)
    elif prior == 'gamma':
        prior_distribution = tfp.distributions.Gamma(
            concentration=0.3, rate=0.3)
    log_prior = tf.reduce_sum(prior_distribution.log_prob(samples),
                              axis=[1, 2])
    return log_prior


def sample_gumbel(shape, eps=1e-15):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=eps, maxval=1)
    return -tf.log(-tf.log(U) + eps)


def gumbel_softmax_sample(logits, temperature, eps=1e-6):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = tf.log(logits + eps) + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(
            tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def get_sampling_setting(class_limit=4):
    # class_limit: for each word we limit the count to be 1,2,3,4 or above 4(the class_limit)
    class_list = [i for i in range(class_limit)]
    class_list = tf.constant(class_list, dtype=tf.float32)
    outcome_vector = list(range(class_limit+1))
    outcome_vector = tf.constant(outcome_vector, dtype=tf.float32)
    k_factorial = tf.exp(tf.lgamma(class_list+1))
    return class_limit, outcome_vector, k_factorial, class_list


def example_from_rate(rate, class_limit, class_list, outcome_vector, k_factorial, batch_size, num_words, num_labels, hard=False, temperature=1):
    rate_tensor = rate[:, :, :, tf.newaxis]

    rate_k = rate_tensor**class_list
    exp_rate = tf.math.exp(-rate_tensor)
    prob_list = rate_k * exp_rate/k_factorial

    rest_prob = 1 - tf.reduce_sum(prob_list, axis=-1)
    rest_prob = rest_prob[:, :, :, tf.newaxis]
    final_prob_list = tf.concat([prob_list, rest_prob], axis=-1)

    count_distribution_samples = gumbel_softmax(
        final_prob_list, temperature, hard=hard)
    count_samples = tf.reduce_sum(tf.multiply(
        count_distribution_samples, outcome_vector), axis=-1)
    # print_out = tf.print(count_distribution_samples, [count_distribution_samples], summarize=-1)

    selected_prediction = tf.keras.layers.Dense(num_labels)(
        tf.reshape(count_samples, [batch_size, num_words]))

    return selected_prediction


def get_elbo(counts,
             document_indices,
             author_indices,
             author_weights,
             document_distribution,
             objective_topic_distribution,
             ideological_topic_distribution,
             ideal_point_distribution,
             num_documents,
             batch_size,
             score_indices,
             time_slice,
             loss_weight=1.,
             num_labels=3,
             num_samples=1):
    """Approximate variational Lognormal ELBO using reparameterization.

    Args:
      counts: A matrix with shape `[batch_size, num_words]`.
      document_indices: An int-vector with shape `[batch_size]`.
      author_indices: An int-vector with shape `[batch_size]`.
      author_weights: A vector with shape `[num_authors]`, constituting how
        lengthy the opinion is above average.
      document_distribution: A positive `Distribution` object with parameter
        shape `[num_documents, num_topics]`.
      objective_topic_distribution: A positive `Distribution` object with
        parameter shape `[num_topics, num_words]`.
      ideological_topic_distribution: A positive `Distribution` object with
        parameter shape `[num_topics, num_words]`.
      ideal_point_distribution: A `Distribution` object over [0, 1] with
        parameter_shape `[num_authors]`.
      num_documents: The number of documents in the total data set (used to
        calculate log-likelihood scale).
      batch_size: Batch size (used to calculate log-likelihood scale).
      num_samples: Number of Monte-Carlo samples.

    Returns:
      elbo: A scalar representing a Monte-Carlo sample of the ELBO. This value is
        averaged across samples and summed across batches.
    """
    document_samples = document_distribution.sample(num_samples)
    objective_topic_samples = objective_topic_distribution.sample(num_samples)

    ideological_topic_samples = ideological_topic_distribution.sample(
        num_samples)
    ideal_point_samples = ideal_point_distribution.sample(num_samples)

    _, num_topics, num_words = objective_topic_samples.get_shape().as_list()

    ideal_point_log_prior = tfp.distributions.Normal(
        loc=0.,
        scale=1.)
    ideal_point_log_prior = tf.reduce_sum(
        ideal_point_log_prior.log_prob(ideal_point_samples), axis=1)

    document_log_prior = get_log_prior(document_samples, 'gamma')
    objective_topic_log_prior = get_log_prior(objective_topic_samples, 'gamma')
    ideological_topic_log_prior = get_log_prior(ideological_topic_samples,
                                                'normal')
    log_prior = (document_log_prior +
                 objective_topic_log_prior +
                 ideological_topic_log_prior +
                 ideal_point_log_prior)

    selected_document_samples = tf.gather(document_samples,
                                          document_indices,
                                          axis=1)
    selected_ideal_points = tf.gather(ideal_point_samples,
                                      author_indices,
                                      axis=1)

    # polarity score
    selected_ideological_topic_samples = tf.exp(
        selected_ideal_points[:, :, tf.newaxis, tf.newaxis] *
        ideological_topic_samples[:, tf.newaxis, :, :])
    # reversed polarity score
    reversed_ideological_topic_samples = tf.exp(
        - selected_ideal_points[:, :, tf.newaxis, tf.newaxis] *
        ideological_topic_samples[:, tf.newaxis, :, :])

    # copy objective_topic_samples batch_size
    copy_num = tf.constant([1, batch_size, 1, 1], tf.int32)
    objective_topic_samples_copys = tf.tile(
        objective_topic_samples[:, tf.newaxis, :, :], copy_num)

    # Normalize by how lengthy the author's opinion is.
    selected_author_weights = tf.gather(author_weights, author_indices)
    selected_ideological_topic_samples = (
        selected_author_weights[tf.newaxis, :, tf.newaxis, tf.newaxis] *
        selected_ideological_topic_samples)
    reversed_ideological_topic_samples = (
        selected_author_weights[tf.newaxis, :, tf.newaxis, tf.newaxis] *
        reversed_ideological_topic_samples)

    document_entropy = -tf.reduce_sum(
        document_distribution.log_prob(document_samples),
        axis=[1, 2])
    objective_topic_entropy = -tf.reduce_sum(
        objective_topic_distribution.log_prob(objective_topic_samples),
        axis=[1, 2])

    test_prob = objective_topic_distribution.prob(objective_topic_samples)
    test_entropy = -tf.reduce_sum(
        test_prob,
        axis=[1, 2])

    ideological_topic_entropy = -tf.reduce_sum(
        ideological_topic_distribution.log_prob(ideological_topic_samples),
        axis=[1, 2])
    ideal_point_entropy = -tf.reduce_sum(
        ideal_point_distribution.log_prob(ideal_point_samples),
        axis=1)
    entropy = (document_entropy +
               objective_topic_entropy +
               ideological_topic_entropy +
               ideal_point_entropy)

    class_limit, outcome_vector, k_factorial, class_list = get_sampling_setting()
    # original rate
    rate = tf.reduce_sum(
        selected_document_samples[:, :, :, tf.newaxis] *
        objective_topic_samples[:, tf.newaxis, :, :] *
        selected_ideological_topic_samples[:, :, :, :],
        axis=2)
    selected_prediction = example_from_rate(
        rate, class_limit, class_list, outcome_vector, k_factorial, batch_size, num_words, num_labels)
    # This is for elbo calculation later
    poisson_count_distribution = tfp.distributions.Poisson(rate=rate)

    # rate based on reversed ideological topic samples
    reversed_rate = tf.reduce_sum(
        selected_document_samples[:, :, :, tf.newaxis] *
        objective_topic_samples[:, tf.newaxis, :, :] *
        selected_ideological_topic_samples[:, :, :, :],
        axis=2)
    reversed_prediction = example_from_rate(
        reversed_rate, class_limit, class_list, outcome_vector, k_factorial, batch_size, num_words, num_labels)

    score_indices = tf.convert_to_tensor(score_indices)

    given_labels = tf.one_hot(score_indices, num_labels)
    loss_real = tf.nn.softmax_cross_entropy_with_logits(
        labels=given_labels, logits=selected_prediction)
    reversed_score = (num_labels - 1) - score_indices
    reversed_labels = tf.one_hot(reversed_score, num_labels)
    loss_reversed = tf.nn.softmax_cross_entropy_with_logits(
        labels=reversed_labels, logits=reversed_prediction)

    score_loss = loss_weight * (loss_reversed + loss_real)
    score_loss = tf.math.reduce_sum(score_loss)

    # Need to un-sparsify the counts to evaluate log-likelihood.
    count_log_likelihood = poisson_count_distribution.log_prob(
        tf.sparse.to_dense(counts))  # add small term? + 1e-10 20201225
    count_log_likelihood = tf.reduce_sum(count_log_likelihood, axis=[1, 2])
    # Adjust for the fact that we're only using a minibatch.
    print(count_log_likelihood, num_documents / batch_size)
    count_log_likelihood = count_log_likelihood * (num_documents / batch_size)

    elbo = log_prior + count_log_likelihood + entropy
    elbo = tf.reduce_mean(elbo)

    return elbo, score_loss,


def initial_distribution(time_slice, source_dir, num_documents, num_words, num_authors, random_state):
    # if pre_initialize_parameters:
    fit_dir = os.path.join(source_dir, "pf-fits", str(time_slice))
    fitted_document_shape = np.load(
        os.path.join(fit_dir, "document_shape.npy")).astype(np.float32)
    fitted_document_rate = np.load(
        os.path.join(fit_dir, "document_rate.npy")).astype(np.float32)
    fitted_topic_shape = np.load(
        os.path.join(fit_dir, "topic_shape.npy")).astype(np.float32)
    fitted_topic_rate = np.load(
        os.path.join(fit_dir, "topic_rate.npy")).astype(np.float32)

    initial_document_loc = fitted_document_shape / fitted_document_rate
    initial_objective_topic_loc = fitted_topic_shape / fitted_topic_rate

    # Initialize lognormal variational parameters.
    (document_loc, document_scale, objective_topic_loc,
     objective_topic_scale) = build_lognormal_variational_parameters(
        initial_document_loc,
        initial_objective_topic_loc,
        num_documents,
        num_words,
        FLAGS.num_topics)
    document_distribution = tfp.distributions.LogNormal(
        loc=document_loc,
        scale=document_scale)
    objective_topic_distribution = tfp.distributions.LogNormal(
        loc=objective_topic_loc,
        scale=objective_topic_scale)

    ideological_topic_loc = tf.get_variable(
        "ideological_topic_loc",
        shape=[FLAGS.num_topics, num_words],
        dtype=tf.float32)
    ideological_topic_scale_logit = tf.get_variable(
        "ideological_topic_scale_logit",
        shape=[FLAGS.num_topics, num_words],
        dtype=tf.float32)
    ideological_topic_scale = tf.nn.softplus(ideological_topic_scale_logit)
    tf.summary.histogram("params/ideological_topic_loc", ideological_topic_loc)
    tf.summary.histogram("params/ideological_topic_scale",
                         ideological_topic_scale)
    ideological_topic_distribution = tfp.distributions.Normal(
        loc=ideological_topic_loc,
        scale=ideological_topic_scale)

    ideal_point_loc = tf.get_variable(
        "ideal_point_loc",
        shape=[num_authors],
        dtype=tf.float32)
    ideal_point_scale_logit = tf.get_variable(
        "ideal_point_scale_logit",
        initializer=tf.initializers.random_normal(mean=0, stddev=1.),
        shape=[num_authors],
        dtype=tf.float32)
    ideal_point_scale = tf.nn.softplus(ideal_point_scale_logit)
    ideal_point_distribution = tfp.distributions.Normal(
        loc=ideal_point_loc,
        scale=ideal_point_scale)
    tf.summary.histogram("params/ideal_point_loc",
                         tf.reshape(ideal_point_loc, [-1]))
    tf.summary.histogram("params/ideal_point_scale",
                         tf.reshape(ideal_point_scale, [-1]))
    return document_distribution, objective_topic_distribution, ideological_topic_distribution, ideal_point_distribution, document_loc, document_scale, objective_topic_loc, objective_topic_scale, ideological_topic_loc, ideological_topic_scale, ideal_point_loc, ideal_point_scale


def initial_doc_distribution(fit_dir, num_documents, random_state):
    pre_document_loc = np.load(
        os.path.join(fit_dir, "document_loc.npy")).astype(np.float32)
    print("document_loc shape", pre_document_loc.shape)
    print(np.mean(pre_document_loc), np.var(pre_document_loc))

    pre_document_scale = np.load(
        os.path.join(fit_dir, "document_scale.npy")).astype(np.float32)

    document_scale_mean = tf.constant(
        np.mean(pre_document_scale), dtype=tf.float32)
    document_scale_var = tf.constant(
        np.var(pre_document_scale), dtype=tf.float32)

    initial_document_loc = np.float32(
        np.exp(random_state.randn(num_documents, FLAGS.num_topics)))
    document_loc = tf.get_variable(
        "document_loc",
        initializer=tf.constant(np.log(initial_document_loc)))
    document_scale_logit = tf.get_variable(
        "document_scale_logit",
        shape=[num_documents, FLAGS.num_topics],
        #initializer=tf.initializers.random_normal(mean=document_scale_mean, stddev=document_scale_var),
        initializer=tf.initializers.random_normal(mean=np.mean(
            pre_document_scale), stddev=np.var(pre_document_scale)),
        dtype=tf.float32)
    document_scale = tf.nn.softplus(document_scale_logit)
    document_distribution = tfp.distributions.LogNormal(
        loc=document_loc,
        scale=document_scale)

    return document_distribution, document_loc, document_scale


def get_distribution(save_dir, time_slice, num_documents, random_state):

    fit_dir = os.path.join(save_dir, str(int(time_slice+1)), "params")

    document_distribution, document_loc, document_scale = initial_doc_distribution(
        fit_dir, num_documents, random_state)

    objective_topic_loc = np.load(
        os.path.join(fit_dir, "objective_topic_loc.npy")).astype(np.float32)
    objective_topic_loc = tf.get_variable(
        "objective_topic_loc",
        initializer=tf.constant(objective_topic_loc))

    objective_topic_scale = np.load(
        os.path.join(fit_dir, "objective_topic_scale.npy")).astype(np.float32)
    objective_topic_scale = tf.get_variable(
        "objective_topic_scale",
        initializer=tf.constant(objective_topic_scale))
    objective_topic_scale = tf.math.abs(objective_topic_scale)

    ideological_topic_loc = np.load(
        os.path.join(fit_dir, "ideological_topic_loc.npy")).astype(np.float32)
    ideological_topic_loc = tf.get_variable(
        "ideological_topic_loc",
        initializer=tf.constant(ideological_topic_loc))

    ideological_topic_scale = np.load(
        os.path.join(fit_dir, "ideological_topic_scale.npy")).astype(np.float32)
    ideological_topic_scale = tf.get_variable(
        "ideological_topic_scale",
        initializer=tf.constant(ideological_topic_scale))
    ideological_topic_scale = tf.math.abs(ideological_topic_scale)

    ideal_point_loc = np.load(
        os.path.join(fit_dir, "ideal_point_loc.npy")).astype(np.float32)
    ideal_point_loc = tf.get_variable(
        "ideal_point_loc",
        initializer=tf.constant(ideal_point_loc))

    ideal_point_scale = np.load(
        os.path.join(fit_dir, "ideal_point_scale.npy")).astype(np.float32)
    ideal_point_scale = tf.get_variable(
        "ideal_point_scale",
        initializer=tf.constant(ideal_point_scale))
    ideal_point_scale = tf.math.abs(ideal_point_scale)

    objective_topic_distribution = tfp.distributions.LogNormal(
        loc=objective_topic_loc,
        scale=objective_topic_scale)
    ideological_topic_distribution = tfp.distributions.Normal(
        loc=ideological_topic_loc,
        scale=ideological_topic_scale)
    ideal_point_distribution = tfp.distributions.Normal(
        loc=ideal_point_loc,
        scale=ideal_point_scale)

    return document_distribution, objective_topic_distribution, ideological_topic_distribution, ideal_point_distribution, document_loc, document_scale, objective_topic_loc, objective_topic_scale, ideological_topic_loc, ideological_topic_scale, ideal_point_loc, ideal_point_scale


def save_distribution(param_save_dir, document_loc_val, document_scale_val, objective_topic_loc_val,
                      objective_topic_scale_val, ideological_topic_loc_val, ideological_topic_scale_val,
                      ideal_point_loc_val, ideal_point_scale_val):
    np.save(os.path.join(param_save_dir, "document_loc"), document_loc_val)
    np.save(os.path.join(param_save_dir, "document_scale"), document_scale_val)
    np.save(os.path.join(param_save_dir, "objective_topic_loc"),
            objective_topic_loc_val)
    np.save(os.path.join(param_save_dir, "objective_topic_scale"),
            objective_topic_scale_val)
    np.save(os.path.join(param_save_dir, "ideological_topic_loc"),
            ideological_topic_loc_val)
    np.save(os.path.join(param_save_dir, "ideological_topic_scale"),
            ideological_topic_scale_val)
    np.save(os.path.join(param_save_dir, "ideal_point_loc"), ideal_point_loc_val)
    np.save(os.path.join(param_save_dir, "ideal_point_scale"),
            ideal_point_scale_val)


def np_save(save_dir, filename, save_data):
    if not tf.gfile.Exists(save_dir):
        tf.gfile.MakeDirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    np.save(save_path, save_data)


def main(argv):
    del argv
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))
    tf.debugging.set_log_device_placement(True)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    project_dir = os.path.abspath(os.path.dirname(__file__))
    source_dir = os.path.join(project_dir, "data/{}".format(FLAGS.data))
    time_dir = os.path.join(source_dir, "time")

    save_dir = os.path.join(source_dir, "btm-fits")

    if tf.gfile.Exists(save_dir):
        tf.logging.warn("Deleting old log directory at {}".format(save_dir))
        tf.gfile.DeleteRecursively(save_dir)
    tf.gfile.MakeDirs(save_dir)

    time_choice = [int(filename) for filename in tf.io.gfile.listdir(time_dir)]
    time_choice.sort(reverse=True)

    for time_slice in time_choice:
        print("time+++++++++++++++++++++", time_slice)

        tf.reset_default_graph()
        tf.set_random_seed(FLAGS.seed)
        random_state = np.random.RandomState(FLAGS.seed)

        (iterator, dev_iterator, test_iterator, author_weights, vocabulary, author_map,
         num_documents, num_words, num_authors) = build_input_pipeline(
            source_dir,
            FLAGS.batch_size,
            random_state,
            time_slice,
            FLAGS.dev_ratio,
            FLAGS.counts_transformation)

        test_check = tf.placeholder(tf.bool)
        dev_check = tf.placeholder(tf.bool)
        document_indices, counts, author_indices, score_indices = tf.cond(
            test_check, lambda: test_iterator.get_next(), lambda: iterator.get_next())
        document_indices, counts, author_indices, score_indices = tf.cond(
            dev_check, lambda: dev_iterator.get_next(), lambda: (document_indices, counts, author_indices, score_indices))

        document_distribution, objective_topic_distribution, ideological_topic_distribution, ideal_point_distribution, document_loc, document_scale, objective_topic_loc, objective_topic_scale, ideological_topic_loc, ideological_topic_scale, ideal_point_loc, ideal_point_scale = initial_distribution(
            time_slice, source_dir, num_documents, num_words, num_authors, random_state)
        num_documents = tf.cast(num_documents, tf.float32)
        elbo, score_loss = get_elbo(counts,
                                    document_indices,
                                    author_indices,
                                    author_weights,
                                    document_distribution,
                                    objective_topic_distribution,
                                    ideological_topic_distribution,
                                    ideal_point_distribution,
                                    num_documents,
                                    FLAGS.batch_size,
                                    score_indices,
                                    time_slice,
                                    num_samples=FLAGS.num_samples)

        optim = tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        train_op = optim.minimize(-elbo)
        optim_class = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        train_class = optim_class.minimize(score_loss)

        init = tf.global_variables_initializer()

        with tf.Session(config=config) as sess:
            param_save_dir = os.path.join(
                save_dir, str(time_slice), "params")
            acc_save_dir = os.path.join(
                save_dir, str(time_slice), "loss")
            sess.run(init)
            sess.run(tf.local_variables_initializer())
            start_time = time.time()

            if time_slice < max(time_choice):
                for step in range(3):
                    (elbo_val, score_loss_val, ) = sess.run(
                        [elbo, score_loss, ], feed_dict={test_check: True, dev_check: False})
                    duration = (time.time() - start_time) / (step + 1)
                    print("Test Step: {:>3d} ELBO: {:.3f} ({:.3f} sec), label loss {:.3f}".format(
                        step, elbo_val, duration, score_loss_val))

            # Training on ELBO first
            elbo_val_laststep = 1e20
            for step in range(FLAGS.max_steps):
                (_, ) = sess.run(
                    [train_op, ], feed_dict={test_check: False, dev_check: False})

                duration = (time.time() - start_time) / (step + 1)
                if step % FLAGS.print_steps == 0:
                    (elbo_val, score_loss_val, ) = sess.run(
                        [elbo, score_loss, ], feed_dict={test_check: False, dev_check: False})
                    print("Step: {:>3d} ELBO: {:.3f} ({:.3f} sec), label loss {:.3f}".format(
                        step, elbo_val, duration, score_loss_val))
                    if -elbo_val >= elbo_val_laststep:
                        print("early stop since ELBO reaching limit")
                        break

                if step % FLAGS.check_steps == 0 or step == FLAGS.max_steps - 1:
                    if not tf.gfile.Exists(param_save_dir):
                        tf.gfile.MakeDirs(param_save_dir)
                    if not tf.gfile.Exists(acc_save_dir):
                        tf.gfile.MakeDirs(acc_save_dir)

                    (document_loc_val, document_scale_val, objective_topic_loc_val,
                     objective_topic_scale_val, ideological_topic_loc_val,
                     ideological_topic_scale_val, ideal_point_loc_val,
                     ideal_point_scale_val, elbo_val, score_loss_val, ) = sess.run([
                         document_loc, document_scale, objective_topic_loc,
                         objective_topic_scale, ideological_topic_loc,
                         ideological_topic_scale, ideal_point_loc, ideal_point_scale, elbo, score_loss, ], feed_dict={test_check: False, dev_check: False})

                    save_distribution(param_save_dir, document_loc_val, document_scale_val, objective_topic_loc_val,
                                      objective_topic_scale_val, ideological_topic_loc_val, ideological_topic_scale_val,
                                      ideal_point_loc_val, ideal_point_scale_val)

                if step % FLAGS.check_steps == 0:
                    (elbo_val, score_loss_val, ) = sess.run(
                        [elbo, score_loss, ], feed_dict={test_check: False, dev_check: True})
                    print("===================================================")
                    print("Dev Step: {:>3d} ELBO: {:.3f} ({:.3f} sec), label loss {:.3f}".format(
                        step, elbo_val, duration, score_loss_val))

                    middle_save_dir = os.path.join(
                        save_dir, str(time_slice), "middle", str(step))
                    if not tf.gfile.Exists(middle_save_dir):
                        tf.gfile.MakeDirs(middle_save_dir)

                    save_distribution(middle_save_dir, document_loc_val, document_scale_val, objective_topic_loc_val,
                                      objective_topic_scale_val, ideological_topic_loc_val, ideological_topic_scale_val,
                                      ideal_point_loc_val, ideal_point_scale_val)

            # Adding classifier loss and adversarial loss to training process
            for step in range(FLAGS.add_steps):
                (_, ) = sess.run(
                    [train_class, ], feed_dict={test_check: False, dev_check: False})
                (_, ) = sess.run(
                    [train_op, ], feed_dict={test_check: False, dev_check: False})
                duration = (time.time() - start_time) / (step + 1)
                if step % FLAGS.print_steps == 0:
                    (elbo_val, score_loss_val, ) = sess.run(
                        [elbo, score_loss, ], feed_dict={test_check: False, dev_check: False})
                    print("Step: {:>3d} ELBO: {:.3f} ({:.3f} sec), label loss {:.3f}".format(
                        step, elbo_val, duration, score_loss_val))


if __name__ == "__main__":
    tf.app.run()
