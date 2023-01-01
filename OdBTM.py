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
from scipy.stats import spearmanr
#from scipy.stats import wasserstein_distance
import tensorflow as tf
import json

import tensorflow_probability as tfp
from collections import Counter


flags.DEFINE_float("learning_rate",
                   default=0.01,
                   help="Adam learning rate.")
flags.DEFINE_integer("max_steps",
                     default=100000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("add_steps",
                     default=20000,
                     help="Number of fine tuning steps to run based on classification loss.")
flags.DEFINE_integer("print_steps",
                     default=2500,
                     help="Number of steps to print and save results.")
flags.DEFINE_integer("check_steps",
                     default=5000,
                     help="Number of steps to check whether stop training and save parameters.")
flags.DEFINE_integer("num_topics",
                     default=30,
                     help="Number of topics.")
flags.DEFINE_integer("batch_size",
                     default=128,
                     help="Batch size.")
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
flags.DEFINE_integer("seed",
                     default=2,
                     help="Random seed to be used.")
flags.DEFINE_float("dev_ratio",
                   default=0.1,
                   help="Split rate of data to dev and train")

FLAGS = flags.FLAGS


def build_database(random_state, num_documents, counts_transformation, counts, brand_indices, score_indices, batch_size, dev_ratio, dev=True, balance=True):
    # Shuffle data.
    print(counts.dtype)
    documents = random_state.permutation(num_documents)
    shuffled_brand_indices = brand_indices[documents]
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
        (documents, shuffled_counts, shuffled_brand_indices, shuffled_score_indices))

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
        ds_pos = dataset.filter(lambda documents, shuffled_counts, shuffled_brand_indices,
                                shuffled_score_indices: tf.reshape(tf.equal(shuffled_score_indices, 2), []))
        ds_neu = dataset.filter(lambda documents, shuffled_counts, shuffled_brand_indices,
                                shuffled_score_indices: tf.reshape(tf.equal(shuffled_score_indices, 1), []))
        ds_neg = dataset.filter(lambda documents, shuffled_counts, shuffled_brand_indices,
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
    brand_indices = np.load(
        os.path.join(data_dir, "brand_indices.npy")).astype(np.int32)
    score_indices = np.load(
        os.path.join(data_dir, "score_indices.npy")).astype(np.int32)
    # timestamps = np.load(
    #    os.path.join(data_dir, "timestamps.npy")).astype(np.int32)
    num_brands = np.max(brand_indices + 1)

    return num_documents, num_words, counts, brand_indices, score_indices, num_brands


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
    return brand_score_real


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
        files inside the rep: `counts.npz`, `brand_indices.npy`,
        `brand_map.txt`, and `vocabulary.txt`.
      batch_size: The batch size to use for training.
      random_state: A NumPy `RandomState` object, used to shuffle the data.
      counts_transformation: A string indicating how to transform the counts.
        One of "nothing", "binary", "log", or "sqrt".
    """
    data_dir = os.path.join(source_dir, "clean")
    if test:
        train_dir = os.path.join(source_dir, "time", str(time_slice))
        num_documents, num_words, counts, brand_indices, score_indices, num_brands = load_data(
            train_dir)
        iterator, dev_iterator = build_database(random_state, num_documents, counts_transformation,
                                                counts, brand_indices, score_indices, batch_size, dev_ratio)
        test_iterator, _ = build_database(random_state, num_documents, counts_transformation,
                                          counts, brand_indices, score_indices, batch_size, dev_ratio, dev=False, balance=False)
    else:
        num_documents, num_words, counts, brand_indices, score_indices, num_brands = load_data(
            data_dir)
        iterator, dev_iterator = build_database(random_state, num_documents, counts_transformation,
                                                counts, brand_indices, score_indices, batch_size, dev_ratio)
        test_iterator = 0
    brand_map = np.loadtxt(os.path.join(data_dir, "brand_map.txt"),
                            dtype=str,
                            delimiter="\n")
    vocabulary = np.loadtxt(os.path.join(data_dir, "vocabulary.txt"),
                            dtype=str,
                            delimiter="\n",
                            comments="<!-")
    brand_score_real = get_brand_score_real(score_indices, brand_indices, brand_map)


    total_counts_per_brand = np.bincount(
        brand_indices,
        weights=np.array(np.sum(counts, axis=1)).flatten())
    counts_per_document_per_brand = (
        total_counts_per_brand / np.bincount(brand_indices))
    # brand weights is how much lengthy each brand's opinion over average is.
    brand_weights = (counts_per_document_per_brand /
                      np.mean(np.sum(counts, axis=1))).astype(np.float32)
    return (iterator, dev_iterator, test_iterator, brand_weights, vocabulary, brand_map,
            num_documents, num_words, num_brands, brand_score_real)


def check_inf_nan(input_tensor):
    input_tensor[np.isnan(input_tensor)] = 0.
    input_tensor[np.isinf(input_tensor)] = 1.
    return input_tensor


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
    selected_prediction = tf.keras.layers.Dense(500)(
        tf.reshape(count_samples, [batch_size, num_words]))
    selected_prediction = tf.keras.layers.Dense(
        num_labels)(selected_prediction)

    return count_samples, selected_prediction


class Wasserstein(object):
    """Class to hold (ref to) data and compute Wasserstein distance."""

    def __init__(self, source_gen, target_gen, batch_size, basedist=None):
        """Inits Wasserstein with source and target data."""
        self.bs = batch_size
        self.source_gen = source_gen
        self.target_gen = target_gen
        self.gradbs = batch_size  # number of source sample to compute gradient
        if basedist is None:
            basedist = self.l2dist
        self.basedist = basedist

    def add_summary_montage(self, images, name, num=9):
        vis_images = tf.split(images[:num], num_or_size_splits=num, axis=0)
        vis_images = tf.concat(vis_images, axis=2)
        tf.summary.image(name, vis_images)
        return vis_images

    def add_summary_images(self, num=9):
        """Visualize source images and nearest neighbors from target."""
        source_ims = self.source_gen.get_batch(bs=num, reuse=True)
        vis_images = self.add_summary_montage(source_ims, 'source_ims', num)

        target_ims = self.target_gen.get_batch()
        _ = self.add_summary_montage(target_ims, 'target_ims', num)

        c_xy = self.basedist(source_ims, target_ims)  # pairwise cost
        idx = tf.argmin(c_xy, axis=1)  # find nearest neighbors
        matches = tf.gather(target_ims, idx)
        vis_matches = self.add_summary_montage(matches, 'neighbors_ims', num)

        vis_both = tf.concat([vis_images, vis_matches], axis=1)
        tf.summary.image('matches_ims', vis_both)

        return

    def l2dist(self, source, target):
        """Computes pairwise Euclidean distances in tensorflow."""
        def flatten_batch(x):
            dim = tf.reduce_prod(tf.shape(x)[1:])
            return tf.reshape(x, [-1, dim])

        def scale_batch(x):
            dim = tf.reduce_prod(tf.shape(x)[1:])
            return x/tf.sqrt(tf.cast(dim, tf.float32))

        def prepare_batch(x):
            return scale_batch(flatten_batch(x))

        target_flat = prepare_batch(target)  # shape: [bs, nt]
        target_sqnorms = tf.reduce_sum(
            tf.square(target_flat), axis=1, keep_dims=True)
        target_sqnorms_t = tf.transpose(target_sqnorms)

        source_flat = prepare_batch(source)  # shape: [bs, ns]
        source_sqnorms = tf.reduce_sum(
            tf.square(source_flat), axis=1, keep_dims=True)

        dotprod = tf.matmul(source_flat, target_flat,
                            transpose_b=True)  # [ns, nt]
        sqdist = source_sqnorms - 2*dotprod + target_sqnorms_t
        # potential tiny negatives are suppressed
        dist = tf.sqrt(tf.nn.relu(sqdist))
        return dist  # shape: [ns, nt]

    def grad_hbar(self, v, gradbs, reuse=True):
        """Compute gradient of hbar function for Wasserstein iteration."""

        #source_ims = self.source_gen.get_batch(bs=gradbs, reuse=reuse)
        source_ims = self.source_gen[:self.bs]
        #target_data = self.target_gen.get_batch()
        target_data = self.target_gen[:self.bs]

        c_xy = self.basedist(source_ims, target_data)
        c_xy -= v  # [gradbs, trnsize]
        # [1] (index of subgradient)
        idx = tf.argmin(c_xy, axis=1)
        target_bs = self.bs
        xi_ij = tf.one_hot(idx, target_bs)  # find matches, [gradbs, trnsize]
        xi_ij = tf.reduce_mean(xi_ij, axis=0, keep_dims=True)    # [1, trnsize]
        grad = 1./target_bs - xi_ij  # output: [1, trnsize]
        return grad

    def hbar(self, v, reuse=True):
        """Compute value of hbar function for Wasserstein iteration."""
        #source_ims = self.source_gen.get_batch(bs=gradbs, reuse=reuse)
        source_ims = self.source_gen[:self.bs]
        #target_data = self.target_gen.get_batch()
        target_data = self.target_gen[:self.bs]

        c_xy = self.basedist(source_ims, target_data)
        c_avg = tf.reduce_mean(c_xy)
        c_xy -= c_avg
        c_xy -= v

        c_xy_min = tf.reduce_min(c_xy, axis=1)  # min_y[ c(x, y) - v(y) ]
        c_xy_min = tf.reduce_mean(c_xy_min)     # expectation wrt x
        return tf.reduce_mean(v, axis=1) + c_xy_min + c_avg  # avg wrt y

    def k_step(self, k, v, vt, c, reuse=True):
        """Perform one update step of Wasserstein computation."""
        grad_h = self.grad_hbar(vt, gradbs=self.gradbs, reuse=reuse)
        vt = tf.assign_add(vt, c/tf.sqrt(k)*grad_h, name='vt_assign_add')
        v = ((k-1.)*v + vt)/k
        return k+1, v, vt, c

    def dist(self, C=.1, nsteps=10, reset=False):
        """Compute Wasserstein distance (Alg.2 in [Genevay etal, NIPS'16])."""
        target_bs = self.bs
        vtilde = tf.Variable(tf.zeros([1, target_bs]), name='vtilde')
        v = tf.Variable(tf.zeros([1, target_bs]), name='v')
        k = tf.Variable(1., name='k')

        k = k.assign(1.)  # restart averaging from 1 in each call
        if reset:  # used for randomly sampled target data, otherwise warmstart
            # reset every time graph is evaluated
            v = v.assign(tf.zeros([1, target_bs]))
            vtilde = vtilde.assign(tf.zeros([1, target_bs]))

        # (unrolled) optimization loop. first iteration, create variables
        k, v, vtilde, C = self.k_step(k, v, vtilde, C, reuse=False)
        # (unrolled) optimization loop. other iterations, reuse variables
        k, v, vtilde, C = tf.while_loop(cond=lambda k, *_: k < nsteps,
                                        body=self.k_step,
                                        loop_vars=[k, v, vtilde, C])
        v = tf.stop_gradient(v)  # only transmit gradient through cost
        val = self.hbar(v)
        return tf.reduce_mean(val)


def get_elbo(counts,
             document_indices,
             brand_indices,
             brand_weights,
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
             num_samples=1,
             add_train="wasserstein_distance"):
    """Approximate variational Lognormal ELBO using reparameterization.

    Args:
      counts: A matrix with shape `[batch_size, num_words]`.
      document_indices: An int-vector with shape `[batch_size]`.
      brand_indices: An int-vector with shape `[batch_size]`.
      brand_weights: A vector with shape `[num_brands]`, constituting how
        lengthy the opinion is above average.
      document_distribution: A positive `Distribution` object with parameter
        shape `[num_documents, num_topics]`.
      objective_topic_distribution: A positive `Distribution` object with
        parameter shape `[num_topics, num_words]`.
      ideological_topic_distribution: A positive `Distribution` object with
        parameter shape `[num_topics, num_words]`.
      ideal_point_distribution: A `Distribution` object over [0, 1] with
        parameter_shape `[num_brands]`.
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
    print("+++++++++++++++++++++++++ num words in elbo", num_words)

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
                                      brand_indices,
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

# Normalize by how lengthy the brand's opinion is.
    selected_brand_weights = tf.gather(brand_weights, brand_indices)
    selected_ideological_topic_samples = (
        selected_brand_weights[tf.newaxis, :, tf.newaxis, tf.newaxis] *
        selected_ideological_topic_samples)
    reversed_ideological_topic_samples = (
        selected_brand_weights[tf.newaxis, :, tf.newaxis, tf.newaxis] *
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
    selected_count_samples, selected_prediction = example_from_rate(
        rate, class_limit, class_list, outcome_vector, k_factorial, batch_size, num_words, num_labels)
    # This is for elbo calculation later
    poisson_count_distribution = tfp.distributions.Poisson(rate=rate)

    # rate based on reversed ideological topic samples
    reversed_rate = tf.reduce_sum(
        selected_document_samples[:, :, :, tf.newaxis] *
        objective_topic_samples[:, tf.newaxis, :, :] *
        reversed_ideological_topic_samples[:, :, :, :],
        axis=2)
    reversed_count_samples, reversed_prediction = example_from_rate(
        reversed_rate, class_limit, class_list, outcome_vector, k_factorial, batch_size, num_words, num_labels)
    # model = tf.keras.Sequential()

    score_indices = tf.convert_to_tensor(score_indices)
    given_labels = tf.cast(tf.one_hot(
        score_indices, num_labels), dtype=tf.float32)
    reversed_score = (num_labels - 1) - score_indices
    reversed_labels = tf.cast(tf.one_hot(
        reversed_score, num_labels), dtype=tf.float32)
    mse = tf.keras.losses.MeanSquaredError()

    if add_train == "classification": 
        loss_real = tf.nn.softmax_cross_entropy_with_logits(
            labels=given_labels, logits=selected_prediction)
        loss_reversed = tf.nn.softmax_cross_entropy_with_logits(
            labels=reversed_labels, logits=reversed_prediction)

    elif add_train == "regression":
        loss_real = tf.math.sqrt(mse(given_labels, selected_prediction))
        loss_reversed = tf.math.sqrt(mse(reversed_labels, reversed_prediction))

    elif add_train == "wasserstein_distance":
        loss_real = tf.maximum(
            0., 1-wasserstein_distance(given_labels, selected_prediction, FLAGS.batch_size))
        loss_reversed = tf.maximum(
            0., 1-wasserstein_distance(reversed_labels, reversed_prediction, FLAGS.batch_size))

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
    # print("elbo", elbo, "log_prior", log_prior, "count_log_likelihood",
    #      count_log_likelihood, "entropy", entropy)
    elbo = tf.reduce_mean(elbo)

    tf.summary.scalar("elbo/elbo", elbo)
    tf.summary.scalar("elbo/log_prior", tf.reduce_mean(log_prior))
    tf.summary.scalar("elbo/count_log_likelihood",
                      tf.reduce_mean(count_log_likelihood))
    tf.summary.scalar("elbo/entropy", tf.reduce_mean(entropy))
    return elbo, score_loss, selected_prediction, reversed_prediction


def wasserstein_distance(given_labels, selected_prediction, batch_size):
    wd = Wasserstein(given_labels, selected_prediction, batch_size)
    return wd.dist()


def initial_distribution(pre_initialize_parameters, source_dir, time_slice, num_documents, num_words, num_brands, random_state):
    if pre_initialize_parameters:
        fit_dir = os.path.join(source_dir, "pf-fits", str(time_slice))
        fitted_document_shape = check_inf_nan(np.load(
            os.path.join(fit_dir, "document_shape.npy")).astype(np.float32))
        fitted_document_rate = check_inf_nan(np.load(
            os.path.join(fit_dir, "document_rate.npy")).astype(np.float32))
        fitted_topic_shape = check_inf_nan(np.load(
            os.path.join(fit_dir, "topic_shape.npy")).astype(np.float32))
        fitted_topic_rate = check_inf_nan(np.load(
            os.path.join(fit_dir, "topic_rate.npy")).astype(np.float32))

        initial_document_loc = fitted_document_shape / fitted_document_rate
        initial_objective_topic_loc = fitted_topic_shape / fitted_topic_rate
    else:
        initial_document_loc = np.float32(
            np.exp(random_state.randn(num_documents, FLAGS.num_topics)))
        initial_objective_topic_loc = np.float32(
            np.exp(random_state.randn(FLAGS.num_topics, num_words)))

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
        shape=[num_brands],
        dtype=tf.float32)
    ideal_point_scale_logit = tf.get_variable(
        "ideal_point_scale_logit",
        initializer=tf.initializers.random_normal(mean=0, stddev=1.),
        shape=[num_brands],
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


def initial_doc_distribution(fit_dir, num_documents, random_state, pre_document_scale):
    fitted_document_shape = check_inf_nan(np.load(
        os.path.join(fit_dir, "document_shape.npy")).astype(np.float32))
    fitted_document_rate = check_inf_nan(np.load(
        os.path.join(fit_dir, "document_rate.npy")).astype(np.float32))
    initial_document_loc = np.float32(
        np.exp(random_state.randn(num_documents, FLAGS.num_topics)))
    initial_document_loc = fitted_document_shape / fitted_document_rate

    
    document_loc = tf.get_variable(
        "document_loc",
        initializer=tf.constant(np.log(initial_document_loc)))

    '''document_scale_mean = tf.constant(
        np.mean(pre_document_scale), dtype=tf.float32)
    document_scale_var = tf.constant(
        np.var(pre_document_scale), dtype=tf.float32)'''
    document_scale_logit = tf.get_variable(
        "document_scale_logit",
        shape=[num_documents, FLAGS.num_topics],
        #initializer=tf.initializers.random_normal(mean=document_scale_mean, stddev=document_scale_var),
        initializer=tf.initializers.random_normal(mean=0, stddev=1.),
        dtype=tf.float32)
    document_scale = tf.nn.softplus(document_scale_logit)

    document_distribution = tfp.distributions.LogNormal(
        loc=document_loc,
        scale=document_scale)

    return document_distribution, document_loc, document_scale


def get_distribution(source_dir, time_slice, num_documents, num_words, random_state, rollback = 0.1):
    doc_dir = os.path.join(source_dir, "pf-fits", str(time_slice))
    fit_dir = os.path.join(source_dir, "obtm-fits",
                           str(int(time_slice-1)), "params")

    pre_document_scale = check_inf_nan(np.load(
        os.path.join(fit_dir, "document_scale.npy")).astype(np.float32))
    document_distribution, document_loc, document_scale = initial_doc_distribution(
        doc_dir, num_documents, random_state, pre_document_scale)

    # Load pre-training topic_loc
    fitted_topic_shape = check_inf_nan(np.load(
        os.path.join(doc_dir, "topic_shape.npy")).astype(np.float32))
    fitted_topic_rate = check_inf_nan(np.load(
        os.path.join(doc_dir, "topic_rate.npy")).astype(np.float32))
    initial_objective_topic_loc = fitted_topic_shape / fitted_topic_rate

    # Load inherited topic_loc at t-1
    objective_topic_loc = check_inf_nan(np.load(
        os.path.join(fit_dir, "objective_topic_loc.npy")).astype(np.float32))
    
    # Now we reweight initial topic_loc at t 
    # by averaging pre-training topic_loc at t and inherited topic_loc at t-1
    objective_topic_loc = rollback*initial_objective_topic_loc + (1-rollback)*objective_topic_loc

    objective_topic_loc = tf.get_variable(
        "objective_topic_loc",
        initializer=tf.constant(objective_topic_loc))

    objective_topic_scale = check_inf_nan(np.load(
        os.path.join(fit_dir, "objective_topic_scale.npy")).astype(np.float32))
    objective_topic_scale = tf.get_variable(
        "objective_topic_scale",
        initializer=tf.constant(objective_topic_scale))
    objective_topic_scale = tf.math.abs(objective_topic_scale)

    ideological_topic_loc = check_inf_nan(np.load(
        os.path.join(fit_dir, "ideological_topic_loc.npy")).astype(np.float32))
    ideological_topic_loc = tf.get_variable(
        "ideological_topic_loc",
        initializer=tf.constant(ideological_topic_loc))

    ideological_topic_scale = check_inf_nan(np.load(
        os.path.join(fit_dir, "ideological_topic_scale.npy")).astype(np.float32))
    ideological_topic_scale = tf.get_variable(
        "ideological_topic_scale",
        initializer=tf.constant(ideological_topic_scale))
    ideological_topic_scale = tf.math.abs(ideological_topic_scale)

    ideal_point_loc = check_inf_nan(np.load(
        os.path.join(fit_dir, "ideal_point_loc.npy")).astype(np.float32))
    ideal_point_loc = tf.get_variable(
        "ideal_point_loc",
        initializer=tf.constant(ideal_point_loc))

    ideal_point_scale = check_inf_nan(np.load(
        os.path.join(fit_dir, "ideal_point_scale.npy")).astype(np.float32))
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

    # As described in the docstring, the data directory must have the following
    # files: counts.npz, brand_indices.npy, vocabulary.txt, brand_map.txt.
    save_dir = os.path.join(source_dir, "obtm-fits")

    if tf.gfile.Exists(save_dir):
        tf.logging.warn("Deleting old log directory at {}".format(save_dir))
        tf.gfile.DeleteRecursively(save_dir)
    tf.gfile.MakeDirs(save_dir)

    time_choice = [int(filename) for filename in tf.io.gfile.listdir(time_dir)]
    time_choice.sort(reverse=False)
    ranking_p_value = 1.
    learning_coefficient = 0.1
    train_steps_recording = {}
    rating_real = None
    rating_generated_keep = None
    ranking_correlation_previous = 0.


    for time_slice in time_choice:
        print("time+++++++++++++++++++++", time_slice)
        rating_real_last = rating_generated_keep
        
        tf.reset_default_graph()
        tf.set_random_seed(FLAGS.seed)
        random_state = np.random.RandomState(FLAGS.seed)

        (iterator, dev_iterator, test_iterator, brand_weights, vocabulary, brand_map,
         num_documents, num_words, num_brands, brand_score_real) = build_input_pipeline(
            source_dir,
            FLAGS.batch_size,
            random_state,
            time_slice,
            FLAGS.dev_ratio,
            FLAGS.counts_transformation)

        test_check = tf.placeholder(tf.bool)
        dev_check = tf.placeholder(tf.bool)
        document_indices, counts, brand_indices, score_indices = iterator.get_next()
        
        current_max_steps = FLAGS.max_steps
        
        initial_time = min(time_choice)
        if time_slice == initial_time:
            document_distribution, objective_topic_distribution, ideological_topic_distribution, ideal_point_distribution, document_loc, document_scale, objective_topic_loc, objective_topic_scale, ideological_topic_loc, ideological_topic_scale, ideal_point_loc, ideal_point_scale = initial_distribution(
                FLAGS.pre_initialize_parameters, source_dir, time_slice, num_documents, num_words, num_brands, random_state)
            current_add_steps = FLAGS.add_steps
            current_min_steps = FLAGS.max_steps
            min_add_steps = FLAGS.add_steps
            elbo_check_steps = FLAGS.check_steps
            label_check_steps = FLAGS.check_steps
        elif time_slice > initial_time:
            document_distribution, objective_topic_distribution, ideological_topic_distribution, ideal_point_distribution, document_loc, document_scale, objective_topic_loc, objective_topic_scale, ideological_topic_loc, ideological_topic_scale, ideal_point_loc, ideal_point_scale = get_distribution(
                source_dir, time_slice, num_documents, num_words, random_state, rollback=ranking_p_value_current*learning_coefficient)
            current_add_steps = 0
            current_min_steps = int(FLAGS.max_steps*(ranking_p_value+0.1))
            min_add_steps = int(FLAGS.add_steps*(ranking_p_value))
            elbo_check_steps = int(current_max_steps/20)
            label_check_steps = int(current_add_steps/20)

        num_documents = tf.cast(num_documents, tf.float32)

        elbo, score_loss, selected_prediction, reversed_prediction = get_elbo(counts,
                                                                                                                                                                                                                                                                   document_indices,
                                                                                                                                                                                                                                                                   brand_indices,
                                                                                                                                                                                                                                                                   brand_weights,
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
            test_save_dir = os.path.join(
                save_dir, str(time_slice), "test")
            acc_save_dir = os.path.join(
                save_dir, str(time_slice), "loss")
            debug_save_dir = os.path.join(
                save_dir, "debug", str(time_slice))
            sess.run(init)
            sess.run(tf.local_variables_initializer())
            start_time = time.time()
            
            for step in range(current_max_steps):
                (_, ) = sess.run(
                    [train_op, ], feed_dict={test_check: False, dev_check: False})

                duration = (time.time() - start_time) / (step + 1)
                if step % FLAGS.print_steps == 0:
                    if not tf.gfile.Exists(debug_save_dir):
                        tf.gfile.MakeDirs(debug_save_dir)

                    (elbo_val, score_loss_val, ) = sess.run(
                        [elbo, score_loss, ], feed_dict={test_check: False, dev_check: False})
                    print("Train Step: {:>3d} ELBO: {:.3f} ({:.3f} sec), label loss {:.3f}".format(
                        step, elbo_val, duration, score_loss_val))
                
                if step % elbo_check_steps == 0 or step == current_max_steps - 1 or step == current_min_steps:
                    if not tf.gfile.Exists(param_save_dir):
                        tf.gfile.MakeDirs(param_save_dir)
                    if not tf.gfile.Exists(test_save_dir):
                        tf.gfile.MakeDirs(test_save_dir)
                    if not tf.gfile.Exists(acc_save_dir):
                        tf.gfile.MakeDirs(acc_save_dir)

                    (document_loc_val, document_scale_val, objective_topic_loc_val,
                     objective_topic_scale_val, ideological_topic_loc_val,
                     ideological_topic_scale_val, ideal_point_loc_val,
                     ideal_point_scale_val, elbo_val, score_loss_val, ) = sess.run([
                         document_loc, document_scale, objective_topic_loc,
                         objective_topic_scale, ideological_topic_loc,
                         ideological_topic_scale, ideal_point_loc, ideal_point_scale, elbo, score_loss, ], feed_dict={test_check: False, dev_check: False})
                    
                    # Now check the spearman
                    # Print ideal point orderings.
                    brand_score_generated = {}
                    for index in range(len(brand_map)): brand_score_generated[brand_map[index]] = ideal_point_loc_val[index]
                    rating_real = []
                    rating_generated = []
                    for brand_i in brand_score_real:
                        rating_real.append(brand_score_real[brand_i])
                        rating_generated.append(brand_score_generated[brand_i])
                    if time_slice < initial_time:
                        rating_real = rating_real_last
                    ranking_correlation, ranking_p_value_real = spearmanr(rating_real, rating_generated)
                    z_mean = (tf.math.log((1+abs(ranking_correlation))/(1-abs(ranking_correlation))))*0.5
                    z_var = tf.constant(1/(num_brands-3))
                    (z_mean_val, z_var_val) = sess.run([z_mean, z_var, ], feed_dict={test_check: False, dev_check: True})
                    
                    z_dist = tfp.distributions.Normal(loc=z_mean, scale=z_var)
                    ranking_p_value_current = z_dist.cdf(abs(ranking_correlation_previous))
                    (ranking_p_value_current_val,) = sess.run([ranking_p_value_current,], feed_dict={test_check: False, dev_check: True})
                    ranking_p_value_current = max(0.05, ranking_p_value_current_val)
                    print("distribution", z_mean_val, z_var_val, ranking_correlation_previous, num_brands)
                    print("cor", ranking_correlation, ranking_p_value_real, ranking_p_value_current, ranking_p_value_current_val)
                    if step == current_min_steps:
                        save_distribution(test_save_dir, document_loc_val, document_scale_val, objective_topic_loc_val,
                                      objective_topic_scale_val, ideological_topic_loc_val, ideological_topic_scale_val,
                                      ideal_point_loc_val, ideal_point_scale_val)
                    if ranking_p_value_current < ranking_p_value:
                        rating_generated_keep = rating_generated[:]
                        save_distribution(param_save_dir, document_loc_val, document_scale_val, objective_topic_loc_val,
                                      objective_topic_scale_val, ideological_topic_loc_val, ideological_topic_scale_val,
                                      ideal_point_loc_val, ideal_point_scale_val)
                        ranking_p_value = ranking_p_value_current
                        ranking_correlation_previous = ranking_correlation
                    else:
                        if step < current_min_steps:
                            rating_generated_keep = rating_generated[:]
                            save_distribution(param_save_dir, document_loc_val, document_scale_val, objective_topic_loc_val,
                                      objective_topic_scale_val, ideological_topic_loc_val, ideological_topic_scale_val,
                                      ideal_point_loc_val, ideal_point_scale_val)
                        else:
                            print(" Early Stop at Train Step: {:>3d} time_slice: {:.1f}, Ranking Result: {:.3f} (p-value: {:.3f})".format(
                                step, time_slice, ranking_correlation, ranking_p_value_current))
                            if time_slice in train_steps_recording:
                                train_steps_recording[time_slice] += [step, "Training stage, Ranking Result: {:.3f} (p-value: {:.3f})".format(ranking_correlation, ranking_p_value_real)]
                            else:
                                train_steps_recording[time_slice] = [step, "Training stage, Ranking Result: {:.3f} (p-value: {:.3f})".format(ranking_correlation, ranking_p_value_real)]
                            break

                if step % elbo_check_steps == 0:
                    (elbo_val, score_loss_val, ) = sess.run(
                        [elbo, score_loss, ], feed_dict={test_check: False, dev_check: True})
                    print("===================================================")
                    print("Dev Step: {:>3d} ELBO: {:.3f} ({:.3f} sec), label loss {:.3f}".format(
                        step, elbo_val, duration, score_loss_val))

            # Adding classifier loss and adversarial loss to training process
            for step in range(current_add_steps):
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

                if step % label_check_steps == 0 or step == current_max_steps - 1:
                    if not tf.gfile.Exists(param_save_dir):
                        tf.gfile.MakeDirs(param_save_dir)
                    if not tf.gfile.Exists(acc_save_dir):
                        tf.gfile.MakeDirs(acc_save_dir)

                    (document_loc_val, document_scale_val, objective_topic_loc_val,
                     objective_topic_scale_val, ideological_topic_loc_val,
                     ideological_topic_scale_val, ideal_point_loc_val,
                     ideal_point_scale_val, elbo_val, score_loss_val,) = sess.run([
                         document_loc, document_scale, objective_topic_loc,
                         objective_topic_scale, ideological_topic_loc,
                         ideological_topic_scale, ideal_point_loc, ideal_point_scale, elbo, score_loss,], feed_dict={test_check: False, dev_check: False})

                    # Now check the spearman
                    # Print ideal point orderings.
                    brand_score_generated = {}
                    for index in range(len(brand_map)): brand_score_generated[brand_map[index]] = ideal_point_loc_val[index]
                    rating_real = []
                    rating_generated = []
                    for brand_i in brand_score_real:
                        rating_real.append(brand_score_real[brand_i])
                        rating_generated.append(brand_score_generated[brand_i])
                    ranking_correlation, ranking_p_value_real = spearmanr(rating_real, rating_generated)

                    z_mean = (tf.math.log((1+abs(ranking_correlation))/(1-abs(ranking_correlation))))*0.5
                    z_var = tf.constant(1/(num_brands-3))
                    (z_mean_val, z_var_val) = sess.run([z_mean, z_var, ], feed_dict={test_check: False, dev_check: True})
                    
                    z_dist = tfp.distributions.Normal(loc=z_mean, scale=z_var)
                    ranking_p_value_current = z_dist.cdf(abs(ranking_correlation_previous))
                    (ranking_p_value_current_val,) = sess.run([ranking_p_value_current,], feed_dict={test_check: False, dev_check: True})
                    ranking_p_value_current = max(0.05, ranking_p_value_current_val)
                    print("cor", ranking_correlation, ranking_p_value_real, ranking_p_value_current, ranking_p_value_current_val)

                    if ranking_p_value_current < ranking_p_value:
                        rating_generated_keep = rating_generated[:]
                        save_distribution(param_save_dir, document_loc_val, document_scale_val, objective_topic_loc_val,
                                      objective_topic_scale_val, ideological_topic_loc_val, ideological_topic_scale_val,
                                      ideal_point_loc_val, ideal_point_scale_val)
                        ranking_p_value = ranking_p_value_current
                        ranking_correlation_previous = ranking_correlation
                    else:
                        if step < min_add_steps:
                            rating_generated_keep = rating_generated[:]
                            save_distribution(param_save_dir, document_loc_val, document_scale_val, objective_topic_loc_val,
                                      objective_topic_scale_val, ideological_topic_loc_val, ideological_topic_scale_val,
                                      ideal_point_loc_val, ideal_point_scale_val)
                        else:
                            print("Early Stop at Add Step: {:>3d} time_slice: {:.1f}, Ranking Result: {:.3f} (p-value: {:.3f})".format(
                                step, time_slice, ranking_correlation, ranking_p_value_current))
                            if time_slice in train_steps_recording:
                                train_steps_recording[time_slice] += [step, "Add stage, Ranking Result: {:.3f} (p-value: {:.3f})".format(ranking_correlation, ranking_p_value_real)]
                            else:
                                train_steps_recording[time_slice] = [step, "Add stage, Ranking Result: {:.3f} (p-value: {:.3f})".format(ranking_correlation, ranking_p_value_real)]
                            break

            with open(os.path.join(save_dir, "train_steps_recording.json"), "w") as json_file:
                json.dump(train_steps_recording, json_file)
                json_file.close()


if __name__ == "__main__":
    tf.app.run()
