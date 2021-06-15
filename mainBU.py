#!/usr/bin/env python
# coding: utf-8


import ast
import numpy as np

import Data_Drinks
import Data_Sandwhiches

NUM_OF_LABELS = 8  # including background


class NoiseReducer:

    def __init__(self, n_activities, t_prior_coeff=0, t_likelihood_coeff=0, t_centroid_coeff=0):
        self.n_activities = n_activities
        self.t_prior_coeff = t_prior_coeff
        self.t_likelihood_coeff = t_likelihood_coeff
        self.t_centroid_coeff = t_centroid_coeff
        self.max_len_trace = 0
        self.t_centroid = None
        self.recorded_traces = None

    def gradientDescent_multiple_variables(self, prior_traces, ground_truth_traces, t_likelihood, t_centroid, theta,
                                           alpha, num_iters):
        m = len(prior_traces)
        for i in range(num_iters):
            J = 0
            for prior_trace, y, t_like in zip(prior_traces, ground_truth_traces, t_likelihood):
                prior_trace = np.array(prior_trace)

                # get the posterior matix
                M_pred = theta[0] * prior_trace + theta[1] * t_like + theta[2] * t_centroid

                # update the weights theta
                theta[0] += (2 * alpha / m) * np.sum((y - M_pred) * prior_trace)
                theta[1] += (2 * alpha / m) * np.sum((y - M_pred) * t_like)
                theta[2] += (2 * alpha / m) * np.sum((y - M_pred) * t_centroid)

                # calculate the cost function after one iteration on all train data
                J += np.sum(np.square(y - M_pred))
            if i % 5 == 0:
                print(f'The loss at iteration {i} is {J / m}')

        return theta[0], theta[1], theta[2]

    def calc_t_likelihood_helper(self, traces, new_trace, traces_frequencies, distance_func, longest_trace_len,
                                 n_activities):
        distances = dict()
        for trace in traces:
            distance = distance_func(np.array(new_trace),
                                     np.array(self.trace_to_matrix(trace, n_activities, longest_trace_len)))
            distances[trace] = distance
        scores = self.calc_scores(distances)
        likelihood_mat = np.zeros((n_activities, longest_trace_len))
        denominator = 0

        for trace in traces:
            likelihood_mat += np.array(self.trace_to_matrix(trace, n_activities, longest_trace_len)) * \
                              traces_frequencies[trace] * scores[trace]
            denominator += traces_frequencies[trace] * scores[trace]
        return likelihood_mat / denominator

    def traces_frequency(self, traces):
        freq_dict = {}
        for trace in traces:
            t = tuple(trace)
            if t in freq_dict:
                freq_dict[t] += 1
            else:
                freq_dict[t] = 1

        return freq_dict

    def modify_newtrace_dims(self, trace, required_length):
        n_additional_columns = required_length - len(trace[0])
        for row in trace[:-1]:
            row += [0] * n_additional_columns
        trace[-1] += [1] * n_additional_columns  # background label
        return trace

    def modify_all_traces_dims(self, traces, required_length):
        modified_traces = []
        for trace in traces:
            modified_trace = self.modify_newtrace_dims(trace, required_length)
            modified_traces.append(modified_trace)
        return modified_traces

    def calc_scores(self, distances_dict):
        scores = dict()
        total_score = sum([np.exp(-value) for value in distances_dict.values()])

        for key in distances_dict.keys():
            scores[key] = np.exp(-distances_dict[key]) / total_score

        return scores

    def calculate_centroid(self, traces):
        longest_trace_len = max([len(trace) for trace in traces])
        centroid = np.zeros((self.n_activities, longest_trace_len))
        m = len(traces)
        for trace in traces:
            matrix_form = self.trace_to_matrix(trace, self.n_activities)
            if len(matrix_form[0]) != longest_trace_len:
                matrix_form = self.modify_newtrace_dims(matrix_form, longest_trace_len)
            centroid = centroid + matrix_form
        return centroid / m

    def convert_stringtraces_to_matrices(self, traces, n_activities, n_cols=None):
        traces_in_matrix_form = []
        for trace in traces:
            trace_matrix = self.trace_to_matrix(trace, n_activities=n_activities, n_cols=None)
            traces_in_matrix_form.append(trace_matrix)
        return traces_in_matrix_form

    def calc_t_likelihood(self, existing_traces, new_trace, n_activities):
        ''' Recieves existing_traces as list of strings and new_trace as matrix'''
        longest_trace_len = max(max([len(trace) for trace in existing_traces]),
                                len(new_trace[0]))  # new trace is a matrix
        if len(new_trace[0]) != longest_trace_len:
            new_trace = self.modify_newtrace_dims(new_trace, longest_trace_len)
        traces_frequencies = self.traces_frequency(existing_traces)
        unique_traces = [trace for trace in set(tuple(trace) for trace in existing_traces)]
        T_likelihood = self.calc_t_likelihood_helper(unique_traces, new_trace, traces_frequencies,
                                                     self.calc_Frobenius_norm, longest_trace_len, n_activities)
        return T_likelihood

    def calc_Frobenius_norm(self, mat1, mat2):
        return np.linalg.norm(mat1 - mat2)

    def t_likelihood_for_all_traces(self, noisy_traces, ground_truth_traces):
        likelihoods_lst = []
        for trace in noisy_traces:
            t_likelihood = self.calc_t_likelihood(ground_truth_traces, trace, self.n_activities)
            likelihoods_lst.append(t_likelihood)

        return likelihoods_lst

    def trace_to_matrix(self, trace, n_activities, n_cols=None):
        '''This is the original trace to matrix conventer. It get as input
            list of string and returns list of lists'''
        width = len(trace)
        trace_mat = [[0] * width for _ in range(n_activities)]

        for i in range(width):
            trace_mat[ord(trace[i]) - ord('A')][i] = 1

        if n_cols:
            if n_cols < width:
                raise ValueError("n_locs must be larger than the length of the trace")
            for trace in trace_mat:
                trace += [0] * (n_cols - width)

        return trace_mat

    def normalize_probabilities(self, posterior_trace):
        '''normalizes returned posterior trace prediction'''
        n_columns = len(posterior_trace[0])
        new_mat = []
        for column in posterior_trace.T:
            column_sum = column.sum()
            column = column / column_sum
            new_mat.append(column)
        new_mat = np.stack(new_mat, axis=1)
        return new_mat

    def normalize_coeff(self, t_prior_coeff, t_likelihood_coeff, t_centroid_coeff):
        '''normalizing coefficients of weights so the prediciton probability will sum to 1 for each activity'''
        coeff_sum = t_prior_coeff + t_likelihood_coeff + t_centroid_coeff
        return t_prior_coeff / coeff_sum, t_likelihood_coeff / coeff_sum, t_centroid_coeff / coeff_sum

    def train(self, X_train, Y_train, lr=0.0001, n_iter=1000):
        self.recorded_traces = Y_train
        self.max_len_trace = max([len(trace) for trace in Y_train])
        traces_likelihood_mats = self.t_likelihood_for_all_traces(X_train, Y_train)
        self.t_centroid = self.calculate_centroid(Y_train)
        Y_train_mats = self.convert_stringtraces_to_matrices(Y_train, self.n_activities)
        Y_train_modified_dimensions = self.modify_all_traces_dims(Y_train_mats, self.max_len_trace)
        self.t_prior_coeff, self.t_likelihood_coeff, self.t_centroid_coeff = self.gradientDescent_multiple_variables(
            X_train, Y_train_modified_dimensions, traces_likelihood_mats, self.t_centroid,
            [self.t_prior_coeff, self.t_likelihood_coeff, self.t_centroid_coeff], lr, n_iter)
        # self.t_prior_coeff, self.t_likelihood_coeff, self.t_centroid_coeff = self.normalize_coeff(t_prior_coeff, t_likelihood_coeff, t_centroid_coeff)

    def predict_converged(self, t_prior, traces=None, delta=0.005, dist_func=None, alpha=0.8, beta=0.1, gamma=0.1):
        if traces is None:
            traces = self.recorded_traces
        if dist_func is None:
            dist_func = self.calc_Frobenius_norm

        # compute initial values
        t_prior = np.array(t_prior)
        t_centroid = self.calculate_centroid(traces, self.n_activities)
        t_likelihood = self.calc_t_likelihood(traces, t_prior, n_activities)
        t_posterior_prev = alpha * t_prior + beta * t_likelihood + gamma * t_centroid

        # compute updated values
        t_likelihood_curr = self.calc_t_likelihood(traces, t_posterior_prev, n_activities)
        t_posterior_curr = alpha * t_posterior_prev + beta * t_likelihood_curr + gamma * t_centroid

        # iterate until convergence
        i = 0
        while dist_func(t_posterior_prev, t_posterior_curr) > delta:
            i += 1
            print(f'iteration #{i}')
            print('curr distance between traces: ', dist_func(t_posterior_prev, t_posterior_curr))
            t_posterior_prev = t_posterior_curr
            t_likelihood_curr = self.calc_t_likelihood(traces, t_posterior_prev, n_activities)
            t_posterior_curr = alpha * t_posterior_prev + beta * t_likelihood_curr + gamma * t_centroid

        return t_posterior_curr

    def predict(self, prior_trace):
        prior_trace = np.array(prior_trace)
        t_likelihood = self.calc_t_likelihood(self.recorded_traces, prior_trace, self.n_activities)

        t_posterior = self.t_prior_coeff * prior_trace + self.t_likelihood_coeff * t_likelihood + self.t_centroid_coeff * self.t_centroid
        # print('prior trace was: ')
        # for row in prior_trace:
        #    print(*row)
        # print()
        # print('posterior trace unnormalized is: ')
        # for row in t_posterior:
        #    print(*row)
        # print()
        t_posterior = self.normalize_probabilities(t_posterior)
        # print('posterior trace normalized is: ')
        # for row in t_posterior:
        #    print(*row)
        # print()
        return t_posterior

    # In[5]:


def txt_to_matrix(text):
    text = ((",".join(text.split(' '))).replace(',,', ', ').replace(',', ', '))
    matrix = ast.literal_eval(text)
    return np.array(matrix)


def label_preprocess(label, n_activities=None):
    if n_activities is None:
        return [item if len(item) == 1 else [item[0]] if len(item) > 1 else [-1] for item in label]
    return [item if len(item) == 1 else [item[0]] if len(item) > 1 else [n_activities - 1] for item in label]


# n_activities are total activities+1 for background which is the last row in the matrix
# activities are 0-6 and background is 7 so in total 8
def label_to_matrix(label, n_activities=8):
    width = len(label)
    label_mat = [[0] * width for _ in range(n_activities)]

    for i in range(width):
        label_mat[label[i][0]][i] = 1

    return label_mat


def swap_first_and_last_elements(probability_matrix):
    new_mat = []
    for timestamp in probability_matrix:
        rotated_timestamp = np.roll(np.array(timestamp), -1).tolist()
        new_mat.append(rotated_timestamp)
    return new_mat


def transpose_matix(m):
    return list(map(list, zip(*m)))


def prepare_labels(labels_lst, n_activities=8):
    matrix_labels = []

    for label in labels_lst:
        clean_label = label_preprocess(label)
        label_matrix = label_to_matrix(clean_label, n_activities)
        matrix_labels.append(label_matrix)
    string_labels = convert_matrices_to_string_vectors(matrix_labels, n_activities)
    return string_labels


def prepare_label_numeric(label, n_activities=None):
    label = label_preprocess(label, n_activities)
    label = [item[0] for item in label]
    label = np.array(label)
    return label


def modify_label_dimensions(label, required_dimension):
    n_additional_entries = required_dimension - label.size
    if n_additional_entries > 0:
        additional_entries = np.ones(n_additional_entries) * 7
        label = np.append(label, additional_entries)
    return label


def prepare_labels_numeric(labels, required_dimension, n_activities=None):
    prepared_labels = []
    for label in labels:
        label = prepare_label_numeric(label, n_activities)
        label = modify_label_dimensions(label, required_dimension)
        prepared_labels.append(label)
    return prepared_labels


def prepare_predictions(predictions_lst):
    post_processed_predictions = []

    for prediction in predictions_lst:
        pred_matrix = txt_to_matrix(prediction)
        pred_matrix = swap_first_and_last_elements(pred_matrix)
        pred_matrix_transpose = transpose_matix(pred_matrix)
        post_processed_predictions.append(pred_matrix_transpose)

    return post_processed_predictions


# In[6]:


def accuracy(vec1, vec2):
    if vec1.size != vec2.size:
        raise ValueError(
            f'The vectors must have same size! in this case first vector has a size of {vec1.size} while the second is of size {vec2.size}')
    return sum(vec1 == vec2) / vec1.size


# In[7]:


def matrix_to_string_vector(trace_matrix, n_activities=8):
    trace = []
    for col in list(zip(*trace_matrix)):
        one_idx = col.index(1)
        trace.append(chr(ord('A') + one_idx))
    return trace


def convert_matrices_to_string_vectors(matrices, n_activities=8):
    string_traces = []
    for mat in matrices:
        string_trace = matrix_to_string_vector(mat, n_activities)
        string_traces.append(string_trace)
    return string_traces


# In[8]:


def convert_matrices_to_final_predicitons(matrices):
    predictions = []
    for mat in matrices:
        prediction = probability_matrix_to_final_prediction(mat)
        predictions.append(prediction)
    return predictions


def probability_matrix_to_final_prediction(probability_matrix):
    probability_matrix = np.array(probability_matrix)
    indexes = probability_matrix.argmax(axis=0)
    return indexes


def convert_labels_to_final_predictions(label, n_activities=8):
    return np.array([item[0] if len(item) == 1 else item[0] if len(item) > 1 else n_activities - 1 for item in label])


# Sandwiches

sandwiches_actions = Data_Sandwhiches.actions
sandwiches_labels = Data_Sandwhiches.labels

X_train_sandwiches = prepare_predictions(sandwiches_actions)
Y_train_sandwiches = prepare_labels(sandwiches_labels)

model_sandwiches = NoiseReducer(NUM_OF_LABELS)
model_sandwiches.train(X_train_sandwiches, Y_train_sandwiches, lr=0.0001, n_iter=100)

# predicting accuracy for original neural network       # FIXME: acc_before is less realability - adding predictions and labels before calculating
final_predictions_raw_sandwiches = convert_matrices_to_final_predicitons(
    X_train_sandwiches)  # TODO: check for option to change code here
prepared_labels_sandwiches = prepare_labels_numeric(sandwiches_labels, 102, 8)
acc_before = []
for prediction, label in zip(final_predictions_raw_sandwiches, prepared_labels_sandwiches):
    acc_before.append(accuracy(prediction, label))
    print(accuracy(prediction, label))

# predicting accuracy for my algorithm
final_predictions_sandwiches = []
for trace in X_train_sandwiches:
    final_predictions_sandwiches.append(model_sandwiches.predict(trace))

final_predictions_sandwiches = convert_matrices_to_final_predicitons(final_predictions_sandwiches)  # argmax
prepared_labels_sandwiches = prepare_labels_numeric(sandwiches_labels, 102, 8)  # check what this do
acc_after = []
for prediction, label in zip(final_predictions_sandwiches, prepared_labels_sandwiches):
    acc_after.append(accuracy(prediction, label))
    print(accuracy(prediction, label))

# which activities are we missing the most? take, put? Is there a pattern?
acc_compare = list(zip(acc_before, acc_after))
print(acc_compare)

for prediction, label in zip(final_predictions_sandwiches, prepared_labels_sandwiches):
    print('models prediction: ', prediction)
    print('ground truth: ', label)

# # Drinks
#
# drinks_actions = Data_Drinks.actions
# drinks_labels = Data_Drinks.labels
#
# X_train_drinks = prepare_predictions(drinks_actions)
# Y_train_drinks = prepare_labels(drinks_labels)
#
# model_drinks = NoiseReducer(NUM_OF_LABELS)
# model_drinks.train(X_train_drinks, Y_train_drinks, lr = 0.0001, n_iter = 100)
#
#
# # predicting accuracy for original neural network
# final_predictions_raw = convert_matrices_to_final_predicitons(X_train_drinks)
# prepared_labels = prepare_labels_numeric(drinks_labels, 125, 8)
#
# for prediction, label in zip(final_predictions_raw, prepared_labels):
#     print(accuracy(prediction, label))
#
# # predicting accuracy for my algorithm
#
# final_predictions = []
# for trace in X_train_drinks:
#     final_predictions.append(model_drinks.predict(trace))
#
# final_predictions = convert_matrices_to_final_predicitons(final_predictions)
# prepared_labels = prepare_labels_numeric(drinks_labels, 125, 8)
#
# for prediction, label in zip(final_predictions, prepared_labels):
#     print(accuracy(prediction, label))


