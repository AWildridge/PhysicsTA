import csv
import random

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch.utils.data import TensorDataset, DataLoader

from models import LSTMClassifier
from text_utils import Vocabulary


def process_df(df, vocab_name, remove_stopwords=False):
    df = df.reset_index()[['Submission', 'Attempt', 'Attempt.1', 'Awarded.1']]
    len_submission = [len(df['Submission'][i]) for i in range(len(df))]
    df['SubmissionLength'] = len_submission

    stop_words = set(stopwords.words('english'))
    filtered_submissions = []
    num_word_tokens = []
    for submission in df['Submission']:
        word_tokens = word_tokenize(submission)
        filtered_sentence = ['SOS']
        if remove_stopwords:
            filtered_sentence += [w.lower() for w in word_tokens if
                                  ((not w.lower() in stop_words) and (np.char.isalpha(w)))]
        else:
            filtered_sentence += [w.lower() for w in word_tokens if np.char.isalpha(w)]
        filtered_sentence.append('EOS')
        filtered_submissions.append(filtered_sentence)
        num_word_tokens.append(len(filtered_sentence))
    df['FilteredSubmission'] = filtered_submissions
    df['NumTokens'] = num_word_tokens

    vocab = Vocabulary(vocab_name)
    # vocab_size = []
    # num_sentences = []
    for filtered_submission in df['FilteredSubmission']:
        vocab.add_sentence(filtered_submission)
        # vocab_size.append(vocab.num_words)
        # num_sentences.append(len(vocab_size))

    '''plt.plot(num_sentences, vocab_size)
    plt.xlabel('# of Sentences Added to Vocab')
    plt.ylabel('# of Words in Vocab')
    plt.show()'''

    vectorized_seqs = []
    for filtered_submission in df['FilteredSubmission']:
        submission_seq = []
        for word in filtered_submission:
            if word is 'SOS' or word is 'EOS':
                submission_seq.append(vocab.to_index(word))
            else:
                submission_seq.append(vocab.to_index(word.lower()))
        padded_seq = np.zeros(vocab.longest_sentence, dtype=int)
        padded_seq[-len(submission_seq):] = np.array(submission_seq)[:vocab.longest_sentence]
        vectorized_seqs.append(padded_seq)
    df['VectorizedFilteredSubmission'] = vectorized_seqs
    return df, vocab


def prepare_dataloaders(df):
    # Class count
    count_class_1, count_class_0 = df['Awarded.1'].value_counts()

    # Divide by class
    df_class_0 = df[df['Awarded.1'] == 0.0]
    df_class_1 = df[df['Awarded.1'] == 1.0]
    class_0_idx = random.sample(range(len(df_class_0)), len(df_class_0))
    df_sample_class_0 = df_class_0.iloc[class_0_idx[:-int(len(df_class_0) / 2)]]

    df_class_0_over = df_sample_class_0.sample(int(count_class_1), replace=True)
    df_class_0 = df_class_0.iloc[class_0_idx[-int(len(df_class_0) / 2):]]

    class_1_idx = random.sample(range(len(df_class_1)), len(df_class_1))
    class_0_over_idx = random.sample(range(len(df_class_0_over)), len(df_class_0_over))

    pos_split_frac = 0.3  # 80% train/20@ val+test split for positive class
    pos_split_id = int(pos_split_frac * len(class_1_idx))
    if len(class_1_idx[-pos_split_id:]) % 2 == 1:
        pos_split_id += 1
    X_train_class1 = np.vstack(df_class_1['VectorizedFilteredSubmission'].values)[class_1_idx[:-pos_split_id]]
    y_train_class1 = np.vstack(df_class_1['Awarded.1'].values)[class_1_idx[:-pos_split_id]]
    X_test_class1 = np.vstack(df_class_1['VectorizedFilteredSubmission'].values)[class_1_idx[-pos_split_id:]]
    y_test_class1 = np.vstack(df_class_1['Awarded.1'].values)[class_1_idx[-pos_split_id:]]

    X_train_class0_over = np.vstack(df_class_0_over['VectorizedFilteredSubmission'].values)[
        class_0_over_idx[:-(len(df_class_0))]]
    y_train_class0_over = np.vstack(df_class_0_over['Awarded.1'].values)[class_0_over_idx[:-(len(df_class_0))]]
    X_test_class0_over = np.vstack(df_class_0_over['VectorizedFilteredSubmission'].values)[
        class_0_over_idx[-(len(df_class_0)):]]
    y_test_class0_over = np.vstack(df_class_0_over['Awarded.1'].values)[class_0_over_idx[-(len(df_class_0)):]]

    X_test_class0 = np.vstack(df_class_0['VectorizedFilteredSubmission'].values)
    y_test_class0 = np.vstack(df_class_0['Awarded.1'].values)

    X_train = np.concatenate((X_train_class1, X_train_class0_over))
    y_train = np.concatenate((y_train_class1, y_train_class0_over))

    X_test = np.concatenate((X_test_class1, X_test_class0_over, X_test_class0))
    y_test = np.concatenate((y_test_class1, y_test_class0_over, y_test_class0))

    val_split_idxs = random.sample(range(len(X_test)), len(X_test))
    neg_split_frac = 0.5  # 50% validation, 50% test
    neg_split_id = int(neg_split_frac * len(val_split_idxs))
    X_val, X_test = X_test[val_split_idxs[:neg_split_id]], X_test[val_split_idxs[neg_split_id:]]
    y_val, y_test = y_test[val_split_idxs[:neg_split_id]], y_test[val_split_idxs[neg_split_id:]]

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    batch_size = len(X_test)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return batch_size, train_loader, val_loader, test_loader


def find_failure_mode(df, name, vocab, epochs=30, output_size=1, criterion=None, lr=0.0005):
    # Over-sample the classes to ensure there is zero class imbalance
    # and split data into train, validation, and test sets
    batch_size, train_loader, val_loader, test_loader = prepare_dataloaders(df)

    # These will usually be more like 32 or 64 dimensional.
    # We will keep them small, so we can see how the weights change as we train.
    embedding_dim = 100
    hidden_dim = 10
    n_layers = 2

    model = LSTMClassifier(embedding_dim, hidden_dim, output_size, n_layers, vocab.num_words, drop_prob=0.7)
    model.to(device)

    if criterion is None:
        criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    counter = 0
    print_every = 1
    clip = 5
    valid_loss_min = np.Inf

    model.train()
    for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        h = model.init_hidden(batch_size)
        for inputs, labels in train_loader:
            if len(inputs) != batch_size:
                continue
            counter += 1
            h = tuple([e.data for e in h])
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            output, h = model(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if counter % print_every == 0:
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for inp, lab in val_loader:
                    val_h = tuple([each.data for each in val_h])
                    inp, lab = inp.to(device), lab.to(device)
                    out, val_h = model(inp, val_h)
                    val_loss = criterion(out.squeeze(), lab.float())
                    val_losses.append(val_loss.item())

                model.train()
                if np.mean(val_losses) <= valid_loss_min:
                    torch.save(model.state_dict(), './' + name + '.pt')
                    valid_loss_min = np.mean(val_losses)
    # Loading the best model
    model.load_state_dict(torch.load('./' + name + '.pt'))

    test_losses = []
    num_correct = 0
    h = model.init_hidden(batch_size)

    model.eval()
    for inputs, labels in test_loader:
        h = tuple([each.data for each in h])
        inputs, labels = inputs.to(device), labels.to(device)
        output, h = model(inputs, h)
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())
        pred = torch.round(output.squeeze())  # Rounds the output to 0/1
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    incorrect_indices = np.where(correct_tensor.cpu().detach().numpy() == 0)
    incorrects = inputs[incorrect_indices].cpu().detach().numpy()
    vectorized_data = np.vstack(df['VectorizedFilteredSubmission'].values)
    incorrect_lookup_table = np.array(
        [[np.array_equal(incorrects[i], vectorized_data[j]) for i in range(len(incorrects))] for j in
         range(len(vectorized_data))])

    failure_modes = [np.asarray(np.where(incorrect_lookup_table == True)).T[i][0] for i in
                     range(len(np.asarray(np.where(incorrect_lookup_table == True)).T))]
    return failure_modes


torch.manual_seed(1)
nltk.download('stopwords')
nltk.download('punkt')

data_filepath = '../data/Fa19_All_essays.csv'
data_filepath1 = '../data/PHYS172H_Fa19_EssayQuizzes.csv'
data_filepath2 = '../data/PHYS172SU_Fa19_EssayQuizzes.csv'

df = pd.read_csv(data_filepath, header=5)
df1 = pd.read_csv(data_filepath1, header=5)
df2 = pd.read_csv(data_filepath2, header=5)
df = pd.concat([df, df1, df2], sort=False).reset_index()

df_q5_p1 = df[['Submission', 'Correct', 'Award Detail', 'Time', 'Attempt', 'Awarded', 'Submission.1', 'Correct.1', 'Award Detail.1', 'Time.1', 'Attempt.1', 'Awarded.1']]
df_q4_p6 = df[['Submission.2', 'Correct.2', 'Award Detail.2', 'Time.2', 'Attempt.2', 'Awarded.2', 'Submission.3', 'Correct.3', 'Award Detail.3', 'Time.3', 'Attempt.3', 'Awarded.3', 'Submission.4', 'Correct.4', 'Award Detail.4', 'Time.4', 'Attempt.4', 'Awarded.4', 'Submission.5', 'Correct.5', 'Award Detail.5', 'Time.5', 'Attempt.5', 'Awarded.5', 'Submission.6', 'Correct.6', 'Award Detail.6', 'Time.6', 'Attempt.6', 'Awarded.6', 'Submission.7', 'Correct.7', 'Award Detail.7', 'Time.7', 'Attempt.7', 'Awarded.7']]
df_q3_pF = df[['Submission.8', 'Correct.8', 'Award Detail.8', 'Time.8', 'Attempt.8', 'Awarded.8', 'Submission.9', 'Correct.9', 'Award Detail.9', 'Time.9', 'Attempt.9', 'Awarded.9', 'Submission.10', 'Correct.10', 'Award Detail.10', 'Time.10', 'Attempt.10', 'Awarded.10']]
df_q1_p6 = df[['Submission.11', 'Correct.11', 'Award Detail.11', 'Time.11', 'Attempt.11', 'Awarded.11', 'Submission.12', 'Correct.12', 'Award Detail.12', 'Time.12', 'Attempt.12', 'Awarded.12']]

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print('=====================================')
print('~~~~~Processing Quiz 5 Problem 1~~~~~')
print('=====================================')

df_q5_p1_filter = df_q5_p1[['Submission', 'Attempt', 'Attempt.1', 'Awarded.1']]
num_entries = len(df_q5_p1)
df_q5_p1_filter = df_q5_p1_filter[pd.notnull(df_q5_p1_filter['Submission'])]
df_q5_p1_filter = df_q5_p1_filter[df_q5_p1_filter['Attempt.1'] == 1.0]
num_filter_entries = len(df_q5_p1_filter)
print("Number of duplicates or NaNs is: " + str(num_entries - num_filter_entries))
print("Number of data points is: " + str(num_filter_entries))
df_q5_p1_filter, q5_p1_vocab = process_df(df_q5_p1_filter, 'q5_p1')

q5_p1_failures = {}
for i in range(1000):
    print(i)
    failure_cases = find_failure_mode(df_q5_p1_filter, 'q5_p1_classifier' + str(i), q5_p1_vocab)
    for failure_idx in failure_cases:
        if failure_idx in q5_p1_failures.keys():
            q5_p1_failures[failure_idx] += 1
        else:
            q5_p1_failures[failure_idx] = 1

print('Saving failures from Quiz 5 Problem 1')
w = csv.writer(open("q5_p1_failures.csv", "w"))
for key, val in q5_p1_failures.items():
    w.writerow([key, val])

print('=====================================')
print('~~~~~Processing Quiz 4 Problem 6~~~~~')
print('=====================================')

filtered_df_q4_p6 = df_q4_p6[['Submission.2', 'Attempt.2', 'Submission.3', 'Attempt.3', 'Submission.4', 'Attempt.4', 'Submission.5', 'Attempt.5', 'Submission.6', 'Attempt.6', 'Submission.7', 'Attempt.7', 'Awarded.7']]
num_entries = len(filtered_df_q4_p6)

filtered_df_q4_p6 = filtered_df_q4_p6[pd.notnull(filtered_df_q4_p6['Submission.2'])]
filtered_df_q4_p6 = filtered_df_q4_p6[pd.notnull(filtered_df_q4_p6['Submission.3'])]
filtered_df_q4_p6 = filtered_df_q4_p6[pd.notnull(filtered_df_q4_p6['Submission.4'])]
filtered_df_q4_p6 = filtered_df_q4_p6[pd.notnull(filtered_df_q4_p6['Submission.5'])]
filtered_df_q4_p6 = filtered_df_q4_p6[pd.notnull(filtered_df_q4_p6['Submission.6'])]

filtered_df_q4_p6 = filtered_df_q4_p6.reset_index()

submissions = []
submission_attempts = []
for i in range(len(filtered_df_q4_p6)):
    submission = str(filtered_df_q4_p6['Submission.2'][i])
    submission += ' ' + str(filtered_df_q4_p6['Submission.3'][i])
    submission += ' ' + str(filtered_df_q4_p6['Submission.4'][i])
    submission += ' ' + str(filtered_df_q4_p6['Submission.5'][i])
    submission += ' ' + str(filtered_df_q4_p6['Submission.6'][i])
    attempt = filtered_df_q4_p6['Attempt.2'][i]
    attempt *= filtered_df_q4_p6['Attempt.3'][i]
    attempt *= filtered_df_q4_p6['Attempt.4'][i]
    attempt *= filtered_df_q4_p6['Attempt.5'][i]
    attempt *= filtered_df_q4_p6['Attempt.6'][i]
    submissions.append(submission)
    submission_attempts.append(attempt)
filtered_df_q4_p6['Submission'] = submissions
filtered_df_q4_p6['Attempt'] = submission_attempts
filtered_df_q4_p6['Awarded.1'] = filtered_df_q4_p6['Awarded.7']
filtered_df_q4_p6['Attempt.1'] = filtered_df_q4_p6['Attempt.7']

filtered_df_q4_p6 = filtered_df_q4_p6[pd.notnull(filtered_df_q4_p6['Submission'])]
# filtered_df_q4_p6 = filtered_df_q4_p6[filtered_df_q4_p6['Attempt'] == 1.0]
filtered_df_q4_p6 = filtered_df_q4_p6[filtered_df_q4_p6['Attempt.1'] == 1.0]
filtered_df_q4_p6 = filtered_df_q4_p6[['Submission', 'Attempt', 'Attempt.1', 'Awarded.1']]

num_filter_entries = len(filtered_df_q4_p6)
print("Number of duplicates or NaNs is: " + str(num_entries - num_filter_entries))
print("Number of data points is: " + str(num_filter_entries))

filtered_df_q4_p6, q4_p6_vocab = process_df(filtered_df_q4_p6, 'q4_p6')

q4_p6_failures = {}
for i in range(1000):
    print(i)
    failure_cases = find_failure_mode(filtered_df_q4_p6, 'q4_p6_classifier' + str(i), q4_p6_vocab, epochs=10, lr=0.005)
    for failure_idx in failure_cases:
        if failure_idx in q4_p6_failures.keys():
            q4_p6_failures[failure_idx] += 1
        else:
            q4_p6_failures[failure_idx] = 1

print('Saving failures from Quiz 4 Problem 6')
w = csv.writer(open("q4_p6_failures.csv", "w"))
for key, val in q4_p6_failures.items():
    w.writerow([key, val])

print('=====================================')
print('~~~~~Processing Quiz 3 Problem F~~~~~')
print('=====================================')

# TODO: Add multi-class classification correctly
'''
filtered_df_q3_pF = df_q3_pF[['Submission.8', 'Attempt.9', 'Awarded.9', 'Attempt.10', 'Awarded.10']]
filtered_df_q3_pF['Submission'] = filtered_df_q3_pF['Submission.8']

filtered_df_q3_pF = filtered_df_q3_pF[pd.notnull(filtered_df_q3_pF['Submission'])]
filtered_df_q3_pF['Attempt.1'] = filtered_df_q3_pF['Attempt.9'] * filtered_df_q3_pF['Attempt.10']
filtered_df_q3_pF['Awarded.1'] = filtered_df_q3_pF['Awarded.9'] + 2 * filtered_df_q3_pF['Awarded.10']
filtered_df_q3_pF = filtered_df_q3_pF[filtered_df_q3_pF['Attempt.1'] == 1.0]
filtered_df_q3_pF = filtered_df_q3_pF[['Submission', 'Attempt.1', 'Awarded.1', 'Awarded.9', 'Awarded.10']]

num_filter_entries = len(filtered_df_q3_pF)
print("Number of duplicates or NaNs is: " + str(num_entries - num_filter_entries))
print("Number of data points is: " + str(num_filter_entries))

filtered_df_q3_pF, q3_pF_vocab = process_df(filtered_df_q3_pF, 'q3_pF')

q3_pF_failures = {}
for i in range(1000):
    print(i)
    failure_cases = find_failure_mode(filtered_df_q3_pF, 'q4_p6_classifier' + str(i), q3_pF_vocab, )
    for failure_idx in failure_cases:
        if failure_idx in q3_pF_failures.keys():
            q3_pF_failures[failure_idx] += 1
        else:
            q3_pF_failures[failure_idx] = 1

print('Saving failures from Quiz 3 Problem F')
w = csv.writer(open("q3_pF_failures.csv", "w"))
for key, val in q3_pF_failures.items():
    w.writerow([key, val])
    '''

print('=====================================')
print('~~~~~Processing Quiz 1 Problem 6~~~~~')
print('=====================================')

filtered_df_q1_p6 = df_q1_p6[['Submission.11', 'Attempt.11', 'Attempt.12', 'Awarded.12']]
filtered_df_q1_p6 = filtered_df_q1_p6.rename(columns={'Submission.11': 'Submission', 'Attempt.11': 'Attempt',
                                                      'Attempt.12': 'Attempt.1', 'Awarded.12': 'Awarded.1'})
filtered_df_q1_p6 = filtered_df_q1_p6[pd.notnull(filtered_df_q1_p6['Submission'])]
filtered_df_q1_p6 = filtered_df_q1_p6[filtered_df_q1_p6['Attempt.1'] == 1.0]
filtered_df_q1_p6 = filtered_df_q1_p6[['Submission', 'Attempt', 'Attempt.1', 'Awarded.1']]

filtered_df_q1_p6, q1_p6_vocab = process_df(filtered_df_q1_p6, 'q1_p6')
q1_p6_failures = {}
for i in range(1000):
    print(i)
    failure_cases = find_failure_mode(filtered_df_q1_p6, 'q1_p6_classifier' + str(i), q1_p6_vocab, epochs=30, lr=0.001)
    for failure_idx in failure_cases:
        if failure_idx in q1_p6_failures.keys():
            q1_p6_failures[failure_idx] += 1
        else:
            q1_p6_failures[failure_idx] = 1

w = csv.writer(open("q1_p6_failures.csv", "w"))
for key, val in q1_p6_failures.items():
    w.writerow([key, val])
