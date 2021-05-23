import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import nltk
import json
import numpy as np
from torch.utils.data import Dataset
import random
from torch.jit import script, trace
import csv
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import pickle


##### CLASSES #####
###########################################################################
class NlpDataset(Dataset):

    """
    - uses the Dataset module from torch.utils.data
    - good for setting up training data using DataLoader from the same module,
    which splits up your pairs into mini_batches that are ready to use in 
    pytorch models.

    """
    def __init__(self, X_train, Y_train):
        super(NlpDataset, self).__init__()
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train


    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    
    def __len__(self):
        return self.n_samples

class LxtClassifierVocab:
    def __init__(self):
        super(LxtClassifierVocab, self).__init__()
        self.word_to_ix = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.ix_to_word = {0 : "<PAD>", 1 : "<SOS>", 2 : "<EOS>", 3 : "<UNK>"}
        self.ntoken = len(self.word_to_ix)
        self.ix_to_lbl = {}
        self.lbl_to_ix = {}

    def create_vocab(self, data):
        """
        Add correct data to:
            -> word_to_idx
            -> ix_to_word
            -> ntoken
        """
        for sent, label in data:
            for word in sent:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
                    self.ix_to_word[len(self.ix_to_word)] = word
                    self.ntoken += 1
            if label not in self.lbl_to_ix:
                self.lbl_to_ix[label] = len(self.lbl_to_ix)
                self.ix_to_lbl[len(self.ix_to_lbl)] = label

class ChatbotVocab:
    def __init__(self):
        super(ChatbotVocab, self).__init__()
        self.word_to_ix = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.ix_to_word = {0 : "<PAD>", 1 : "<SOS>", 2 : "<EOS>", 3 : "<UNK>"}
        self.ntoken = len(self.word_to_ix)

    def create_vocab(self, data):
        """
        Add correct data to:
            -> word_to_idx
            -> ix_to_word
            -> ntoken
        """
        for sent, reply in data:
            for word in sent:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
                    self.ix_to_word[len(self.ix_to_word)] = word
                    self.ntoken += 1
            for word in reply:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
                    self.ix_to_word[len(self.ix_to_word)] = word
                    self.ntoken += 1

##### PREPROCESSING FUNCTIONS #####
###########################################################################
def load_classifier_data(json_file):
    """
    returns -> a list of tuples in the format (sentence, label)
    ex )  [(['hey', 'there'], 'casual'), (['I', 'want', 'to', 'make', 'a', 'booking'], 'booking')]

    PRE-CONDITIONS:
        -> json file must be in the format: 
        {
            "patterns":
            {
                "pattern_name":
                {
                    "user_input": "response",
                            ...
                }, 
                ...
            }
        }
    """
    data_out = list()
    with open(json_file, 'r') as f:
        data_in = json.load(f)
    for label in data_in["patterns"].keys():
        for user_input in data_in["patterns"][label].keys():
            data_out.append((user_input, label))
    return data_out

def load_classifier_X_Y_train(data, vocab):
    """
    i) Turn user_input into bow vector and add it to X_train
    2) append labels ix to Y_train

    PRE-CONDITIONS:
        -> Vocab must have word_to_ix, label_to_ix
    """

    X_train = list()
    Y_train = list()
    for user_input, label in data:
        # setup X
        tokenized = tokenize_sentence(user_input)
        bow = make_bow_vector(tokenized, vocab.word_to_ix)
        X_train.append(bow)
        # setup Y
        ix = vocab.lbl_to_ix[label]
        Y_train.append(ix)

    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.int64)

    return X_train, Y_train

def load_chatbot_data(chatbot_json, is_tagged=True):
    """
    -> Chatbot data should map input sentence to a corresponding reply as a label.
        ex) [(["Hello", "There"], ["Hello", ".", "How", "are", "you", "?"])]

    returns -> a list of tuples in the format (sentence, reply)
    """
    
    data_out = list()
    with open(chatbot_json, 'r') as f:
        data_in = json.load(f)
    for label in data_in["patterns"].keys():
        pattern = data_in["patterns"][label]
        for user_input, utter_reply in pattern.items():
            data_out.append( (user_input, utter_reply) )
    return data_out
            

def tokenize_sentence(sentence, is_tagged=False):

    """
    Takes in a sentence (string) and turns it into a list of words

    - if is_tagged=True then a <SOS> and <EOS> token is added to the start
    and end of the tokenized sentence, resepectively.
    """

    sentence = sentence.lower()
    sentence = nltk.word_tokenize(sentence)
    if is_tagged is True:
        new_sentence = ['<SOS>']
        new_sentence.extend(sentence)
        new_sentence.append('<EOS>')
        sentence = new_sentence
    return sentence

def join_sentence(sentence):
    """
    Where sentence is a tokenized list ex) ["hello", "world"] --> "Hello World"
    """
    final_string = ""
    for i in range(0, len(sentence)):
        if i is not (len(sentence)-1):
            final_string += (str(sentence[i])+" ")
        else:
            final_string += sentence[i]
    return final_string

    
def make_bow_vector(sentence, word_to_ix):
    """
    Create bag of word vectors for training data
    ex) [1, 0, 0, 0] ...
    """
    vec = np.zeros(len(word_to_ix))
    for word in sentence:
        if word not in word_to_ix.keys():
            word = "<UNK>"
        vec[word_to_ix[word]] += 1
    return vec


def dump_pickle(data, filename):
    """
    save data to a given filename using pickle
    """
    outfile = open(filename, "wb")
    pickle.dump(data, outfile)
    outfile.close()

def load_pickle(filename):
    """
    load data from a given filename using pickle
    """
    infile = open(filename, "rb")
    data_out = pickle.load(infile)
    infile.close()
    return data_out


##### SEQUENCE MODEL FUNCTIONS #####
###########################################################################
def get_lengths(X_batch, device, isTransposed=True, isSorted=False):
    """
    returns -> a tensor of sentence lengths in a given batch 
    -> used in the torch.nn.utils.rnn.pack_padded_sequence function,
    -> if isSorted=False, then pack_padded_sequence(enforce_sorted=False)
    -> if isSorted=True, then pack_padded_sequence(enforce_sorted=True), and batch should be sorted aswell
    """
    if isTransposed:
        # undo the transpose
        x = X_batch.transpose(0, 1)
    else:
        x = X_batch
    lengths = list()
    for sentence in x:
        length = 0
        for token in sentence:
            if token == 0:
                break
            length += 1
        lengths.append(length)
    if isSorted:
        lengths.sort()
        lengths.reverse()
    lengths = torch.tensor(lengths)
    lengths = lengths.to(device)
    return lengths

def create_sequence_training_batches(data, word_to_idx, batch_size, max_length):
    """
    -> Creates (X_batch, Y_batch) data to use in sequence models where:
            - each row represents a single time step
            - each colum represents a single sequence, that is of max_length

    -> creates Y_train in order to compute loss 

    -> returns a list of tuples, each representing a mini_batch

    ex) [ ( |Torch.Tensor, shape=(max_length, batch_size)|, |Torch.Tensor, shape=(max_length, batch_size)| ) ],
            where max_length represents the maximum length of the input and output sequence
    """
    mini_batches = list()
    # Divide the dataset into batch_size parts.
    n_batches = len(data) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    final_idx = n_batches*batch_size
    data = data[:final_idx]
    # Init indexing
    start_idx = 0
    end_idx = batch_size
    while end_idx <= final_idx:
        X_batch = list()
        Y_batch = list()
        for d in data[start_idx:end_idx]:
            # setup input sequence
            X = list()
            for word in d[0]:
                X.append(word_to_idx[word])
            if len(X) > max_length:
                X[max_length-1] = word_to_idx["<EOS>"]
                X = X[:max_length]
            else:
                while len(X) < max_length:
                    X.append(word_to_idx["<PAD>"])
            X_batch.append(X)
            # setup output sequence
            Y = list()
            for word in d[1]:
                Y.append(word_to_idx[word])
            if len(Y) > max_length:
                Y[max_length-1] = word_to_idx["<EOS>"]
                Y = Y[:max_length]
            else:
                while len(Y) < max_length:
                    Y.append(word_to_idx["<PAD>"])
            Y_batch.append(Y)
        # add data to mini_batches list
        X_batch = torch.tensor(X_batch).transpose(0, 1)
        Y_batch = torch.tensor(Y_batch).transpose(0, 1)
        mini_batches.append((X_batch, Y_batch))
        # onto the next batch
        start_idx=end_idx
        end_idx+=batch_size
    return mini_batches

##### CHATBOT INFERENCE FUNCTIONS #####
###########################################################################
def classify(user_input, model, vocab, device):
    """
    Classify whether the given user_input is 'book', 'faq', 'greet', 'goodbye'
    """
    tokenized = tokenize_sentence(user_input)
    bow = make_bow_vector(tokenized, vocab.word_to_ix)
    bow = torch.tensor(bow, dtype=torch.float32).to(device)
    output = model(bow)
    with torch.no_grad():
        probabilities = F.softmax(output, dim=0)
        predicted_ix = torch.argmax(probabilities).item()
    predicted = vocab.ix_to_lbl[predicted_ix]
    return predicted

def utter_reply(user_input, searcher, vocab, max_length, device):
    """
    Respond to a given user input with a witty reply
    """
    processed = preprocess_seq2seq_input_for_chat(user_input, vocab, max_length)
    processed = processed.to(device)
    lengths = get_lengths(processed, device)
    tokens, scores = searcher(processed, lengths, max_length)
    words = list()
    for token in tokens:
        ix = token.item()
        word = vocab.ix_to_word[ix]
        words.append(word)
    reply = join_sentence(words)
    return reply
    
def preprocess_seq2seq_input_for_chat(user_input, vocab, max_length):
    """
    takes in a user_input (string) and turns it into a tensor of indices (padded and of size max_length)
    """
    ix_batch = list()
    tokenized = tokenize_sentence(user_input)
    for token in tokenized:
        if token not in vocab.word_to_ix.keys():
            token = "<UNK>"
            ix = vocab.word_to_ix[token]
            ix_batch.append(ix)
        else:
            ix = vocab.word_to_ix[token]
            ix_batch.append(ix)
    if len(ix_batch) > max_length:
        ix_batch[max_length-1] = vocab.word_to_ix["<EOS>"]
        ix_batch = ix_batch[:max_length]
    else:
        while len(ix_batch) < max_length:
            ix_batch.append(vocab.word_to_ix["<PAD>"])
    
    ix_batch = torch.LongTensor(ix_batch)
    ix_batch = ix_batch.reshape(-1, 1)

    return ix_batch


