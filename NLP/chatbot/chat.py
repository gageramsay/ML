from nlp_helpers import *
from model import SequenceClassifierNN, PaddedEncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F



hidden_sizes = {"dim1": 256, "dim2": 128, "dim3": 64}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##### LOAD DATA #####
###################################################################################################################
# Load classifier variables and models
# classifier = SequenceClassifierNN(classifier_vocab.ntoken, len(classifier_vocab.ix_to_lbl), hidden_sizes).to(device)
# classifier_data = torch.load("saved_data/classifier/classifier_v1.pth")
# classifier.load_state_dict(classifier_data["model_state_dict"])

load_seq2seq_file = "NLP/chatbot/saved_data/seq2seq/seq2seq_ml_25_bs_1000_lr_0.0001_es_1024_nLayer_2_do_0.1_am_general_optim_Adam.pth"
seq_2_seq_data = torch.load(load_seq2seq_file)

encoder_state_dict = seq_2_seq_data["encoder_state_dict"]
encoder_optim_state_dict = seq_2_seq_data["encoder_optim_state_dict"]
decoder_state_dict = seq_2_seq_data["decoder_state_dict"]
decoder_optim_state_dict = seq_2_seq_data["decoder_optim_state_dict"]
batch_size = seq_2_seq_data["batch_size"]
learning_rate = seq_2_seq_data["learning_rate"]
optim_type = seq_2_seq_data["optim_type"]
vocab_save_name = seq_2_seq_data["vocab_save_name"]
ntoken = seq_2_seq_data["ntoken"]
emb_size = seq_2_seq_data["emb_size"]
nlayer = seq_2_seq_data["nlayer"] 
dropout = seq_2_seq_data["dropout"]
max_length = seq_2_seq_data["max_length"]
attn_model = seq_2_seq_data["attn_model"]

# setup variables and models

chatbot_vocab = load_pickle("NLP/chatbot/saved_data/seq2seq/chatbot_vocab_nsamples_10125.pickle")
SOS_token = chatbot_vocab.word_to_ix["<SOS>"]
encoder = PaddedEncoderRNN(ntoken, emb_size, nlayer, dropout).to(device)
decoder = LuongAttnDecoderRNN(attn_model, ntoken, emb_size, ntoken, nlayer, dropout).to(device)
encoder.load_state_dict(encoder_state_dict)
decoder.load_state_dict(decoder_state_dict)
searcher = GreedySearchDecoder(encoder, decoder, device, SOS_token).to(device)




##### Chat #####
###################################################################################################################
print()
print(":::::::::::::")
print("Lets chat! enter 'q' to exit the chat")
print()
while True:
    user_input = input("> ")
    if user_input == 'q':
        break
    else:
        reply = utter_reply(user_input, searcher, chatbot_vocab, max_length, device)
        print()
        print(f"bot: {reply}")
        print()




