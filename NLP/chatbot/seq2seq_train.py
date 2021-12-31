##### IMPORTS
from nlp_helpers import *
import torch
import torch.nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from model import PaddedEncoderRNN, LuongAttnDecoderRNN
from torch.utils.tensorboard import SummaryWriter
import sys
import time
##### SETUP DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.current_device()
print("type of device: ")
print(device.type)
torch.backends.cudnn.benchmark = True 
##### LOAD DATA
FILE = "datasets/intents.json"
data = load_chatbot_data(FILE)
vocab = ChatbotVocab()
vocab.create_vocab(data)
PAD_token = vocab.word_to_ix["<PAD>"]
SOS_token = vocab.word_to_ix["<SOS>"]
ntoken = vocab.ntoken
##### HYPERPARAMETERS
num_epochs = 100000
max_lengths = [25]
batch_sizes = [1000]
learning_rates = [0.0001]
emb_sizes = [1024]
nlayers = [2]
dropouts = [0.1]
attn_models = ["general"]
optim_types = ["Adam"]

##### GRID SEARCH
for max_length in max_lengths:
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for emb_size in emb_sizes:
                for nlayer in nlayers:
                        for dropout in dropouts:
                            for attn_model in attn_models:
                                for optim_type in optim_types:
                                    # Train 
                                    ##### MODEL VARIABLES
                                    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
                                    encoder = PaddedEncoderRNN(ntoken, emb_size, nlayer, dropout).to(device)
                                    decoder = LuongAttnDecoderRNN(attn_model, ntoken, emb_size, ntoken, nlayer, dropout).to(device)
                                    mini_batches = create_sequence_training_batches(data, vocab.word_to_ix, batch_size, max_length)
                                    print("number of samples: " + str(len(data)))
                                    print("number of batches: " + str(len(mini_batches))) 
                                    print()
                                    chatbot_pickle = dump_pickle(vocab, f"saved_data/seq2seq/chatbot_vocab_nsamples_{len(data)}.pickle")
                                    if optim_type == "SGD":
                                        encoder_optim = SGD(encoder.parameters(), lr=learning_rate, momentum=0.9)
                                        decoder_optim = SGD(decoder.parameters(), lr=learning_rate, momentum=0.9)
                                    elif optim_type == "Adam":
                                        encoder_optim = Adam(encoder.parameters(), lr=learning_rate)
                                        decoder_optim = Adam(decoder.parameters(), lr=learning_rate)
                                    save_name = f"seq2seq_ml_{max_length}_bs_{batch_size}_lr_{learning_rate}_es_{emb_size}_nLayer_{nlayer}_do_{dropout}_am_{attn_model}_optim_{optim_type}"
                                    ##### TRAINING LOOP
                                    print()
                                    print(f"TRAINING on {save_name}")
                                    print()
                                    current_loss = 100
                                    tb = SummaryWriter(f"runs/seq2seq/{save_name}")
                                    done_start_time = time.time()
                                    start_time = time.time()
                                    for epoch in range(0, num_epochs+1):
                                        epoch_loss = 0.0
                                        for mini_batch in mini_batches:
                                            # init forward pass
                                            encoder_optim.zero_grad()
                                            decoder_optim.zero_grad()
                                            input_batch = mini_batch[0].to(device)
                                            target_batch = mini_batch[1].to(device)
                                            input_lengths = get_lengths(input_batch, device)
                                            # forward
                                            encoder_output, encoder_hidden = encoder(input_batch, input_lengths)
                                            dec_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
                                            dec_input = dec_input.to(device)
                                            dec_hidden = encoder_hidden[:decoder.n_layers]
                                            for t in range(1, max_length):
                                                dec_output, dec_hidden = decoder(dec_input, dec_hidden, encoder_output)
                                                dec_input = target_batch[t].view(1, -1)
                                                if t == 1:
                                                    loss = criterion(dec_output, target_batch[t])
                                                else:
                                                    loss += criterion(dec_output, target_batch[t])
                                            loss = loss / max_length
                                            # backward
                                            loss.backward()
                                            _ = nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
                                            _ = nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
                                            encoder_optim.step()
                                            decoder_optim.step()
                                            # for printing 
                                            epoch_loss+=loss.item()
                                        if epoch % 100 == 0:
                                            end_time = time.time()
                                            epoch_time = end_time-start_time
                                            print("timelapse after 100 epoch: " +str(epoch_time))
                                            tb.add_scalar("Training Loss", epoch_loss/len(mini_batches), epoch)
                                            print(f"::::: Loss for epoch {epoch}/{num_epochs}: {epoch_loss/len(mini_batches)}")
                                            current_loss = epoch_loss/len(mini_batches)
                                            if current_loss <= 0.5:
                                                break
                                    ##### POST TRAINING
                                    done_end_time = time.time()
                                    all_done_time = done_end_time - done_start_time
                                    print("this all took " + str(all_done_time) + " seconds")
                                    tb.close()
                                    training_data = {
                                        "encoder_state_dict": encoder.state_dict(),
                                        "encoder_optim_state_dict": encoder_optim.state_dict(),
                                        "decoder_state_dict": decoder.state_dict(),
                                        "decoder_optim_state_dict": decoder_optim.state_dict(),
                                        "batch_size": batch_size,
                                        "learning_rate": learning_rate,
                                        "optim_type": optim_type,
                                        "vocab_save_name": "saved_data/seq2seq/chatbot_vocab.pickle",
                                        "ntoken" : vocab.ntoken,
                                        "emb_size": emb_size,
                                        "nlayer": nlayer,
                                        "dropout": dropout,
                                        "max_length": max_length,
                                        "attn_model": attn_model
                                    }
                                    save_file = f"saved_data/seq2seq/{save_name}.pth"
                                    torch.save(training_data, save_file)
                
