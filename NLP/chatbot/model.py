import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SequenceClassifierNN(nn.Module):
    """
    GOAL is to classify a sequence to a given label

    PRE-CONDITIONS:
        -> all of the sequences should be the same length
    """
    def __init__(self, ntoken, num_labels, hidden_sizes):
        super(SequenceClassifierNN, self).__init__()
        # sizes
        self.input_size = ntoken
        self.hidden_size_1 = hidden_sizes["dim1"]
        self.hidden_size_2 = hidden_sizes["dim2"]
        self.hidden_size_3 = hidden_sizes["dim3"]
        self.output_size = num_labels
        # layers
        self.dropout = nn.Dropout(0.1)
        self.l1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.l2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.l3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.out = nn.Linear(self.hidden_size_3, self.output_size)
        # activation
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input_batch):
        out = self.l1(input_batch)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.out(out)
        return out


class PaddedEncoderRNN(nn.Module):
    """
    Encoder part of the seq2seq architecture

    PRE-CONDITIONS:
        -> all of the sequences should be the same length (Use padding)
        -> input_batch should be of shape [max_length, batch_size]
    """
    # NOTE: input batch and lengths do not have to be sorted... (using enforce_sorted=False)

    def __init__(self, ntoken, emb_size, nlayers, dropout):
        super(PaddedEncoderRNN, self).__init__()
        # variables
        self.ntoken = ntoken
        self.emb_size = emb_size
        self.n_layers = nlayers
        # layers
        self.embedding = nn.Embedding(self.ntoken, self.emb_size)
        self.gru = nn.GRU(self.emb_size, self.emb_size, self.n_layers)
        self.l1 = nn.Linear(self.emb_size, 128)
        self.out = nn.Linear(64, self.ntoken)
        # activation
        self.relu = nn.ReLU()
    
    def forward(self, input_batch, input_lengths, hidden=None):
        embedded = self.embedding(input_batch)
        packed = pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        gru_output, gru_hidden = self.gru(packed, hidden)
        output, _ = pad_packed_sequence(gru_output)
        return output, gru_hidden

# Luong attention Layer
#####
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()
        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
        
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, input_size, hidden_size, output_size, n_layers, dropout):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        # Define layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attention = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step) 
        embedded = self.embedding_dropout(embedded)
        # Forward through the unidirectional gru
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate the attention weights from the current GRU output
        attention_weights = self.attention(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get "weighted sum" context vector
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and gru output using Luong 
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # predict next word using luong
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # return output and final hidden state
        return output, hidden


class GreedySearchDecoder(nn.Module):
    """
    Used for inference when implementing a conversational chatbot
    """
    def __init__(self, encoder, decoder, device, SOS_token):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.SOS_token = SOS_token
        self.device = device

    def forward(self, input_seq, input_length, max_length):
        # forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden layer input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self.device, dtype=torch.long) * self.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # forward pass
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # REcord token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        
        return all_tokens, all_scores


