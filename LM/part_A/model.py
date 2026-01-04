import torch
import torch.nn as nn

class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_RNN, self).__init__()
        
        # Set as embedding algorithm the one from PyTorch
        # Padding index will be set to 'pad_index' in order to ignore them during the training
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # Set RNN as recurrent layer
        # emb_size: size of each embedding vector
        # hidden_size: number of features in the hidden state h (the bigger, the "powerful")
        # n_layers: number of RNN stacked upon each other
        # batch_first=True: tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        
        # Set padding token index
        self.pad_token = pad_index
        
        # Set the linear layer that compute the output starting from the hidden state
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):

        # Transform the input sequence into an embedding sequence
        emb = self.embedding(input_sequence)
        
        # Give the embedding sequence to the RNN
        # First output is the sequence of embedding outputs (one for each time step)
        # The second output is the hidden state at the last time step (we don't need it here)
        rnn_out, _ = self.rnn(emb)
        
        # Transform the RNN output into the output sequence (vocabolary dimension)
        # Permute to have (batch, vocab size, seq) as output shape (required by nn.CrossEntropyLoss)
        output = self.output(rnn_out).permute(0, 2, 1)
        
        return output
    
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM, self).__init__()
        
        # Set as embedding algorithm the one from PyTorch
        # Padding index will be set to 'pad_index' in order to ignore them during the training
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # Set LSTM as recurrent layer
        # emb_size: size of each embedding vector
        # hidden_size: number of features in the hidden state h (the bigger, the "powerful")
        # n_layers: number of LSTM stacked upon each other
        # batch_first=True: tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        
        # Set padding token index
        self.pad_token = pad_index
        
        # Set the linear layer that compute the output starting from the hidden state
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):

        # Transform the input sequence into an embedding sequence
        emb = self.embedding(input_sequence)
        
        # Give the embedding sequence to the LSTM
        # First output is the sequence of embedding outputs (one for each time step)
        # The second output is the hidden state at the last time step (we don't need it here)
        lstm_out, _ = self.lstm(emb)
        
        # Transform the LSTM output into the output sequence (vocabolary dimension)
        # Permute to have (batch, vocab size, seq) as output shape (required by nn.CrossEntropyLoss)
        output = self.output(lstm_out).permute(0, 2, 1)
        
        return output

class LM_LSTM_dropout(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1, emb_dropout=0.1, out_dropout=0.1):
        super(LM_LSTM_dropout, self).__init__()
        
        # Set as embedding algorithm the one from PyTorch
        # Padding index will be set to 'pad_index' in order to ignore them during the training
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # Set LSTM as recurrent layer
        # emb_size: size of each embedding vector
        # hidden_size: number of features in the hidden state h (the bigger, the "powerful")
        # n_layers: number of LSTM stacked upon each other
        # batch_first=True: tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)

        # Not used the pytorch dropout of LSTM because it does not work with 1 layer
        
        # Set padding token index
        self.pad_token = pad_index
        
        # Set the linear layer that compute the output starting from the hidden state
        self.output = nn.Linear(hidden_size, output_size)

        # Set the dropout layers
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.out_dropout = nn.Dropout(p=out_dropout)
        
    def forward(self, input_sequence):

        # Transform the input sequence into an embedding sequence
        emb = self.embedding(input_sequence)
        
        # Apply dropout to the embeddings
        emb = self.emb_dropout(emb)
        
        # Give the embedding sequence to the LSTM
        # First output is the sequence of embedding outputs (one for each time step)
        # The second output is the hidden state at the last time step (we don't need it here)
        lstm_out, _ = self.lstm(emb)
        
        # Apply dropout to the LSTM output
        lstm_out = self.out_dropout(lstm_out)
        
        # Transform the LSTM output into the output sequence (vocabolary dimension)
        # Permute to have (batch, vocab size, seq) as output shape (required by nn.CrossEntropyLoss)
        output = self.output(lstm_out).permute(0, 2, 1)
        
        return output