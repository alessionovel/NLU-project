import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        
        # Set as embedding algorithm the one from PyTorch
        # Padding index will be set to 'pad_index' in order to ignore them during the training
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        # Set LSTM as recurrent layer
        # emb_size: size of each embedding vector
        # hid_size: number of features in the hidden state h (the bigger, the "powerful")
        # n_layers: number of LSTM stacked upon each other
        # batch_first=True: tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=False, batch_first=True)    
        
        # Output layers (slots and intents)
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        
    def forward(self, utterance, seq_lengths):

        # Transform the input sequence into an embedding sequence
        utt_emb = self.embedding(utterance)
        
        # Use pack_padded_sequence to handle variable-length sequences
        # This ignore the padded tokens in the LSTM computation
        # In this case this is needed because we have to know the the value of the real last hidden state
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)

        # Give the embedding sequence to the RNN
        # The second output is the hidden state at the last time step (we'll use it for intent classification)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
       
        # Use pad_packed_sequence to return to padded sequences (to run the linear layer)
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Get the last hidden state from the last LSTM layer
        # Since batch_first=True, the shape is (n_layers, batch, hid_size)
        last_hidden = last_hidden[-1,:,:]
        
        # Compute slots and intent outputs (logits)
        # From hidden size to output size
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        
        # Permute to have (batch, vocab size, seq) as output shape (required by nn.CrossEntropyLoss)
        slots = slots.permute(0, 2, 1) 
        return slots, intent
    
class ModelIAS_Bidirectional(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS_Bidirectional, self).__init__()
        
        # Set as embedding algorithm the one from PyTorch
        # Padding index will be set to 'pad_index' in order to ignore them during the training
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        # Set LSTM as recurrent layer
        # emb_size: size of each embedding vector
        # hid_size: number of features in the hidden state h (the bigger, the "powerful")
        # n_layers: number of LSTM stacked upon each other
        # batch_first=True: tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)    
        
        # Output layers (slots and intents)
        self.slot_out = nn.Linear(hid_size * 2, out_slot)
        self.intent_out = nn.Linear(hid_size * 2, out_int)
        
    def forward(self, utterance, seq_lengths):

        # Transform the input sequence into an embedding sequence
        utt_emb = self.embedding(utterance)
        
        # Use pack_padded_sequence to handle variable-length sequences
        # This ignore the padded tokens in the LSTM computation
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)

        # Give the embedding sequence to the RNN
        # The second output is the hidden state at the last time step (we'll use it for intent classification)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
       
        # Use pad_packed_sequence to return to padded sequences (to run the linear layer)
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Get the last hidden state from the last LSTM layer
        # Since batch_first=True, the shape is (n_layers, batch, hid_size * 2)
        # We have to concatenate the last forward and last backward hidden states
        last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1)
        
        # Compute slots and intent outputs (logits)
        # From hidden size to output size
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        
        # Permute to have (batch, vocab size, seq) as output shape (required by nn.CrossEntropyLoss)
        slots = slots.permute(0, 2, 1) 
        return slots, intent

class ModelIAS_Dropout(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, emb_dropout=0.0, out_dropout=0.0):
        super(ModelIAS_Dropout, self).__init__()
        
        # Set as embedding algorithm the one from PyTorch
        # Padding index will be set to 'pad_index' in order to ignore them during the training
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        # Set LSTM as recurrent layer
        # emb_size: size of each embedding vector
        # hid_size: number of features in the hidden state h (the bigger, the "powerful")
        # n_layers: number of LSTM stacked upon each other
        # batch_first=True: tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)    
        
        # Output layers (slots and intents)
        self.slot_out = nn.Linear(hid_size * 2, out_slot)
        self.intent_out = nn.Linear(hid_size * 2, out_int)

        # Set the dropout layers
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.out_dropout = nn.Dropout(p=out_dropout)
        
    def forward(self, utterance, seq_lengths):

        # Transform the input sequence into an embedding sequence
        utt_emb = self.embedding(utterance)
        # Apply dropout to the embeddings
        utt_emb = self.emb_dropout(utt_emb)
        
        # Use pack_padded_sequence to handle variable-length sequences
        # This ignore the padded tokens in the LSTM computation
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)

        # Give the embedding sequence to the RNN
        # The second output is the hidden state at the last time step (we'll use it for intent classification)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
       
        # Use pad_packed_sequence to return to padded sequences (to run the linear layer)
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Apply dropout to the LSTM output for slots
        utt_encoded = self.out_dropout(utt_encoded)
        
        # Get the last hidden state from the last LSTM layer
        # Since batch_first=True, the shape is (n_layers, batch, hid_size * 2)
        # We have to concatenate the last forward and last backward hidden states
        last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1)

        # Apply dropout to the LSTM output for intents
        last_hidden = self.out_dropout(last_hidden)
        
        # Compute slots and intent outputs (logits)
        # From hidden size to output size
        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        
        # Permute to have (batch, vocab size, seq) as output shape (required by nn.CrossEntropyLoss)
        slots = slots.permute(0, 2, 1) 
        return slots, intent