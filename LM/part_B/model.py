import torch
import torch.nn as nn

class LM_LSTM(nn.Module):
    # In this implementation, since we will then use weight trying, we use a single parameter 'hidden_size' for both embedding size and hidden size
    def __init__(self, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM, self).__init__()
        
        # Set as embedding algorithm the one from PyTorch
        # Padding index will be set to 'pad_index' in order to ignore them during the training
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=pad_index)
        
        # Set LSTM as recurrent layer
        # emb_size: size of each embedding vector (in this case equal to hidden_size)
        # hidden_size: number of features in the hidden state h (the bigger, the "powerful")
        # n_layers: number of LSTM stacked upon each other
        # batch_first=True: tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        
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

class LM_LSTM_weightTying(nn.Module):
    def __init__(self, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM_weightTying, self).__init__()
        
        # Set as embedding algorithm the one from PyTorch
        # Padding index will be set to 'pad_index' in order to ignore them during the training
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=pad_index)
        
        # Set LSTM as recurrent layer
        # emb_size: size of each embedding vector (in this case equal to hidden_size)
        # hidden_size: number of features in the hidden state h (the bigger, the "powerful")
        # n_layers: number of LSTM stacked upon each other
        # batch_first=True: tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        
        # Set padding token index
        self.pad_token = pad_index
        
        # Set the linear layer that compute the output starting from the hidden state
        self.output = nn.Linear(hidden_size, output_size)
        
        # Make the embedding and the output layer weights tied (share the same weights)
        # Since in pytorch the weights of a linear layer are stored in a transposed way with respect to the embedding layer, we can directly assign them
        self.output.weight = self.embedding.weight

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

class LM_LSTM_variationalDropout(nn.Module):
    def __init__(self, hidden_size, output_size, pad_index=0, n_layers=1, dropout_prob=0.1):
        super(LM_LSTM_variationalDropout, self).__init__()

        # We create an LSTM architecture by our own so we need to store these parameters
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        
        # Set as embedding algorithm the one from PyTorch
        # Padding index will be set to 'pad_index' in order to ignore them during the training
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=pad_index)
        
        # Since with nn.LSTM it is not possible to access the individual LSTM nodes to apply the dropout mask, we create a sequence of LSTMCells manually.
        # A cell for each layer
        self.cells = nn.ModuleList([
            nn.LSTMCell(hidden_size, hidden_size) for _ in range(n_layers)
        ])  
        
        # Set padding token index
        self.pad_token = pad_index
        
        # Set the linear layer that compute the output starting from the hidden state
        self.output = nn.Linear(hidden_size, output_size)
        
        # Make the embedding and the output layer weights tied (share the same weights)
        # Since in pytorch the weights of a linear layer are stored in a transposed way with respect to the embedding layer, we can directly assign them
        self.output.weight = self.embedding.weight

        # Set the dropout probability
        self.dropout_prob = dropout_prob

    def forward(self, input_sequence):

        # Transform the input sequence into an embedding sequence
        emb = self.embedding(input_sequence)

        # Get batch size and sequence length in order to know the dimensions of the dropout masks
        batch_size, seq_len, _ = emb.size()

        # Initialize hidden and cell states for each layer
        states = []
        for _ in range(self.n_layers):
            h = torch.zeros(batch_size, self.hidden_size).to(emb.device)
            c = torch.zeros(batch_size, self.hidden_size).to(emb.device)
            states.append((h, c))

        # Initialize masks for variational dropout
        # One mask per input and one mask per hidden state
        # I didn't implement a different mask for each gate because I should have rewritten the LSTM cell from scratch
        input_masks = []
        hidden_masks = []

        # We apply dropout only during training
        # We want to keep the same mask for all time steps
        if self.training:
            for _ in range(self.n_layers):
                # Scale to keep magnitude consistent
                scale = 1.0 / (1.0 - self.dropout_prob)

                # Input mask
                input_masks.append(torch.zeros(batch_size, self.hidden_size).to(emb.device).bernoulli_(1.0 - self.dropout_prob) * scale)

                # Hidden state mask
                hidden_masks.append(torch.zeros(batch_size, self.hidden_size).to(emb.device).bernoulli_(1.0 - self.dropout_prob) * scale)
        
        # Manual loop over time steps
        o = []
        for t in range(seq_len):
            # Input at time step t
            x = emb[:, t, :]

            # Loop over layers
            for layer_idx, cell in enumerate(self.cells):
                h, c = states[layer_idx]

                # Apply dropout masks (only in training)
                if self.training:
                    x = x * input_masks[layer_idx]
                    h = h * hidden_masks[layer_idx]
                
                # Pass through LSTM cell
                h_new, c_new = cell(x, (h, c))

                # Update states
                states[layer_idx] = (h_new, c_new)

                # Set input for next layer (the output of the current one)
                x = h_new
            
            # Store output of the lasy layer
            o.append(x)
        
        # Collect outputs over time steps into a single tensor
        lstm_out = torch.stack(o, dim=1)

        # Transform the LSTM output into the output sequence (vocabolary dimension)
        # Permute to have (batch, vocab size, seq) as output shape (required by nn.CrossEntropyLoss)
        output = self.output(lstm_out).permute(0, 2, 1)
        
        return output