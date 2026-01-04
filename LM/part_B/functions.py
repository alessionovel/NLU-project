import torch
import torch.nn as nn
import math

# This function performs a training loop over the given batch and returns the average loss per token
def train_loop(data, optimizer, criterion, model, clip=5):
    # Set the model to training mode to enable gradient computation and some layers' behaviors
    model.train()

    # Lists to accumulate loss and number of tokens
    loss_array = []
    number_of_tokens = []

    # Iterate over each sample in the batch to update model parameters
    for sample in data:
        # Reset gradients in order to avoid accumulation from previous iterations
        optimizer.zero_grad()
        # Perform a forward pass
        output = model(sample['source'])
        # Compute the loss
        loss = criterion(output, sample['target'])

        # Accumulate loss and number of tokens for reporting
        # In this case, the eval_criterion uses sum reduction, so we can directly accumulate the loss without multiplying by number of tokens
        # We want to normalize the loss with the length of the sequence, so we will make a weighted average
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])

        # Backpropagation Through Time (BPTT), as described in the paper.
        loss.backward()
        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # Update model parameters
        optimizer.step()

    # Return the average loss per token for the batch
    return sum(loss_array) / sum(number_of_tokens)

# This function performs an evaluation loop over the given batch and returns the perplexity
def eval_loop(data, eval_criterion, model):
    # Set the model to evaluation mode in order to be more efficient by disabling gradient computation and some layers' behaviors
    # Look at training loop for detailed comments
    model.eval()
    loss_array = []
    number_of_tokens = []
    # Disable gradient computation for evaluation to save memory and computations
    with torch.no_grad():
        for sample in data:
            # Get the prediction from the model
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            # In this case, the eval_criterion uses sum reduction, so we can directly accumulate the loss
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
    
    # Perplexity computation (exponential of the average loss per token)
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    return ppl

# Function called to initialize model weights in order to improve initial convergence
# Different strategies are applied depending on the layer type
def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.RNN, nn.LSTM, nn.GRU]:
            for name, param in m.named_parameters():
            # Removed orthogonal initialization since it is not supported with mps
            # I read that initialization is not that influent in practice, so it shouldn't affect the final results
                if 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        elif type(m) == nn.Linear:
            nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)