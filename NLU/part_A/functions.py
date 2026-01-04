import torch
import torch.nn as nn
import numpy as np
# Import the conll evaluation script that professor provided
from conll import evaluate
from sklearn.metrics import classification_report

# Special token IDs
PAD_TOKEN = 0

# This function performs a training loop over the given batch and returns the average loss per token
def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    # Set the model to training mode to enable gradient computation and some layers' behaviors
    model.train()

    # List to accumulate loss
    loss_array = []
    # In this case we don't need to count tokens since the loss is not dependent on the length of the sequences
    # We want accuracy per utterance, so we don't need to normalize by length

    # Iterate over each sample in the batch to update model parameters
    for sample in data:
        # Reset gradients in order to avoid accumulation from previous iterations
        optimizer.zero_grad()
        # Perform a forward pass
        slots, intents = model(sample['utterances'], sample['slots_len'])
        # Compute the losses
        loss_intent = criterion_intents(intents, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_slot + loss_intent

        # Accumulate loss for reporting
        loss_array.append(loss.item())

        # Backpropagation Through Time (BPTT), as described in the paper.
        loss.backward()
        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # Update model parameters
        optimizer.step()

    # Return the average loss over all samples
    return np.mean(loss_array)

# This function performs an evaluation loop over the given batch and returns the various metrics
def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    # Set the model to evaluation mode in order to be more efficient by disabling gradient computation and some layers' behaviors
    # Look at training loop for detailed comments
    model.eval()
    loss_array = []
    ref_intents, hyp_intents = [], []
    ref_slots, hyp_slots = [], []

    # Disable gradient computation for evaluation to save memory and computations
    with torch.no_grad():
        for sample in data:
            # Get the predictions from the model
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss_array.append((loss_intent + loss_slot).item())

            # Since metrics are computed using real labels, we need to convert the predicted IDs back to labels

            # Get predicted intents
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]

            # Prepare data for sklearn evaluation
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Get predicted slots
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'][id_seq].item()
                utterance = [lang.id2word[x.item()] for x in sample['utterances'][id_seq][:length]]
                gt_slots = [lang.id2slot[x.item()] for x in sample['y_slots'][id_seq][:length]]
                # Since it happened that in initial iterations (where the model is randomly initialized) 'pad' token was predicted, causing a error during evaluation, we hardcode it to 'O' (Outside) here
                out_slots = [lang.id2slot[x.item()] if lang.id2slot[x.item()] != 'pad' else 'O' for x in seq[:length]]
                
                # Prepare data for conll evaluation
                ref_slots.append([(utterance[i], gt_slots[i]) for i in range(length)])
                hyp_slots.append([(utterance[i], out_slots[i]) for i in range(length)])
    
    # Compute metrics for slots using conll evaluation
    results_slots = evaluate(ref_slots, hyp_slots)
    # Compute metrics for intents using sklearn
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    
    return results_slots, report_intent, np.mean(loss_array)

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