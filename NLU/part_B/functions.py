import torch
import torch.nn as nn
import numpy as np
# Import the conll evaluation script that professor provided
from conll import evaluate
from sklearn.metrics import classification_report

# In this case we don't need to define PAD_TOKEN since we use -100 for ignore_index in slot labels
# PAD_TOKEN = 0

# This function performs a training loop over the given batch and returns the average loss per token
def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5, device='cpu'):
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
        
        # Move data to device (since we don't have a collate function we have to do it here)
        input_ids = sample['input_ids'].to(device)
        mask = sample['attention_mask'].to(device)
        intent_labels = sample['intent'].to(device)
        slot_labels = sample['slots'].to(device)
        
        # Perform a forward pass
        slots_logits, intent_logits = model(input_ids, mask)
        
        # Compute the losses
        loss_intent = criterion_intents(intent_logits, intent_labels)
        loss_slot = criterion_slots(slots_logits, slot_labels)
        loss = loss_intent + loss_slot
        
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
def eval_loop(data, criterion_slots, criterion_intents, model, lang, tokenizer, device='cpu'):
    # Set the model to evaluation mode in order to be more efficient by disabling gradient computation and some layers' behaviors
    # Look at training loop for detailed comments
    model.eval()
    loss_array = []
    ref_intents, hyp_intents = [], []
    ref_slots, hyp_slots = [], []
    
    # Disable gradient computation for evaluation to save memory and computations
    with torch.no_grad():
        for sample in data:
            # Move data to device (since we don't have a collate function we have to do it here)
            input_ids = sample['input_ids'].to(device)
            mask = sample['attention_mask'].to(device)
            intent_labels = sample['intent'].to(device)
            slot_labels = sample['slots'].to(device)
            
            # Get the predictions from the model
            slots_logits, intent_logits = model(input_ids, mask)
            loss_intent = criterion_intents(intent_logits, intent_labels)
            loss_slot = criterion_slots(slots_logits, slot_labels)
            loss_array.append((loss_intent + loss_slot).item())

            # Since metrics are computed using real labels, we need to convert the predicted IDs back to labels
            
            # Get predicted intents
            out_intents = [lang.id2intent[x] for x in torch.argmax(intent_logits, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in intent_labels.tolist()]

            # Prepare data for sklearn evaluation
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Get predicted slots
            output_slots = torch.argmax(slots_logits, dim=1)
            for id_seq, seq in enumerate(output_slots):
                
                # Get the ground truth labels (ids) for the current sequence
                gt_ids = slot_labels[id_seq].tolist()
                
                # Get the raw input ids
                curr_input_ids = input_ids[id_seq].tolist()
                
                # Temporary lists for the current sentence
                tmp_ref = []
                tmp_hyp = []
                
                # Iterate over the sequence length
                for id_word, word_id in enumerate(curr_input_ids):
                    
                    # Get the ground truth label id for the current word
                    gt_label_id = gt_ids[id_word]
                    
                    # We skip the tokens that are padding, special tokens or subword tokens
                    if gt_label_id != -100:
                        # Decode the word
                        word = tokenizer.decode([word_id])
                        
                        # Decode the Ground Truth slot
                        gt_slot = lang.id2slot[gt_label_id]
                        
                        # Decode the Predicted slot
                        pred_id = seq[id_word].item()
                        raw_slot = lang.id2slot[pred_id]
                        # Since it happened that in initial iterations (where the model is randomly initialized) 'pad' token was predicted, causing a error during evaluation, we hardcode it to 'O' (Outside) here
                        pred_slot = raw_slot if raw_slot != 'pad' else 'O'
                        
                        tmp_ref.append((word, gt_slot))
                        tmp_hyp.append((word, pred_slot))
                
                # Prepare data for conll evaluation
                ref_slots.append(tmp_ref)
                hyp_slots.append(tmp_hyp)
                
    # Compute metrics for slots using conll evaluation
    # It will receive only the first token of each word (since we compute labels only for them)
    results_slots = evaluate(ref_slots, hyp_slots)
    # Compute metrics for intents using sklearn
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    
    return results_slots, report_intent, np.mean(loss_array)

# In this case we don't need an init_weights function since we use the pretrained BERT weights as starting point