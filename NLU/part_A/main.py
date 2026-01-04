import torch
import torch.optim as optim
import torch.nn as nn
import json
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from functools import partial
from collections import Counter

# Import defined models and functions
from model import ModelIAS, ModelIAS_Bidirectional, ModelIAS_Dropout
from utils import load_data, Lang, IntentsAndSlots, collate_fn
from functions import train_loop, eval_loop, init_weights

# Device configuration (try mps since I have a mac with M2 chip)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Print the device being used
print(f"Using device: {DEVICE}")

# Parameters definition
config = {
    "experiment_name": "Dropout_IAS_AdamW",
    "optimizer": "AdamW",
    "hid_size": 800,
    "emb_size": 1200,
    "lr": 0.0005,
    "patience": 10,
    "batch_size": 128,
    "n_epochs": 100,
    "clip": 5,
    "emb_dropout": 0.6,
    "out_dropout": 0.7
}

if __name__ == "__main__":

    # Print configuration
    print("Experiment Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # Data loading
    tmp_train_raw = load_data('dataset/atis.train.json')
    test_raw = load_data('dataset/atis.test.json')

    # In this case the dataset does not provide a dev set, so we create it from the training set
    # Create dev set from train set (10% as standard)
    portion = 0.1
    intents_labels = [x['intent'] for x in tmp_train_raw]
    # Make the split stratified according to the intent labels
    count_y = Counter(intents_labels)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents_labels):
    # If some intents occurs only once, we put them directly in training
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    # Create vocabulary:
    #   For words, look only at training set
    #   For slots and intents, look at all sets (because we don't want 'unk' labels during evaluation)
    words = sum([x['utterance'].split() for x in train_raw], [])
    slots = set(sum([line['slots'].split() for line in (train_raw + dev_raw + test_raw)], []))
    intents = set([line['intent'] for line in (train_raw + dev_raw + test_raw)])
    
    lang = Lang(words, intents, slots, cutoff=0)

    # Create datasets (in PyTorch Dataset format)
    train_ds = IntentsAndSlots(train_raw, lang)
    dev_ds = IntentsAndSlots(dev_raw, lang)
    test_ds = IntentsAndSlots(test_raw, lang)

    # Create dataLoaders
    # Use collate_fn to pad sequences in the batch and hardcode the pad token index
    pad_idx = lang.word2id['pad']
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=partial(collate_fn, device=DEVICE))

    # In evaluation, we can use larger batches since we don't need to store gradients (so less memory usage)
    # The batch size of the evaluation does not affect the results
    dev_loader = DataLoader(dev_ds, batch_size=128, collate_fn=partial(collate_fn, device=DEVICE))
    test_loader = DataLoader(test_ds, batch_size=128, collate_fn=partial(collate_fn, device=DEVICE))

    # Initialize the model with the various parameters
    # Remember to change also the experiment_name in order to log and save the model correctly

    '''
    model = ModelIAS(
        hid_size=config["hid_size"], 
        out_slot=len(lang.slot2id), 
        out_int=len(lang.intent2id), 
        emb_size=config["emb_size"], 
        vocab_len=len(lang.word2id), 
        pad_index=pad_idx
    ).to(DEVICE)
    '''

    '''
    model = ModelIAS_Bidirectional(
        hid_size=config["hid_size"], 
        out_slot=len(lang.slot2id), 
        out_int=len(lang.intent2id), 
        emb_size=config["emb_size"], 
        vocab_len=len(lang.word2id), 
        pad_index=pad_idx
    ).to(DEVICE)
    '''

    
    model = ModelIAS_Dropout(
        hid_size=config["hid_size"], 
        out_slot=len(lang.slot2id), 
        out_int=len(lang.intent2id), 
        emb_size=config["emb_size"], 
        vocab_len=len(lang.word2id), 
        pad_index=pad_idx,
        emb_dropout=config["emb_dropout"],
        out_dropout=config["out_dropout"]
    ).to(DEVICE)
    
    
    # Initialize model weights depending on the layer type
    model.apply(init_weights)

    # Setup optimizer and loss functions
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        print("Using Adam optimizer")
    elif config["optimizer"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
        print("Using AdamW optimizer")

    # With slots, we could have padding tokens, so we ignore them in the loss computation
    criterion_slots = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # For intents, no need to ignore anything (there are no padding tokens)
    criterion_intents = nn.CrossEntropyLoss()

    # Set variables to store and update the best model
    best_loss = float('inf')
    best_model = None
    best_epoch = 0

    # Lists for plotting training progress
    train_losses = []
    dev_losses = []

    # Setup of the patience parameter
    # If the dev loss does not improve for a certain number of epochs, we stop training early
    patience = config["patience"]
    stopped_early = False
    epochs_ran = 0

    # Training loop
    for epoch in range(1, config["n_epochs"] + 1):
        # clip is the max norm for gradient clipping, used to avoid exploding gradients
        train_loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=config["clip"])
        
        # Compute F1 on dev set
        results_dev, intent_res, dev_loss = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
        f1_dev = results_dev['total']['f']
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | Dev F1: {f1_dev:.4f} | Dev Intent Acc: {intent_res['accuracy']:.4f}")

        # Check if this is the best model so far and save it
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            # Reset patience
            patience = config["patience"]
        else:
            patience -= 1
            if patience == 0:
                print(f"No improvement for {config['patience']} epochs. Early stopping at epoch {epoch}.")
                stopped_early = True
                epochs_ran = epoch
                break
        
        # Store train and dev losses
        train_losses.append(train_loss)
        dev_losses.append(dev_loss)

    # Once training is done, evaluate on the test set using the best model
    model.load_state_dict(best_model)
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
    print(f"\nBest Dev Model Results:")
    print(f"Test F1: {results_test['total']['f']:.4f} | Test Intent Acc: {intent_test['accuracy']:.4f}")

    # Create results dictionary to log
    results = {
        "config": config,
        "results": {
            "best_dev_loss": round(best_loss, 4),
            "test_f1": round(results_test['total']['f'], 4),
            "test_intent_acc": round(intent_test['accuracy'], 4),
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "epochs_ran": epochs_ran if stopped_early else config["n_epochs"]
        }
    }

    log_filename = "experiments_results.json"

    # Load existing logs if the file exists, otherwise create a new one
    if os.path.exists(log_filename):
        with open(log_filename, "r") as f:
            all_logs = json.load(f)
    else:
        all_logs = []

    all_logs.append(results)

    with open(log_filename, "w") as f:
        json.dump(all_logs, f, indent=4)


    # Save the best model weights in the bin/ directory
    os.makedirs("bin", exist_ok=True)
    model_filename = f"bin/{config['experiment_name']}_F1-{round(results_test['total']['f'], 4)}.pt"

    saving_object = {
        "epoch": best_epoch,
        "model": best_model,
        "optimizer": optimizer.state_dict(),
        "w2id": lang.word2id,
        "s2id": lang.slot2id,
        "i2id": lang.intent2id
    }

    torch.save(saving_object, model_filename)
    print(f"Log saved in {log_filename} and model saved in bin/")

    # Plot train and dev losses and save alongside the model
    sampled_epochs = list(range(1, len(train_losses) + 1))
    fig = plt.figure(num=3, figsize=(8, 5))
    fig.patch.set_facecolor('white')
    plt.title(f"Training and Dev Losses - {config['experiment_name']}")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(sampled_epochs, train_losses, label='Train loss')
    plt.plot(sampled_epochs, dev_losses, label='Dev loss')
    plt.legend()
    fig_filename = f"bin/{config['experiment_name']}_{round(results_test['total']['f'], 4)}_loss_plot.png"
    plt.savefig(fig_filename, bbox_inches='tight')
    plt.close(fig)