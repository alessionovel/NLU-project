import torch
import torch.optim as optim
import torch.nn as nn
import json
import os
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from functools import partial
import math

# Import defined models and functions
from model import LM_LSTM, LM_RNN, LM_LSTM_dropout
from utils import read_file, Lang, PennTreeBank, collate_fn
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
    "experiment_name": "AdamW_LSTM",
    "model_type": "LSTM",
    "optimizer": "AdamW",
    "emb_size": 500,
    "hidden_size": 800,
    "lr": 0.0005,
    "patience": 10,
    "batch_size": 64,
    "clip": 5,
    "n_epochs": 100,
    "emb_dropout": 0.5,
    "out_dropout": 0.6
}

if __name__ == "__main__":

    # Print configuration
    print("Experiment Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    # Data loading
    train_raw = read_file("dataset/ptb.train.txt")
    dev_raw = read_file("dataset/ptb.valid.txt")
    test_raw = read_file("dataset/ptb.test.txt")
    
    # Create vocabulary from the training set
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    # Create datasets (in PyTorch Dataset format)
    train_ds = PennTreeBank(train_raw, lang)
    dev_ds = PennTreeBank(dev_raw, lang)
    test_ds = PennTreeBank(test_raw, lang)

    # Create dataLoaders
    # Use collate_fn to pad sequences in the batch and hardcode the pad token index
    pad_idx = lang.word2id["<pad>"]
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=partial(collate_fn, pad_token=pad_idx, device=DEVICE))

    # In evaluation, we can use larger batches since we don't need to store gradients (so less memory usage)
    # The batch size of the evaluation does not affect the results
    dev_loader = DataLoader(dev_ds, batch_size=128, collate_fn=partial(collate_fn, pad_token=pad_idx, device=DEVICE))
    test_loader = DataLoader(test_ds, batch_size=128, collate_fn=partial(collate_fn, pad_token=pad_idx, device=DEVICE))

    # Initialize the model with the various parameters
    vocab_len = len(lang.word2id)

    # Select the model
    # Remember to change also the experiment_name in order to log and save the model correctly
    
    '''
    model = LM_RNN(emb_size=config["emb_size"], 
                   hidden_size=config["hidden_size"], 
                   output_size=vocab_len, 
                   pad_index=pad_idx).to(DEVICE)
    '''
    
    '''
    model = LM_LSTM(emb_size=config["emb_size"], 
                   hidden_size=config["hidden_size"], 
                   output_size=vocab_len, 
                   pad_index=pad_idx).to(DEVICE)
    '''

    
    model = LM_LSTM_dropout(emb_size=config["emb_size"], 
                   hidden_size=config["hidden_size"], 
                   output_size=vocab_len, 
                   pad_index=pad_idx,
                   emb_dropout=config["emb_dropout"],
                   out_dropout=config["out_dropout"]).to(DEVICE)
    
    
    # Initialize model weights depending on the layer type
    model.apply(init_weights)

    # Setup optimizer and loss functions
    if config["optimizer"] == "SGD":
        print("Using SGD optimizer")
        optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "AdamW":
        print("Using AdamW optimizer")
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

    # In training, use the default mean reduction (in order to make the loss length-independent)
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_idx)
    # In evaluation, use sum reduction since we don't have to update weights but only need the total loss
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')

    # Set variables to store and update the best model
    best_ppl = float('inf')
    best_model = None
    best_epoch = 0

    # Lists for plotting training progress
    ppls_train = []
    ppls_dev = []

    # Setup of the patience parameter
    # If the dev perplexity does not improve for a certain number of epochs, we stop training early
    patience = config["patience"]
    stopped_early = False
    epochs_ran = 0

    # Training loop
    for epoch in range(1, config["n_epochs"] + 1):
        # clip is the max norm for gradient clipping, used to avoid exploding gradients
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip=config["clip"])

        # Compute perplexity on the dev set
        ppl_dev = eval_loop(dev_loader, criterion_eval, model)
        
        print(f"Epoch {epoch} | Train Loss: {loss:.4f} | Dev PPL: {ppl_dev:.2f}")
        
        # Check if this is the best model so far and save it
        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
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
        
        # Store train and dev perplexities
        # For train we have to compute it from the loss
        ppls_train.append(math.exp(loss))
        ppls_dev.append(ppl_dev)


    # Once training is done, evaluate on the test set using the best model
    model.load_state_dict(best_model)
    final_test_ppl = eval_loop(test_loader, criterion_eval, model)
    print(f"\nBest Dev PPL: {best_ppl:.2f} at epoch {best_epoch}")
    print(f"Final Test PPL: {final_test_ppl:.2f}")

    # Create results dictionary to log
    results = {
        "config": config,
        "results": {
            "best_dev_ppl": round(best_ppl, 2),
            "test_ppl": round(final_test_ppl, 2),
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
    model_filename = f"bin/{config['experiment_name']}_PP-{round(final_test_ppl, 2)}.pt"
    
    saving_object = {
        "epoch": best_epoch,
        "model": best_model,
        "optimizer": optimizer.state_dict(),
        "w2id": lang.word2id
    }
    
    torch.save(saving_object, model_filename)
    print(f"Log saved in {log_filename} and model saved in bin/")

    # Plot train and dev perplexities and save alongside the model
    sampled_epochs = list(range(1, len(ppls_train) + 1))
    fig = plt.figure(num=3, figsize=(8, 5))
    fig.patch.set_facecolor('white')
    plt.title(f"Train and Dev Perplexities - {config['experiment_name']}")
    plt.ylabel('Perplexity')
    plt.xlabel('Epochs')
    plt.plot(sampled_epochs, ppls_train, label='Train perplexity')
    plt.plot(sampled_epochs, ppls_dev, label='Dev perplexity')
    plt.legend()
    fig_filename = f"bin/{config['experiment_name']}_{round(final_test_ppl, 2)}_perplexities_plot.png"
    plt.savefig(fig_filename, bbox_inches='tight')
    plt.close(fig)