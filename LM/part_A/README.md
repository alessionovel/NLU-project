# Language Model - Part A

In this part, the baseline LM_RNN is modified by adding a set of techniques that might improve the performance. Modifications are added one at a time incrementally. If adding a modification decreases the performance, it can be removed and we move forward with the others. However, unsuccessful experiments are documented and commented on in the report. For each experiment, the performance is expressed with Perplexity (PPL).

One of the important tasks of training a neural network is hyperparameter optimization. Thus, hyperparameters were tuned to minimize the PPL and results achieved with the best configuration are reported (in particular **the learning rate**).

### Mandatory Requirements

For all experiments, the perplexity must be below 250 (***PPL < 250***).

### Experiments Conducted

1. **Replace RNN with a Long-Short Term Memory (LSTM) network**
2. **Add two dropout layers**:
   - One after the embedding layer
   - One before the last linear layer
3. **Replace SGD with AdamW optimizer**

## Repository Structure

### Files

- **`model.py`**: Contains the neural network architecture implementations
- **`functions.py`**: Training and evaluation functions
- **`utils.py`**: Utility functions for data loading and preprocessing
- **`main.py`**: Main script for running experiments
- **`experiments_results.json`**: Complete log of all experiments with configurations and results

### Folders

- **`bin/`**: Contains the best performing model for each experimental step:
  - `Baseline_RNN_PP-155.03.pt` - Baseline RNN model
  - `Baseline_LSTM_PP-138.34.pt` - LSTM replacement
  - `Dropout_LSTM_PP-110.08.pt` - LSTM with dropout layers
  - `AdamW_LSTM_PP-104.10.pt` - Final model with AdamW optimizer

- **`bin_others/`**: Contains all other models trained during experimentation and hyperparameter tuning

- **`dataset/`**: Penn Treebank dataset files
  - `ptb.train.txt` - Training set
  - `ptb.valid.txt` - Validation set
  - `ptb.test.txt` - Test set

- **`plots/`**: Visualization of training results and comparisons

## Experiments Results

The `experiments_results.json` file contains configurations of each model with their results in the exact order experiments were conducted. This allows for understanding the reasoning behind parameter changes and the incremental improvement process.

Each entry includes:
- Model configuration (architecture, hyperparameters)
- Training details (optimizer, learning rate, etc.)
- Performance metrics (training PPL, validation PPL, test PPL)

## Usage

To run an experiment:

```bash
python main.py
```

### Important Configuration Steps

Before running `main.py`, you must update the configuration in the script:

1. **Update the `config` dictionary** at the top of `main.py`:
   - Set `"experiment_name"` to the name of your experiment (this will be the name of the saved model file)
   - Adjust hyperparameters such as `lr` (learning rate), `n_epochs`, `emb_dropout`, `out_dropout`, etc.
   - The experiment name will be used to save the model checkpoint

2. **Select the model to use**:
   - In `main.py`, find the model selection section (around line 76-91)
   - **Comment out** the model you don't want to use
   - **Uncomment** the model you want to train
   - Available models:
     - `LM_RNN` - Baseline RNN model
     - `LM_LSTM` - LSTM-based model
     - `LM_LSTM_dropout` - LSTM with dropout regularization

3. **Ensure the experiment name matches the model type** for consistency in tracking results

All parameters are defined in the `config` dictionary within `main.py`, so no command-line arguments are needed.

## Results Summary

| Step | Model | Best Test PPL |
|------|-------|---------------|
| 1    | Baseline RNN | 155.03 |
| 2    | LSTM | 138.34 |
| 3    | LSTM + Dropout | 110.08 |
| 4    | LSTM + Dropout + AdamW | 104.10 |

All models successfully meet the requirement of PPL < 250, with the final model achieving a test perplexity of **104.10**.