# Language Model - Part B

In this part, starting from the baseline LSTM model from Part A, a set of advanced regularization techniques are applied to further improve the performance. Modifications are added one at a time incrementally. If adding a modification decreases the performance, it can be removed and we move forward with the others. However, unsuccessful experiments are documented and commented on in the report. For each experiment, the performance is expressed with Perplexity (PPL).

One of the important tasks of training a neural network is hyperparameter optimization. Thus, hyperparameters were tuned to minimize the PPL and results achieved with the best configuration are reported (in particular **the learning rate**).

### Mandatory Requirements

For all experiments, the perplexity must be below 250 (***PPL < 250***) and it should be lower than the one achieved in Part A (i.e., baseline LSTM with PPL of 138.34).

### Experiments Conducted

The following regularization techniques are applied:

1. **Weight Tying**
2. **Variational Dropout** (without DropConnect)
3. **Non-monotonically Triggered Average SGD (NT-AvSGD)**

## Repository Structure

### Files

- **`model.py`**: Contains the neural network architecture implementations
- **`functions.py`**: Training and evaluation functions
- **`utils.py`**: Utility functions for data loading and preprocessing
- **`main.py`**: Main script for running experiments
- **`experiments_results.json`**: Complete log of all experiments with configurations and results

### Folders

- **`bin/`**: Contains the best performing model for each experimental step:
  - `Baseline_LSTM_PP-134.49.pt` - Baseline LSTM model (from Part A)
  - `WeightTying_LSTM_PP-121.6.pt` - LSTM with Weight Tying
  - `VariationalDropout_LSTM_PP-104.76.pt` - LSTM with Variational Dropout
  - `NT-AvSGD_LSTM_PP-98.61.pt` - Final model with NT-AvSGD optimizer

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
     - `LM_LSTM` - Baseline LSTM model
     - `LM_LSTM_WeightTying` - LSTM with Weight Tying
     - `LM_LSTM_VariationalDropout` - LSTM with Variational Dropout
     - `LM_LSTM_AvSGD` - LSTM with Non-monotonically Triggered Average SGD

3. **Ensure the experiment name matches the model type** for consistency in tracking results

All parameters are defined in the `config` dictionary within `main.py`, so no command-line arguments are needed.

## Results Summary

| Step | Model | Best Test PPL |
|------|-------|---------------|
| 1    | Baseline LSTM (from Part A) | 134.49 |
| 2    | LSTM + Weight Tying | 121.6 |
| 3    | LSTM + Variational Dropout | 104.76 |
| 4    | LSTM + NT-AvSGD | 98.61 |

All models successfully meet the requirement of PPL < 250 and are lower than the baseline LSTM from Part A, with the final model achieving a test perplexity of **98.61**.