# Natural Language Understanding - Part A

In this part, you apply incremental modifications to the baseline Model IAS (Intent and Slot filling model) to improve performance on intent classification and slot filling tasks. Modifications are applied incrementally, and unsuccessful experiments are documented. Performance metrics include **Accuracy** for intent classification and **F1 score** for slot filling (evaluated using conll format).

One of the important tasks of training a neural network is hyperparameter optimization. Thus, hyperparameters are tuned to improve accuracy and F1 scores, with particular attention to **the learning rate**.

### Mandatory Requirements

All experiments must use the **ATIS dataset**. The baseline and all subsequent models must achieve meaningful improvements over the initial baseline:
- **Intent Classification**: Measure accuracy
- **Slot Filling**: Measure F1 score using conll script

### Modifications Applied

The following modifications are applied incrementally to the baseline Model IAS:

1. **Bidirectionality** - Convert unidirectional RNNs to bidirectional
2. **Dropout Layer** - Add dropout for regularization

## Repository Structure

### Files

- **`model.py`**: Contains the neural network architecture implementations
- **`functions.py`**: Training and evaluation functions
- **`utils.py`**: Utility functions for data loading and preprocessing
- **`conll.py`**: Script for evaluating slot filling with conll format
- **`main.py`**: Main script for running experiments
- **`experiments_results.json`**: Complete log of all experiments with configurations and results

### Folders

- **`bin/`**: Contains the best performing model for each experimental step

- **`bin_others/`**: Contains all other models trained during experimentation and hyperparameter tuning

- **`dataset/`**: ATIS dataset files
  - Training set
  - Validation set
  - Test set

- **`plots/`**: Visualization of training results and comparisons

## Experiments Results

The `experiments_results.json` file contains configurations of each model with their results in the exact order experiments were conducted. This allows for understanding the reasoning behind parameter changes and the incremental improvement process.

Each entry includes:
- Model configuration (architecture, hyperparameters)
- Training details (optimizer, learning rate, etc.)
- Performance metrics (training accuracy, validation accuracy, test accuracy, training F1, validation F1, test F1)

## Usage

To run an experiment:

```bash
python main.py
```

### Important Configuration Steps

Before running `main.py`, you must update the configuration in the script:

1. **Update the `config` dictionary** at the top of `main.py`:
   - Set `"experiment_name"` to the name of your experiment (this will be the name of the saved model file)
   - Adjust hyperparameters such as `lr` (learning rate), `n_epochs`, `dropout`, etc.
   - The experiment name will be used to save the model checkpoint

2. **Select the model to use**:
   - In `main.py`, find the model selection section
   - **Comment out** the model you don't want to use
   - **Uncomment** the model you want to train
   - Available models:
     - `Model_IAS` - Baseline intent and slot filling model
     - `Model_IAS_Bidirectional` - Model IAS with bidirectional RNNs
     - `Model_IAS_Bidirectional_Dropout` - Model IAS with bidirectionality and dropout

3. **Ensure the experiment name matches the model type** for consistency in tracking results

All parameters are defined in the `config` dictionary within `main.py`, so no command-line arguments are needed.

## Results Summary

| Step | Model | Intent Accuracy | Slot F1 |
|------|-------|-----------------|---------|
| 1    | Baseline Model IAS | 0.9261 | 0.9328 |
| 2    | Model IAS + Bidirectional | 0.9485 | 0.9461 |
| 3    | Model IAS + Bidirectional + Dropout | 0.9619 | 0.9495 |
| 4    | Model IAS + Bidirectional + Dropout (AdamW) | 0.9597 | 0.9510 |

All models successfully achieve strong performance on the ATIS dataset, with the best test F1 score of **0.9510** and best intent accuracy of **0.9619**.