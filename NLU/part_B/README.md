# Natural Language Understanding - Part B

In this part, you apply fine-tuning to a pre-trained BERT model using a multi-task learning setting for intent classification and slot filling tasks. The main challenge addressed in this implementation is handling the **sub-tokenization issue** that arises when using BERT's WordPiece tokenizer.

The implementation demonstrates effective multi-task learning for joint intent and slot filling. Hyperparameters are tuned to optimize both intent classification accuracy and slot filling F1 score.

### Mandatory Requirements

All experiments must use the **ATIS dataset**. The models are built using pre-trained BERT:
- **BERT-base** or **BERT-large**
- **Intent Classification**: Measured by accuracy
- **Slot Filling**: Measured by F1 score using conll script
- **Multi-task Learning**: Joint training on both intent classification and slot filling tasks

### Models Tested

The following BERT-based models were experimented with:

1. **BERT-base** - Fine-tuned pre-trained BERT-base-uncased model

## Repository Structure

### Files

- **`model.py`**: Contains the BERT-based neural network architecture implementations for multi-task learning
- **`functions.py`**: Training and evaluation functions for intent and slot filling tasks
- **`utils.py`**: Utility functions for data loading, preprocessing, and handling sub-tokenization
- **`conll.py`**: Script for evaluating slot filling with conll format
- **`main.py`**: Main script for running fine-tuning experiments
- **`experiments_results.json`**: Complete log of all experiments with configurations and results

### Folders

- **`bin/`**: Contains the best performing model for each experimental configuration

- **`bin_others/`**: Contains additional models trained during experimentation and hyperparameter tuning. **Note**: Not all models attempted are stored here due to file size constraints; only the most relevant checkpoints are maintained.

- **`dataset/`**: ATIS dataset files
  - Training set
  - Validation set
  - Test set

- **`plots/`**: Visualization of training results and comparisons

## Key Implementation Details

### Sub-tokenization Handling

BERT uses WordPiece tokenization which can split words into multiple tokens (sub-tokens). For slot filling, this requires special handling to ensure labels align correctly with sub-tokens. The implementation addresses this by:
- Mapping word-level slot labels to sub-token representations
- Using the first token representation for sub-word tokens during prediction
- Ensuring evaluation metrics correctly handle this mapping

## Experiments Results

The `experiments_results.json` file contains configurations of each model with their results in the exact order experiments were conducted.

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
   - Select BERT model size: `"bert-base-uncased"` or `"bert-large-uncased"`
   - Adjust hyperparameters such as `lr` (learning rate), `n_epochs`, `batch_size`, etc.
   - The experiment name will be used to save the model checkpoint

2. **Select the model to use**:
   - In `main.py`, find the model selection section
   - **Comment out** the model you don't want to use
   - **Uncomment** the model you want to train
   - Available models:
     - `BertIAS` - Base BERT-based intent and slot filling model
     - Other variants as implemented for the multi-task learning approach

3. **Ensure the experiment name matches the model configuration** for consistency in tracking results

All parameters are defined in the `config` dictionary within `main.py`, so no command-line arguments are needed.

## Results Summary

Results from fine-tuning BERT models on the ATIS dataset:

| Configuration | Learning Rate | Dropout | Test F1 | Test Intent Accuracy |
|---|---|---|---|---|
| BERT Adam | 5×10⁻⁵ | 0.6 | **0.9580** | **0.9776** |
| BERT AdamW | 2×10⁻⁵ | 0.6 | 0.9548 | **0.9776** |

The BERT Adam configuration achieves the best test F1 score of **0.9580**, while both configurations achieve the same test intent accuracy of **0.9776**.