import json
import torch
import torch.utils.data as data
from collections import Counter
from transformers import BertTokenizer

# Here we don't need to define PAD_TOKEN since we use BERT tokenizer
# PAD_TOKEN = 0

def load_data(path):
    '''Load the dataset json file'''
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

# Vocabulary class mapping words, intents, and slots to unique IDs (and vice versa)
# Vocabulary is created from the provided lists, with optional cutoff for words based on frequency
# Special tokens are added first
# In this case, word2id is not used for model input as we use BERT tokenizer
class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        
        # Reverse mapping (from IDs to original words, slots, intents)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    # Function to create word vocabulary
    # In this case, it won't be used because we use BERT tokenizer
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': 0}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    # Function to create label vocabulary (for intents and slots)
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = 0 
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

# PyTorch Dataset class for the Intents and Slots dataset
# This class prepares the data for the task
# Tokenization and padding are done with BERT tokenizer
# We use max_len as the paper
class BertIntentsAndSlots(data.Dataset):
    def __init__(self, dataset, lang, tokenizer, max_len=50):
        self.dataset = dataset
        self.lang = lang
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Set ignore_index for slot labels (to ignore padding and non-first sub-tokens)
        # This tokens will be ignored in training and evaluation
        self.ignore_index = -100

    def __len__(self):
        return len(self.dataset)

    # Function used by DataLoader to get a sample from the dataset
    # In this case, embedding and padding are handled here
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get the utterance, slots, and intent for the current item
        utterance = item['utterance'].split()
        slots = item['slots'].split()
        intent = item['intent']

        # Lists for tokens and corresponding slot IDs
        bert_tokens = []
        slot_ids = []

        # Add the special [CLS] token at the beginning
        bert_tokens.append("[CLS]")
        # We don't need to assign slot label to it
        # This token will be used for classification task only
        slot_ids.append(self.ignore_index)

        # Iterate over each word and its corresponding slot label
        for word, slot_label in zip(utterance, slots):
            # Tokenize the word with BERT tokenizer
            sub_tokens = self.tokenizer.tokenize(word)
            
            # Handle the case where a word might be tokenized into no sub-tokens
            if not sub_tokens:
                sub_tokens = ['[UNK]']

            # Add the sub-tokens to the list
            bert_tokens.extend(sub_tokens)

            # The first sub-token gets the original slot label
            if slot_label in self.lang.slot2id:
                slot_ids.append(self.lang.slot2id[slot_label])
            else:
                # Handle the case where the slot label is not in the vocabulary (not possible in this case)
                slot_ids.append(self.ignore_index)

            # Not first sub-tokens get the ignore_index
            for _ in range(len(sub_tokens) - 1):
                slot_ids.append(self.ignore_index)

        # Add the special [SEP] token at the end of the sequence
        bert_tokens.append("[SEP]")
        slot_ids.append(self.ignore_index)

        # Truncate if longer than max_len
        # This is needed because we need all the sequences to be of the same length for batching
        # The paper states that for ATIS dataset, max_len of 50 is okay because all the utterances are shorter than that
        if len(bert_tokens) > self.max_len:
            bert_tokens = bert_tokens[:self.max_len]
            slot_ids = slot_ids[:self.max_len]

        # Convert tokens to input IDs (using BERT's vocab)
        input_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)

        # Create the attention mask (to not consider padding tokens and not first sub-tokens)
        attention_mask = [1] * len(input_ids)

        # Padding if shorter than max_len
        padding_len = self.max_len - len(input_ids)
        
        # Fill the padding
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
        attention_mask = attention_mask + [0] * padding_len

        # For slot IDs, pad with ignore_index
        slot_ids = slot_ids + [self.ignore_index] * padding_len

        # Convert to tensors
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        slot_ids_tensor = torch.tensor(slot_ids, dtype=torch.long)
        
        # Handle intent
        intent_id = self.lang.intent2id.get(intent, -1)
        intent_tensor = torch.tensor(intent_id, dtype=torch.long)

        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'slots': slot_ids_tensor,
            'intent': intent_tensor
        }
    
# In this case, we don't need to define a collate function since the padding is already handled in the Dataset class