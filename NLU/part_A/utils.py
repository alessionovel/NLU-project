import json
import torch
import torch.utils.data as data
from collections import Counter

# Special token IDs
PAD_TOKEN = 0

def load_data(path):
    '''Load the dataset json file'''
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

# Vocabulary class mapping words, intents, and slots to unique IDs (and vice versa)
# Vocabulary is created from the provided lists, with optional cutoff for words based on frequency
# Special tokens are added first
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
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
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
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

# PyTorch Dataset class for the Intents and Slots dataset
# This class prepares the data for the task
class IntentsAndSlots(data.Dataset):
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    # Function used by DataLoader to get a sample from the dataset
    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        return {'utterance': utt, 'slots': slots, 'intent': intent}
    
    # Function that maps intents to IDs using the given vocabulary (Lang object)
    def mapping_lab(self, data, mapper):
        # If an intent is not in the vocabulary, map it to 'unk'
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    # Function that maps words and labels to IDs using the given vocabulary (Lang object)
    def mapping_seq(self, data, mapper):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                # If a word/label is not in the vocabulary, map it to 'unk'
                tmp_seq.append(mapper[x] if x in mapper else mapper[self.unk])
            res.append(tmp_seq)
        return res

# Collate function to be used in DataLoader for padding sequences in a batch
# This is necessary since sequences can have different lengths, with this function we add padding tokens to make them the same length
def collate_fn(data, device):
    # We sort the sentences in the batch by length in descending order
    # This is done because models are optimized for processing sequences of similar lengths together
    data.sort(key=lambda x: len(x['utterance']), reverse=True)

    # Create a dictionary where each key maps to a list of corresponding items from all samples in the batch
    new_item = {key: [d[key] for d in data] for key in data[0].keys()}
    
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        # We take the maximum sentence length of the batch
        max_len = max(lengths) if max(lengths) > 0 else 1
        # Create a tensor with size (batch_size, max_len) filled with the pad_token
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        # Fill the tensor with the actual sequences (up to their length). The rest remains as pad_token
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        return padded_seqs.detach(), torch.LongTensor(lengths)

    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    return {
        "utterances": src_utt.to(device),
        "intents": intent.to(device),
        "y_slots": y_slots.to(device),
        "slots_len": y_lengths.to(device)
    }