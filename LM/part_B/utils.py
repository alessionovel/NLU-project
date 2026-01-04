import torch
from torch.utils.data import Dataset

# Function to read the file and append an end-of-sentence token to each line
def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            # .strip() removes starting and ending whitespaces
            # Append the end-of-sentence token at the end of each line
            output.append(line.strip() + " " + eos_token)
    return output

# Vocabulary class mapping from words to unique IDs (and vice versa)
# Vocabulary is created from the provided corpus, with optional special tokens added first (like <pad> and <eos>)
class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        # Reverse mapping (from IDs to original words)
        self.id2word = {v:k for k, v in self.word2id.items()}
    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

# PyTorch Dataset class for the Penn Tree Bank dataset
# This class prepares the data for the task
class PennTreeBank(Dataset):
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []
        # For each word, the model has to predict the next one, so the target of a word is the following word in the sequence
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1])
            self.target.append(sentence.split()[1:])
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    # Function used by DataLoader to get a sample from the dataset
    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        return {'source': src, 'target': trg}
    
    # Function that maps words to IDs using the given vocabulary (Lang object)
    def mapping_seq(self, data, lang):
        res = []
        for seq in data:
            # If a word is not in the vocabulary, it is skipped
            # PennTreeBank dataset does not have unknown words, so this is not an issue here
            tmp_seq = [lang.word2id[x] for x in seq if x in lang.word2id]
            res.append(tmp_seq)
        return res

# Collate function to be used in DataLoader for padding sequences in a batch
# This is necessary since sequences can have different lengths, with this function we add padding tokens to make them the same length
def collate_fn(data, pad_token, device):
    # We sort the sentences in the batch by length in descending order
    # This is done because models are optimized for processing sequences of similar lengths together
    data.sort(key=lambda x: len(x["source"]), reverse=True)

    # Create a dictionary where each key maps to a list of corresponding items from all samples in the batch
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    sequences = new_item["source"]
    lengths = [len(seq) for seq in sequences]
    # We take the maximum sentence length of the batch
    max_len = max(lengths) if lengths else 1
    # Create a tensor with size (batch_size, max_len) filled with the pad_token
    padded_src = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
    # Fill the tensor with the actual sequences (up to their length). The rest remains as pad_token
    for i, seq in enumerate(sequences):
        padded_src[i, :len(seq)] = seq

    # Same for target sequences
    sequences_trg = new_item["target"]
    padded_trg = torch.LongTensor(len(sequences_trg), max_len).fill_(pad_token)
    for i, seq in enumerate(sequences_trg):
        padded_trg[i, :len(seq)] = seq

    new_item["source"] = padded_src.to(device)
    new_item["target"] = padded_trg.to(device)
    new_item["number_tokens"] = sum(lengths)
    return new_item