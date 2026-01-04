import torch
import torch.nn as nn
from transformers import BertModel

class JointBERT(nn.Module):
    def __init__(self, out_slot, out_int, model_name='bert-base-uncased', dropout=0.1):
        super(JointBERT, self).__init__()

        # Embedding layer not needed as BERT includes its own embeddings
        
        # Load pre-trained BERT model (from Hugging Face Transformers)
        self.bert = BertModel.from_pretrained(model_name)
        
        # Get the hidden size of the chosen BERT model
        self.hidden_size = self.bert.config.hidden_size
        
        # Set the output layer of slot filling (from BERT hidden size to number of slot classes)
        self.slot_out = nn.Linear(self.hidden_size, out_slot)
        
        # Set the output layer of intent detection (from BERT hidden size to number of intent classes)
        self.intent_out = nn.Linear(self.hidden_size, out_int)

        # Set the dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, utt_emb, attention_mask):
        # Compute BERT outputs
        # Attention mask is used to avoid computation on padding tokens
        # So we don't need to use pack_padded_sequence or similar functions
        outputs = self.bert(input_ids=utt_emb, attention_mask=attention_mask)
        
        # Get the output tokens of the sequence
        sequence_output = outputs.last_hidden_state
        
        # Get the output token of the [CLS] token
        pooled_output = outputs.pooler_output
        
        # Apply dropout to both outputs
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        # Apply the linear layers to get logits
        slots_logits = self.slot_out(sequence_output)
        intent_logits = self.intent_out(pooled_output)
        
        # Permute to have (batch, vocab size, seq) as output shape (required by nn.CrossEntropyLoss)
        slots_logits = slots_logits.permute(0, 2, 1)
        
        return slots_logits, intent_logits