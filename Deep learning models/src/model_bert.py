import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import init
from transformers import BertModel


class BertFakeNewsDetect(nn.Module):
    def __init__(self, args):
        super(BertFakeNewsDetect, self).__init__()
        
        # Import bert model
        self.bert = BertModel.from_pretrained(args.bert_path)
        self.lstm = nn.LSTM(args.embed_dim, args.hidden_dim, num_layers=args.nums_layer, batch_first=True,
                            bidirectional=True)
        
        # Attention learning parameters
        self.attention = nn.Parameter(torch.randn(1, args.hidden_dim))
        self.fc = nn.Linear(args.hidden_dim, 2) # fc: fully connected

        self.embed_drop = nn.Dropout(args.embed_dropout)
        self.lstm_drop = nn.Dropout(args.lstm_dropout)
        self.fc_drop = nn.Dropout(args.fc_dropout)
        self.args = args

        init.xavier_normal_(self.fc.weight)
        init.constant_(self.fc.bias, 0.)

    def attention_layer(self, x, attention_mask):
        x = F.tanh(x)  # [b, s, d]
        att_score = torch.matmul(x, self.attention.unsqueeze(-1))  # [b, s, 1]
        attention_mask = attention_mask.unsqueeze(dim=-1)  # [b*s*1]
        att_score = att_score.masked_fill(attention_mask.eq(0), float('-inf'))  # [b*s*1]
        att_score = F.softmax(att_score, dim=1)  # b*s*1
        r = (x * att_score).sum(dim=1)  # [b, d]
        return F.tanh(r)

    def forward(self, input_ids, attention_mask):
        length = attention_mask.sum(1)
        bert_output = self.bert(input_ids, attention_mask)
        seq_out, pooled_out = bert_output[0], bert_output[1]
        if self.args.model_name == 'no_drop_out':
            pass
        else:
            seq_out = self.embed_drop(seq_out)
        
        packed_input = pack_padded_sequence(seq_out, length.data.tolist(), batch_first=True,
                                            enforce_sorted=False)
        out, (_, _) = self.lstm(packed_input)  # [batch, s, 2*hidden_dim]
        
        paded_out = pad_packed_sequence(out, batch_first=True, total_length=attention_mask.size(1))[0]
        paded_out = paded_out.contiguous()
        b, s, _ = paded_out.size()
        paded_out = paded_out.view(b, s, 2, -1)   # [b, s, 2, hidden_dim], two-way lstm, each step has two hidden_dim
        out = paded_out.sum(dim=2)  # [b, s, hidden_dim]  # Combine bidirectional features

        if self.args.model_name == 'no_attention':
            sent_vec = out[:, 0, :]  # [b, hidden_dim]
        else:
            sent_vec = self.attention_layer(out, attention_mask)  # [b, hidden_dim]

        if self.args.model_name == 'no_drop_out':
            pass
        else:
            sent_vec = self.fc_drop(sent_vec)
        logits = self.fc(sent_vec)
        return logits
