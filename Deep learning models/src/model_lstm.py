import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import init


class FakeNewsDetect(nn.Module):
    def __init__(self, args, embeddings=None):
        super(FakeNewsDetect, self).__init__()
        
        # Create learnable embedding
        if embeddings == None:
            self.embed = nn.Embedding(args.vocab_size, args.embed_dim, padding_idx=0)
        else:
            self.embed = nn.Embedding.from_pretrained(embeddings, freeze=False)
        
        self.lstm = nn.LSTM(args.embed_dim, args.hidden_dim, num_layers=args.nums_layer, batch_first=True,
                            bidirectional=True)
        
        # Attention learning parameters
        self.attention = nn.Parameter(torch.randn(1, args.hidden_dim))
        self.fc = nn.Linear(args.hidden_dim, 2)
        
        self.embed_drop = nn.Dropout(args.embed_dropout)
        self.lstm_drop = nn.Dropout(args.lstm_dropout)
        self.fc_drop = nn.Dropout(args.fc_dropout)
        self.args = args

        init.xavier_normal_(self.fc.weight)
        init.constant_(self.fc.bias, 0.)

    def attention_layer(self, x, mask):
        x = F.tanh(x)  # [b, s, d]
        att_score = torch.matmul(x, self.attention.unsqueeze(-1))  # [b, s, 1]
        mask = mask.unsqueeze(dim=-1)  # [b*s*1]
        att_score = att_score.masked_fill(mask.eq(0), float('-inf'))  # [b*s*1]
        att_score = F.softmax(att_score, dim=1)  # b*s*1
        r = (x * att_score).sum(dim=1)  # [b, d]
        return F.tanh(r)

    def forward(self, src, mask):
        length = mask.sum(1)
        input_embedding = self.embed(src)
        
        if self.args.model_name == 'no_drop_out':
            pass
        else:
            input_embedding = self.embed_drop(input_embedding)
        
        packed_input = pack_padded_sequence(input_embedding, length.data.tolist(), batch_first=True,
                                            enforce_sorted=False)
        out, (_, _) = self.lstm(packed_input)  # [batch, s, 2*hidden_dim]
        
        paded_out = pad_packed_sequence(out, batch_first=True, total_length=mask.size(1))[0]
        # Copy the original tensor, keep the memory address of the tensor element continuous in the memory space
        paded_out = paded_out.contiguous()
        b, s, _ = paded_out.size()
        # print("b,s =", b, s)
        # print("padedOut=",paded_out, type(paded_out))
        # b: batch_size, s: sequence_length
        paded_out = paded_out.view(b, s, 2, -1)  # [b, s, 2, hidden_dim]，Two-way lstm, each step has two hidden_dim
        # print("padedOutView=", paded_out, type(paded_out))
        out = paded_out.sum(dim=2)  # [b, s, hidden_dim]  # Combine bi-directional features
        
        if self.args.model_name == 'no_attention':
            sent_vec = out[:, 0, :]   # [b, hidden_dim]
        else:
            sent_vec = self.attention_layer(out, mask)  # [b, hidden_dim]
            
        if self.args.model_name == 'no_drop_out':
            pass
        else:
            sent_vec = self.fc_drop(sent_vec)
        logits = self.fc(sent_vec)
        return logits
