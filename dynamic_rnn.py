import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn


class DynamicLstm(nn.Module):
    def __init__(self, vocab_size, embded_size, lstm_hidden):
        super(DynamicLstm, self).__init__()
        self.embed = nn.Embedding(vocab_size, embded_size)
        self.lstm = nn.LSTM(embded_size, lstm_hidden, batch_first=True)

    def forward(self, ids_list):
        len_txt = torch.tensor([len(i) for i in ids_list])
        lens, ids = len_txt.sort(descending=True)
        un_ids = ids.sort()[1]

        pad_ids = pad_sequence(ids_list, batch_first=True)
        pad_ids = pad_ids[ids]

        embed_out = self.embed(pad_ids)
        packed_out = pack_padded_sequence(embed_out, lens, batch_first=True)
        out, (ht, ct) = self.lstm(packed_out)

        pad_back_out = pad_packed_sequence(out, batch_first=True)[0]
        pad_back_out = torch.index_select(pad_back_out, 0, un_ids)
        return pad_back_out, (ht, ct)
