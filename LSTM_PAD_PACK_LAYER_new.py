import torch as t
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

a = t.tensor([[1,2,3],[6,0,0],[4,5,0]]) #(batch_size, max_length)
lengths = t.tensor([3,1,2])

# 排序
a_lengths, idx = lengths.sort(0, descending=True)
_, un_idx = t.sort(idx, dim=0)
a = a[idx]

# 定义层
emb = t.nn.Embedding(20,2,padding_idx=0)
lstm = t.nn.LSTM(input_size=2, hidden_size=4, batch_first=True)

a_input = emb(a)
a_packed_input = t.nn.utils.rnn.pack_padded_sequence(input=a_input, lengths=a_lengths, batch_first=True)
packed_out, _ = lstm(a_packed_input)
out, _ = pad_packed_sequence(packed_out)
# 根据un_idx将输出转回原输入顺序
out = t.index_select(out.permute(1,0,2), 0, un_idx)

