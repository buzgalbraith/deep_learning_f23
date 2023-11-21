import torch
import random
import string
class VariableDelayEchoDataset(torch.utils.data.IterableDataset):

  def __init__(self, max_delay=8, seq_length=20, size=1000):
    self.max_delay = max_delay
    self.seq_length = seq_length
    self.size = size

  def __len__(self):
    return self.size

  def __iter__(self):
    for _ in range(self.size):
      seq = torch.tensor([random.choice(range(1, N + 1)) for i in range(self.seq_length)], dtype=torch.int64)
      delay = random.randint(0, self.max_delay)
      result = torch.cat((torch.zeros(delay), seq[:self.seq_length - delay])).type(torch.int64)
      yield seq, delay, result
N = 26
class VariableDelayGRUMemory(torch.nn.Module):

  def __init__(self, hidden_size, max_delay):
    super().__init__()
    self.gru = torch.nn.GRU(N + 1, hidden_size, batch_first=True)
    self.decoder = torch.nn.Linear(hidden_size, N + 1)
  def forward(self, x, delays):
    # inputs:
    # x - tensor of shape (batch size, seq length, N + 1)
    # delays - tensor of shape (batch size)
    # returns:
    # logits (scores for softmax) of shape (batch size, seq_length, N + 1)
    h_t = torch.zeros(1, x.shape[0], self.gru.hidden_size).to(x.device)
    y_hat, _ = self.gru(x, h_t)
    delayed_hidden_states = torch.zeros(x.shape[0], x.shape[1], self.gru.hidden_size).to(x.device) ## (batch size, seq length, hidden size)
    for i in range(x.shape[0]): ## batch size
        delay = delays[i].item()
        for j in range(x.shape[1]): ## seq length
            if j < delay:  ## if j < delay, then we don't have a delayed hidden state
                delayed_hidden_states[i, j] = y_hat[i, j] 
            else: ## if j >= delay, then we have a delayed hidden state
                delayed_hidden_states[i, j] = y_hat[i, j - delay] ## delayed hidden state is the hidden state at j-delay
    logits = self.decoder(delayed_hidden_states) ## (batch size, seq length, N + 1)
    return logits



  @torch.no_grad()
  def test_run(self, s, delay):
    # This function accepts one string s containing lowercase characters a-z,
    # and a delay - the desired output delay.
    # You need to map those characters to one-hot encodings,
    # then get the result from your network, and then convert the output
    # back to a string of the same length, with 0 mapped to ' ',
    # and 1-26 mapped to a-z.
    st = torch.tensor([ord(char) - ord('a') + 1 for char in s], dtype=torch.int64)
    st_tilde = idx_to_onehot(st)
    delay = torch.tensor([delay], dtype=torch.int64)
    logits = self.forward(st_tilde.unsqueeze(0), delay.unsqueeze(0))
    _, predicted_indices = logits.max(dim=-1)
    predicted_chars = ''.join([chr(i.item() + ord('a') - 1) if i.item() > 0 else ' ' for i in predicted_indices.squeeze()])
    return predicted_chars
    
def idx_to_onehot(x, k=N+1):
  """ Converts the generated integers to one-hot vectors """
  ones = torch.sparse.torch.eye(k)
  shape = x.shape
  res = ones.index_select(0, x.view(-1).type(torch.int64))
  return res.view(*shape, res.shape[-1])



import time
start_time = time.time()

MAX_DELAY = 8
SEQ_LENGTH = 20



def train():
  epoch_loss = 0
  batch_list = []
  for batch in loader:
      batch_list.append(batch)
      if len(batch_list) == batch_size:
          for inputs, delays, targets in batch_list:
              optimizer.zero_grad()
              one_hot_inputs = idx_to_onehot(inputs)

              logits = model(one_hot_inputs, delays)

              targets = targets.view(-1)

              logits = logits.view(-1, logits.shape[-1])[:targets.shape[0], :]

              loss = criterion(logits, targets[:logits.shape[0]])
              epoch_loss += loss.item()
              loss.backward()
              optimizer.step()
          batch_list = []
  return epoch_loss / len(loader)
def test_variable_delay_model(model, seq_length=20):
  """
  This is the test function that runs 100 different strings through your model,
  and checks the error rate.
  """
  total = 0
  correct = 0
  for i in range(500):
    s = ''.join([random.choice(string.ascii_lowercase) for i in range(seq_length)])
    d = random.randint(0, model.max_delay)
    result = model.test_run(s, d)
    if d > 0:
      z = zip(s[:-d], result[d:])
    else:
      z = zip(s, result)
    for c1, c2 in z:
      correct += int(c1 == c2)
    total += len(s) - d

  return correct / total

# TODO: implement model training here.
hidden_size = 256
model = VariableDelayGRUMemory(hidden_size, MAX_DELAY)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

batch_size = 32
loader = torch.utils.data.DataLoader(VariableDelayEchoDataset(max_delay=MAX_DELAY, size=1000), batch_size=batch_size)


epoch = 1
acc = 0
while acc < 0.99:
  epoch_loss = train()
  acc = test_variable_delay_model(model)
  print("epoch: {0}, loss: {1}, acc: {2}".format(epoch, epoch_loss, acc))
  epoch += 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")