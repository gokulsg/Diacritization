import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import string
import time

vocab = dict(zip(string.ascii_lowercase, range(1,27)))
vocab['<pad>'] = 0

train = open("words_random_train.txt","r",encoding = "utf-8").read().split("\n")
train.remove("")
train_x = [i.split("\t")[1] for i in train]
train_y = [i.split("\t")[0] for i in train]

dev = open("words_random_dev.txt","r",encoding = "utf-8").read().split("\n")
dev_x = [i.split("\t")[1] for i in dev]
dev_y = [i.split("\t")[0] for i in dev]

short_line = ['á','é','í','ý','ú','ó']
little_hook = ['ě','č','š','ž','ř']
circle = ['ů']


label = []

little_hook_cnt = 0
short_line_cnt = 0
circle_cnt = 0
other_char_cnt = 0

for word in train_y:
    temp_lst = []
    word_lst = list(word)
    for char in word_lst:
        if char in short_line:
            temp_lst.append(1)
            short_line_cnt +=1
        elif char in little_hook:
            temp_lst.append(2)
            little_hook_cnt +=1
        elif char in circle:
            temp_lst.append(3)
            circle_cnt +=1
        else:
            temp_lst.append(0)
            other_char_cnt +=1
    
    label.append(temp_lst)
    
## 0 = other characters; 1 = short_line ; 2 = little_hook; 3 = circle;##

    
dev_label = []

for word in dev_y:
    temp_lst = []
    word_lst = list(word)
    for char in word_lst:
        if char in short_line:
            temp_lst.append(1)
        elif char in little_hook:
            temp_lst.append(2)
        elif char in circle:
            temp_lst.append(3)
        else:
            temp_lst.append(0)
    
    dev_label.append(temp_lst)
    

from torch.utils.data import Dataset, DataLoader

class TaggingDataset(Dataset):
    def __init__(self, sentences, tag_sequences):
        self.sentences = sentences
        self.tag_sequences = tag_sequences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sample = {"char": self.sentences[idx], "class": self.tag_sequences[idx]}
        return sample

def tagging_collate_fn(batch):
    tensors = []
    for instance in batch:
        sent_t = torch.tensor(instance["char"])
        pos_t = torch.tensor(instance["class"])
        tensors.append(torch.stack([sent_t, pos_t]))

    return torch.stack(tensors)

rnn_x = []
for word in train_x:
    temp_lst = []
    word_lst = list(word)
    for char in word_lst:
        temp_lst.append(vocab[char])
    
    rnn_x.append(temp_lst)

train_dataset = TaggingDataset(rnn_x, label)
train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=tagging_collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rnn_dev = []
for word in dev_x:
    temp_lst = []
    word_lst = list(word)
    for char in word_lst:
        temp_lst.append(vocab[char])
    
    rnn_dev.append(temp_lst)

dev_dataset = TaggingDataset(rnn_dev, dev_label)
dev_dataloader = DataLoader(dev_dataset, batch_size=1, collate_fn=tagging_collate_fn)

from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, hidden_dim=64, layers=2, dropout_val=0.1):   
        super().__init__()
        self.embedding = nn.Embedding(27,10) #embedding size - 10
        self.lstm = nn.LSTM(10, hidden_dim, num_layers = layers, bidirectional = True)
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(dropout_val)
        self.fc1 = nn.Linear(hidden_dim, 4)
        
    def forward(self, text):
        embedded = self.embedding(text)
        outputs, (hidden, cell) = self.lstm(embedded.view(len(text), 1, -1))
        predictions = self.fc(outputs.view(len(text), -1))
        predictions = self.dropout(predictions)
        predictions = self.fc1(predictions)
        return predictions
def accuracy(tag_scores,tag):
    _ , predicted_idx = torch.max(tag_scores, 1)
    return torch.sum(predicted_idx == tag).item()/len(tag)
    
def train(train_dataloader, loss, optimizer, model, device):
    model.train()
    train_total_acc = 0
    train_total_loss = 0
    
    for batch in train_dataloader: 
        model.zero_grad()
        word = batch[0][0].to(device)
        label = batch [0][1].to(device)
        scores = model(word)
        
        loss_val = loss(scores, label)
        
        loss_val.backward()
        optimizer.step()

        train_total_loss += loss_val.item()
        
        train_total_acc += accuracy(scores, label)
    
    return train_total_loss, train_total_acc

def evaluate(dev_dataloader, loss, model, device):
    eval_total_loss = 0
    eval_total_acc = 0
    model.eval()
    for eval_batch in dev_dataloader:
        word = eval_batch[0][0].to(device)
        label = eval_batch [0][1].to(device)
        scores = model(word)

        loss_val_eval = loss(scores, label)
        eval_total_loss += loss_val_eval.item()
        
        eval_total_acc += accuracy(scores, label)
        
    return eval_total_loss, eval_total_acc  


def rnnmodel():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel().to(device)
    loss = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        print("Epoch : ",epoch+1)
        start_time = time.time()

        train_total_loss, train_total_acc = train(train_dataloader, loss, optimizer, model, device)
        eval_total_loss, eval_total_acc = evaluate(dev_dataloader, loss, model, device)   

        print('Train : Total accuracy : ', train_total_acc/len(train_dataloader))
        print('Train : Loss : ',train_total_loss/len(train_dataloader))
        print('Dev : Total Accuracy : ',eval_total_acc/len(dev_dataloader))
        print('Dev : Total loss : ',eval_total_loss/len(dev_dataloader))
        print('Time taken per epoch : ',time.time() - start_time)
        print('________________________________________________________________________________________')
    
    return model