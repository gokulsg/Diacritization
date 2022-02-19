import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import string
import time
import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

def plot():
    x = ['little_hook', 'short_line', 'circle', 'other_char']
    y = [little_hook_cnt, short_line_cnt, circle_cnt, other_char_cnt]

    plt.barh(x,y)
    plt.title('Distribution of classes in train dataset')
    plt.ylabel('Classes')
    plt.xlabel('Number of occurance')

    # Adding values to bar chart #
    for index, value in enumerate(y):
        plt.text(value, index, str(value))

    plt.plot()
    #plt.show()

    
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
    
# X and y values for the baseline model #
X_val = []
for word in train_x:
    char_lst = list(word)
    for char in char_lst:
        temp = list(bin(vocab[char])[2:].zfill(5)) # Using binary encoding for memory efficiency
        X_val.append([int(item) for item in temp])
        
X_lab = [item for sublist in label for item in sublist]

training_data = torch.utils.data.TensorDataset(torch.Tensor(X_val), torch.Tensor(X_lab))
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=512)

X_val_dev = []
for word in dev_x:
    char_lst = list(word)
    for char in char_lst:
        temp = list(bin(vocab[char])[2:].zfill(5)) # Using binary encoding for memory efficiency
        X_val_dev.append([int(item) for item in temp])
        
X_lab_dev = [item for sublist in dev_label for item in sublist]

dev_data = torch.utils.data.TensorDataset(torch.Tensor(X_val_dev), torch.Tensor(X_lab_dev))
dev_dataloader = torch.utils.data.DataLoader(dev_data, batch_size=512)

def calculate_correct_tag_num(prediction,y):
    prediction = torch.max(prediction,1)[1]
    correct = 0
    for i,j in zip(prediction,y):
        if i==j:
            correct+=1
    return correct


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    tot_loss = 0
    accuracy = 0
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X.float())
        loss = loss_fn(pred, y.type(torch.LongTensor))

        # Backpropagation
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        accuracy +=calculate_correct_tag_num(pred, y)
    print("loss: ", tot_loss/size)
    print("Accuracy: ", accuracy/size)

def eval_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    tot_loss = 0
    accuracy = 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X.float())
        loss = loss_fn(pred, y.type(torch.LongTensor))
        
        tot_loss += loss.item()
        accuracy +=calculate_correct_tag_num(pred, y)
    print("Validation loss: ", tot_loss/size)
    print("Validation Accuracy: ", accuracy/size)
    
# Baseline model #
class SimpleModel(nn.Module):
    def __init__(self, hidden_size = 24):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(5, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 4)
        
    def forward(self,x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    
net = SimpleModel()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

def baseline():
    epochs = 20
    for t in range(epochs):
        start_t = time.time()
        print("Epoch : ", t+1)
        train_loop(train_dataloader, net, loss_function, optimizer)
        eval_loop(dev_dataloader, net, loss_function)
        print("Time taken: ",time.time() - start_t)