import sys
sys.path.append("./NET")
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from MyDataset import MyDataset,root
from Net import Net
import numpy as np
train_data = MyDataset(txt = root + 'train.txt',transform = transforms.ToTensor())
test_data = MyDataset(txt = root + 'test.txt',transform = transforms.ToTensor())

train_loader = DataLoader(dataset=train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(dataset=test_data,batch_size=64)
print(len(train_data))

model = Net()
model = model.cuda()
print(model)

optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()

class My_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        y = y.cpu()
        print(y)
        y = torch.zeros(64, 10).scatter_(1, y, 1)
        y = y.cuda()
        return -torch.add(y*torch.log(x))

loss_fun2 = My_loss()    

for epoch in range(40):
    train_loss = 0.
    train_acc = 0.
    i = 0
    for batch_x,batch_y in train_loader:
        i = i + 1
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        out = model(batch_x)
        # loss =loss_func(out,batch_y)
        loss =loss_fun2(out,batch_y)
        train_loss += loss
        pred =torch.max(out,1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('epoch {}'.format(epoch + 1),loss)
        # print(train_correct)
        # print(train_acc.cpu().numpy())
    print('epoch {}'.format(epoch + 1),'Train Loss:',train_loss/i,'ACC:',train_acc.cpu().numpy()/(len(train_data)))
torch.save(model, './Model/model.pkl')  # 保存整个网络