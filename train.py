import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from MyDataset import MyDataset,root
from Net import Net

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

for epoch in range(10):
    train_loss = 0.
    train_acc = 0.
    for batch_x,batch_y in train_loader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        out = model(batch_x)
        loss =loss_func(out,batch_y)
        train_loss += loss
        pred =torch.max(out,1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch {}'.format(epoch + 1),loss)
    # print('epoch {}'.format(epoch + 1),'Train Loss:{:.6f},ACC:{:.6f}'.format(train_loss/(len(train_data)),train_acc/(len(train_data))))
