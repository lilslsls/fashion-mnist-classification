import sys
sys.path.append("./NET")
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from MyDataset import MyDataset,root
from Net import Net
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import numpy as np
test_data = MyDataset(txt = root + 'test.txt',transform = transforms.ToTensor())
test_loader = DataLoader(dataset=test_data,batch_size=64)

model = torch.load("./Model/model.pkl")
model = model.cuda()
# test_acc = 0

# for batch_x,batch_y in test_loader:
#     batch_x = batch_x.cuda()
#     batch_y = batch_y.cuda()
#     out = model(batch_x)
#     pred = torch.max(out,1)[1]
#     num_correct =(pred == batch_y).sum()
#     test_acc += num_correct
# print(test_acc.cpu().numpy()/len(test_data))

def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)

# img = Image.open('./picture/22.jpg') # 图片是400x300 宽x高
# plt.figure("dog")
# plt.imshow(img)
# plt.show()
src=cv.imread('./picture/8.jpg')
cv.imshow('1',src)
cv.waitKey(0)

src2 = toTensor(src)
src2 = src2.cuda()
out = model(src2)
print(out)
pred = torch.max(out,1)[1]
pred = pred.cpu().numpy()
# print(pred.cpu().numpy())
if pred == 0:
    print("T恤")
elif pred == 1:
    print("裤子")
elif pred == 2:
    print("套头衫")
elif pred == 3:
    print("裙子")
elif pred == 4:
    print("外套")
elif pred == 5:
    print("凉鞋")
elif pred == 6:
    print("衬衫")
elif pred == 7:
    print("运动鞋")
elif pred == 8:
    print("包")
else :
    print("踝靴")



