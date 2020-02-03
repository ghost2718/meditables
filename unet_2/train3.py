import torch
import torch.nn as nn
import cv2
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from networks import UNet
from torchsummary import summary
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from data import SegDataset, SegDataset1
import json
from collections import defaultdict
import torch.nn.functional as F
from utils import dice_loss,load_train_test,calc_loss
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy

num_class = 3
num_epochs = 200

#image_dirs = ["../MedTrain_Actual", "../MedVal_Actual"]
#mask_dirs = ["../train3masked", "../val3masked"]

image_dirs = ["../MedVal_Actual"]
mask_dirs = ["../valgray"]

test_dirs = ["../MedTest_Actual"]
mask_dirs1 = ["../test3masked"]

train_transforms = transforms.Compose([
    transforms.ToTensor()])

data = SegDataset(image_dirs,mask_dir = mask_dirs,transform = train_transforms)
print("THe size of the complete dataset is {}".format(data.__len__()))

train_loader,test_loader = load_train_test(data,valid_size = 0.01)

print("The size of training examples is : {}".format(len(train_loader.dataset)))
print("The size of testing examples is : {}".format(len(test_loader.dataset)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = UNet(num_class).to(device)
summary(model,(1,512,512))

d = open('sample.json', 'w+')

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=4e-4)

#model.load_state_dict(torch.load("../outputs5/checkpoints/ckpt_0_31.pth"))
transforms1 = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()])

test_data = SegDataset1(test_dirs,mask_dir=mask_dirs1,transform=transforms1,train=False)
t_dataloader = DataLoader(test_data,batch_size=1,shuffle=False,pin_memory=False)
print(len(test_data))

count1 = 0
#for t_data in t_dataloader:
#    count1 = count1 + 1
#    tr_data = Variable(t_data[0].cuda())
#    print("File", t_data[1][0][18:])
#    print(t_data[0].shape)
#    t_gen = model(tr_data)
#    print(t_gen)
#    gen_out = Variable(tr_data).detach().cpu()
#    utils.save_image(gen_out, '../outputs5/testresults/A'+t_data[1][0][18:]+'.jpg')
#    gen_out = t_gen.detach().cpu()
#    utils.save_image(gen_out, '../outputs5/testresults/G'+t_data[1][0][18:]+'.jpg')
#    print("Saved Image")
















model = nn.DataParallel(model).to(device)
#model = model.to(device)

for epoch in range(400):
    metrics = defaultdict(float)
    epoch_loss = 0
    # train_loss = []
    # val_loss = []
    # epoch_train_loss = []
    # epoch_val_loss = []
    epoch_samples = 0
    for batch_idx,batch in enumerate(train_loader):
          model.train()
          img = batch[0].to(device)
          mask = batch[1].to(device)
          print("Shapes", img.shape, mask.shape)
          u1, c1 = numpy.unique(mask.cpu().numpy(), return_counts=True)
          print("UC", u1, c1)
          epoch_samples += img.size(0)
          pred_mask = model(img)
          loss = calc_loss(pred_mask,mask,metrics)
          epoch_loss += loss.item()
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          print("EPOCH:{0} || BATCH NO:{1} || LOSS:{2}".format(epoch,batch_idx,loss.item()))
          if batch_idx%3000 == 0:
              torch.save(model.module.state_dict(),"../outputs5/checkpoints/ckpt_{}_{}.pth".format(batch_idx,epoch))
              metrics["batch_idx"] = batch_idx
              metrics["epoch"] = epoch
              metrics["epoch_samples"] = epoch_samples
              a = open("../outputs5/train_logs/train_metrics_{}_{}.json".format(batch_idx,epoch), 'w+')
              with open("../outputs5/train_logs/train_metrics_{}_{}.json".format(batch_idx,epoch), 'w') as wrf:
                  json.dump(metrics, wrf)
              for idx in range(img.shape[0]):
                  x = img[idx]
                  #print("Save Shape", x.shape, mask[idx].shape, torch.cat((x,x,x),1).shape)
                  img_list = [torch.cat((x,x,x),0),mask[idx],pred_mask[idx]]
                  utils.save_image(img_list,"../outputs5/image_outs/batch_out_{0}_{1}_{2}.png".format(epoch,batch_idx,idx))
                  print("Images saved to Directory")
              # print()
              # epoch_train_loss.append(metrics["loss"]/len(train_loader.dataset))
              print("Testing on Validation Set")
              test_metrics = defaultdict(float)
              for test_idx,test_batch in enumerate(test_loader):
                  model.eval()
                  test_img  = test_batch[0].to(device)
                  test_mask = test_batch[1].to(device)
                  pred_test = model(test_img)
                  loss = calc_loss(pred_mask,mask,test_metrics)
                  for idx in range(test_img.shape[0]):
                       print("Save Shape", img[idx].shape, mask[idx].shape, pred_mask[idx].shape)
                       #img[idx] = cv2.resize(img[idx], (3,1,512,512), interpolation = cv2.INTER_AREA)
                       x = img[idx]
                       img_list = [torch.cat((x,x,x),0),mask[idx],pred_mask[idx]]
                       utils.save_image(img_list,"../outputs5/image_outs/test_batch_out_{0}_{1}_{2}.png".format(epoch,test_idx,idx))
                       print("Images saved to Test Directory")
              test_metrics["test_samples"] = len(test_loader.dataset)
              c = open("../outputs5/train_logs/test_metrics_{}_{}.json".format(batch_idx,epoch),'w+')
              with open("../outputs5/train_logs/test_metrics_{}_{}.json".format(batch_idx,epoch),'w') as wrf1:
                  json.dump(test_metrics, wrf1)
    b = open("../outputs5/train_logs/epoch_metrics_{}.json".format(epoch), 'w+')
    with open("../outputs5/train_logs/epoch_metrics_{}.json".format(epoch), 'w') as wrf2:
        json.dump(metrics, wrf2)

model.load_state_dict(torch.load(filepath))
transforms1 = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()])

train_data = SegDataset("2013",transform = train_transforms,train = False)
t_dataloader = DataLoader(train_data,batch_size=1,shuffle=False,pin_memory=False)
print(len(train_data))

for t_data in t_dataloader:
    count1 = count1 + 1
    tr_data = Variable(t_data)
    print(t_data.shape)
    t_gen = model(tr_data)
    print(t_gen)
    gen_out = Variable(tr_data).detach().cpu()
    utils.save_image(gen_out, 'saved/'+'Actual_'+'_count_'+str(count1)+'.jpg')
    gen_out = t_gen.detach().cpu()
    utils.save_image(gen_out, 'saved/'+'Generated_'+'_count_'+str(count1)+'.jpg')
