import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import os
import PIL
from PIL import Image
import torch
import numpy
import numpy as np

# class SegDataset(Dataset):
#     def __init__(self,rootdir,mask_dir=None,weights_dir=None,transform=None,imageloader=cv2.imread,train = True):
#         super(Dataset,self).__init__()
#         rootdir.sort()
#         mask_dir.sort()
#         weights_dir.sort()
#         self.root = rootdir
#         self.masks = mask_dir
#         self.files = []
#         self.train = train
#         self.weights = weights_dir
#         if train:
#             for path_1,path_2,path_3 in zip(rootdir,mask_dir,weights_dir):
#                 paths = [path_1,path_2,path_3]
#                 for imagefile in os.listdir(paths[0]):
#                     num_file = os.path.splitext(imagefile)[0] + ".npy"
#                     imagepath = os.path.join(paths[0],imagefile)
#                     maskpath = os.path.join(paths[1],imagefile)
#                     weightpath = os.path.join(paths[2],num_file)
#
#                     files = [imagepath,maskpath,weightpath]
#                     self.files.append(files)
#             self.files.sort(key = self.take_first)
#             print(len(self.files))
#         else:
#           for imagefile in os.listdir(rootdir):
#             imagepath = os.path.join(rootdir,imagefile)
#             self.files.append(imagepath)
#             self.files.sort()
#
#         self.transform = transform
#         self.imageloader = imageloader
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self,idx):
#       if self.train:
#         print(self.files[idx][0])
#         print(self.files[idx][1])
#         print(self.files[idx][2])
#         image = self.imageloader(self.files[idx][0])
#         mask = self.imageloader(self.files[idx][1])
#         weight = np.load(self.files[idx][2])
#         # image = cv2.cvtColor(image,)
#         thresh,bit_mask = cv2.threshold(mask,100,255,cv2.THRESH_BINARY)
#         # bit_mask = cv2.bitwise_not(bit_mask)
#         #ret,gray_im = cv2.threshold(gray_im, 0, 255, cv2.THRESH_OTSU)
#         # gray_im = gray_im/255.0
#         pil_image = PIL.Image.fromarray(image).convert('L')
#         pil_mask = PIL.Image.fromarray(bit_mask).convert('L')
#         t_weights = torch.from_numpy(weight).float()
#         t_weights = t_weights.unsqueeze(0)
#         if self.transform:
#             im = self.transform(pil_image)
#             m = self.transform(pil_mask)
#         print(im.shape)
#         print(m.shape)
#         print(t_weights.shape)
#         return (im,m,t_weights)
#       else:
#         image = self.imageloader(self.files[idx])
#         pil_image = PIL.Image.fromarray(image).convert('L')
#         if self.transform:
#           im = self.transform(pil_image)
#         return im
#
#
#     @staticmethod
#     def take_first(elem):
#       return elem[0]
#
# train_transforms = transforms.Compose([
#     transforms.ToTensor()])

# tdata = SegDataset(["/Users/vaishnav/Desktop/Dassrc/Datasets_and_Masked/ICDAR_2013_Actual_Augmented","/Users/vaishnav/Desktop/Dassrc/Datasets_and_Masked/ICDAR_2017_Actual_Augmented"],mask_dir = ["/Users/vaishnav/Desktop/Dassrc/Datasets_and_Masked/ICDAR_2013_Masked_Augmented","/Users/vaishnav/Desktop/Dassrc/Datasets_and_Masked/ICDAR_2017_Masked_Augmented"],weights_dir=["/Users/vaishnav/Desktop/Dassrc/Datasets_and_Masked/ICDAR_2013_Masked_Augmentedweights","/Users/vaishnav/Desktop/Dassrc/Datasets_and_Masked/ICDAR_2017_Masked_Augmentedweights"],transform = train_transforms)
# im,m,t = tdata.__getitem__(35)
# print(torch.unique(t))
# print(tdata.__getitem__(35))

# class SegDataset(Dataset):
#     def __init__(self,rootdir,mask_dir=None,transform=None,imageloader=cv2.imread,train = True):
#         super(Dataset,self).__init__()
#         rootdir.sort()
#         mask_dir.sort()
#         self.root = rootdir
#         self.masks_3 = mask_dir
#         self.masks_3 = mask_dir
#         self.files = []
#         self.train = train
# #        print(self.root)
# #        print(self.masks)
#         # self.main_path = "/Users/vaishnav/Desktop/Dassrc/Datasets_and_Masked"
#         if train:
#             for path1,path2 in zip(rootdir,mask_dir):
#                 for imagefile in os.listdir(path1):
#                     imagepath = os.path.join(path1,imagefile)
#                     maskpath = os.path.join(path2,imagefile)
#                     files = [imagepath,maskpath]
#                     self.files.append(files)
#                     self.files.sort(key = self.take_first)
#                     # print(len(self.files))
#         else:
#           for path1,path2 in zip(rootdir,mask_dir):
#               for imagefile in os.listdir(path1):
#                   imagepath = os.path.join(path1,imagefile)
#                   self.files.append(imagepath)
#                   self.files.sort()
#
#         self.transform = transform
#         self.imageloader = imageloader
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self,idx):
#       if self.train:
#         image = self.imageloader(self.files[idx][0])
#         mask = self.imageloader(self.files[idx][1])
#         # image = cv2.cvtColor(image,)
#         image = cv2.resize(image, (512,512), interpolation=cv2.INTER_AREA)
#         #print("IS", image.shape, mask.shape)
#         bit_mask = cv2.resize(mask, (512,512), interpolation=cv2.INTER_AREA)
#         u1, c1 = numpy.unique(bit_mask, return_counts=True)
# #        print(u1, c1, bit_mask.shape)
#         #thresh,bit_mask = cv2.threshold(bit_mask,100,255,cv2.THRESH_BINARY)
#         #bit_mask = cv2.bitwise_not(bit_mask)
#         #ret,gray_im = cv2.threshold(gray_im, 0, 255, cv2.THRESH_OTSU)
#         #gray_im = gray_im/255.0
#         pil_image = PIL.Image.fromarray(image).convert('L')
#         #pil_mask = PIL.Image.fromarray(bit_mask).convert('L')
#         pil_mask = PIL.Image.fromarray(bit_mask)
#         if self.transform:
#             im = self.transform(pil_image)
#             m = self.transform(pil_mask)
#         return (im,m)
#       else:
#         image = self.imageloader(self.files[idx])
#         #pil_image = PIL.Image.fromarray(image).convert('L')
#         pil_image = PIL.Image.fromarray(image)
#         if self.transform:
#           im = self.transform(pil_image)
#         return im
#
#     @staticmethod
#     def take_first(elem):
#       return elem[0]

# train_transforms = transforms.Compose([transforms.ToTensor()])


# root_dirs = ["ICDAR_2013_Actual_Augmented","ICDAR_2017_Actual_Augmented","Marmot_Actual_Augmented","UNLV_Actual_Augmented"]
# mask_dirs = ["ICDAR_2013_Masked_Augmented","ICDAR_2017_Masked_Augmented","Marmot_Masked_Augmented","UNLV_Masked_Augmented"]
#
# train_data = SegDataset(root_dirs,mask_dir = mask_dirs,transform = train_transforms)

class SegDataset1(Dataset):
    def __init__(self,rootdir,transform=None,imageloader=cv2.imread):
        super(Dataset,self).__init__()
        # rootdir.sort()
        # mask_dir.sort()
        self.root = rootdir
        self.files = []
        print(self.root)
        # self.main_path = "/Users/vaishnav/Desktop/Dassrc/Datasets_and_Masked"
        # if train:
        #     for path1,path2 in zip(rootdir,mask_dir):
        #         for imagefile in os.listdir(path1):
        #             imagepath = os.path.join(path1,imagefile)
        #             maskpath = os.path.join(path2,imagefile)
        #             files = [imagepath,maskpath]
        #             self.files.append(files)
        #             self.files.sort(key = self.take_first)
        #             # print(len(self.files))
        # else:
        # for path1 in zip(rootdir,mask_dir):
        for imagefile in os.listdir(rootdir):
            imagepath = os.path.join(rootdir,imagefile)
            self.files.append(imagepath)
            self.files.sort()
        self.transform = transform
        self.imageloader = imageloader
    def __len__(self):
        return len(self.files)
    def __getitem__(self,idx):
      # if self.train:
        image = self.imageloader(self.files[idx])
        # mask = self.imageloader(self.files[idx][1])
        # image = cv2.cvtColor(image,)
        image = cv2.resize(image, (512,512), interpolation=cv2.INTER_AREA)
#        print("IS", image.shape, mask.shape)
        # bit_mask = cv2.resize(mask, (512,512), interpolation=cv2.INTER_AREA)
#        print("MS", mask.shape)
#        thresh,bit_mask = cv2.threshold(mask,100,255,cv2.THRESH_BINARY)
        #bit_mask = cv2.bitwise_not(bit_mask)
        #ret,gray_im = cv2.threshold(gray_im, 0, 255, cv2.THRESH_OTSU)
        #gray_im = gray_im/255.0
        # pil_image = PIL.Image.fromarray(image).convert('L')
        # #pil_mask = PIL.Image.fromarray(bit_mask).convert('L')
        # # pil_mask = PIL.Image.fromarray(bit_mask)
        # if self.transform:
        #     im = self.transform(pil_image)
        #     m = self.transform(pil_mask)
        # return (im,m)
      # else:
        # image = self.imageloader(self.files[idx])
        pil_image = PIL.Image.fromarray(image).convert('L')
        #pil_image = PIL.Image.fromarray(image)
        if self.transform:
          im = self.transform(pil_image)
        return im,self.files[idx]
    @staticmethod
    def take_first(elem):
      return elem[0]


class SegDataset12(Dataset):
    def __init__(self,rootdir,mask_dir=None,transform=None,imageloader=cv2.imread,train=False):
        super(Dataset,self).__init__()
        rootdir.sort()
        mask_dir.sort()
        self.root = rootdir
        self.masks = mask_dir
        self.files = []
        self.train = train
        print(self.root)
        print(self.masks)
        # self.main_path = "/Users/vaishnav/Desktop/Dassrc/Datasets_and_Masked"
        if train:
            for path1,path2 in zip(rootdir,mask_dir):
                for imagefile in os.listdir(path1):
                    imagepath = os.path.join(path1,imagefile)
                    maskpath = os.path.join(path2,imagefile)
                    files = [imagepath,maskpath]
                    self.files.append(files)
                    self.files.sort(key = self.take_first)
                    # print(len(self.files))
        else:
          for path1,path2 in zip(rootdir,mask_dir):
              for imagefile in os.listdir(path1):
                  imagepath = os.path.join(path1,imagefile)
                  self.files.append(imagepath)
                  self.files.sort()

        self.transform = transform
        self.imageloader = imageloader

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
      if self.train:
        image = self.imageloader(self.files[idx][0])
        mask = self.imageloader(self.files[idx][1])
        # image = cv2.cvtColor(image,)
        image = cv2.resize(image, (512,512), interpolation=cv2.INTER_AREA)
#        print("IS", image.shape, mask.shape)
        bit_mask = cv2.resize(mask, (512,512), interpolation=cv2.INTER_AREA)
#        print("MS", mask.shape)
#        thresh,bit_mask = cv2.threshold(mask,100,255,cv2.THRESH_BINARY)
        #bit_mask = cv2.bitwise_not(bit_mask)
        #ret,gray_im = cv2.threshold(gray_im, 0, 255, cv2.THRESH_OTSU)
        #gray_im = gray_im/255.0
        pil_image = PIL.Image.fromarray(image).convert('L')
        #pil_mask = PIL.Image.fromarray(bit_mask).convert('L')
        pil_mask = PIL.Image.fromarray(bit_mask)
        if self.transform:
            im = self.transform(pil_image)
            m = self.transform(pil_mask)
        return (im,m)
      else:
        image = self.imageloader(self.files[idx])
        pil_image = PIL.Image.fromarray(image).convert('L')
        #pil_image = PIL.Image.fromarray(image)
        if self.transform:
          im = self.transform(pil_image)
        return im,self.files[idx]

    @staticmethod
    def take_first(elem):
      return elem[0]


#########################################################################################################################################################

class SegDataset(Dataset):
    def __init__(self,rootdir,transform=None,imageloader=cv2.imread):
        super(Dataset,self).__init__()
        # rootdir.sort()
        # mask_dir.sort()
        self.root = rootdir
        im_dir = rootdir + "/images"
        mask_3 = rootdir + "/3class"
        mask_1 = rootdir + "/1class"
        self.im_dir = rootdir + "/images"
        self.masks_3 = rootdir + "/3class"
        self.masks_1 = rootdir + "/1class"
        self.files = []
#        print(self.root)
#        print(self.masks)
        # self.main_path = "/Users/vaishnav/Desktop/Dassrc/Datasets_and_Masked"


        for imagefile in os.listdir(im_dir):
            imagepath = os.path.join(im_dir,imagefile)
            mask_3_file = imagefile + ".npy"
            mask3path = os.path.join(mask_3,mask_3_file)
            mask_1_file = imagefile
            mask1path = os.path.join(mask_1,mask_1_file)
            files = [imagepath,mask3path,mask1path]
            self.files.append(files)
            self.files.sort(key = self.take_first)
                    # print(len(self.files))
        # else:
        #   for path1,path2 in zip(rootdir,mask_dir):
        #       for imagefile in os.listdir(path1):
        #           imagepath = os.path.join(path1,imagefile)
        #           self.files.append(imagepath)
        #           self.files.sort()

        self.transform = transform
        self.imageloader = imageloader

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        image = self.imageloader(self.files[idx][0])
        mask1 = self.imageloader(self.files[idx][2])
        mask3 = np.load(self.files[idx][1])
#        print("Mask 3 shape : {}".format(mask3.shape))
#        print("Unique Values in mask 3 : {}".format(np.unique(mask3)))
        mask1 = cv2.cvtColor(mask1,cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (512,512), interpolation=cv2.INTER_AREA)
        #print("IS", image.shape, mask.shape)
        mask1 = cv2.resize(mask1, (512,512), interpolation=cv2.INTER_NEAREST)
        mask3 = cv2.resize(mask3, (512,512), interpolation=cv2.INTER_NEAREST)
        thresh,mask1 = cv2.threshold(mask1,100,255,cv2.THRESH_BINARY)
        cv2.imwrite("temp.png",mask1)
        u1, c1 = numpy.unique(mask1, return_counts=True)
#        print("Unique Values in 1 class Mask : {}".format(u1))
        pil_image = PIL.Image.fromarray(image).convert('L')
        pil_mask = PIL.Image.fromarray(mask1).convert('L')
#        print(u1, c1, bit_mask.shape)

        #bit_mask = cv2.bitwise_not(bit_mask)
        #ret,gray_im = cv2.threshold(gray_im, 0, 255, cv2.THRESH_OTSU)
        #gray_im = gray_im/255.0
        # pil_image = PIL.Image.fromarray(image).convert('L')
        #pil_mask = PIL.Image.fromarray(bit_mask).convert('L')
        # pil_mask1 = PIL.Image.fromarray(mask1)
        if self.transform:
            im = self.transform(pil_image)
            m1 = self.transform(pil_mask)
            # m1 = m1/255
            m1 = m1.type(torch.LongTensor)
            # m1 = m1.squeeze(0)
#            print(torch.unique(m1))
#            print(m1.shape)
            # m3 = self.transform(mask3)

            m3 = torch.from_numpy(mask3)
            # m3 = m3.unsqueeze(0)
#            print(m3.shape)
            # m3 = m3.permute(2,0,1)
#            print(torch.unique(m3))
        return (im,m1,m3)

    @staticmethod
    def take_first(elem):
      return elem[0]

# train_transforms = transforms.Compose([
#     transforms.ToTensor()])




# trainset = SegDataset("./train_data",transform = train_transforms)
#
# image,mask1,mask3 = trainset.__getitem__(10)
