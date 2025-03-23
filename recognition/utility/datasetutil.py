import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import cv2
import time
import os
import copy
import sys
import glob as _glob
import csv
import skimage.io as iio
import skimage.transform as skiT
import skimage.color as skiC
import recognition.utility.dtype as dtype
from torch.utils.data import Dataset

import matplotlib.cm as cm

import torchvision.transforms.functional as TF

from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

csv.register_dialect(
    'mydialect',
    delimiter = ',',
    quotechar = '"',
    doublequote = True,
    skipinitialspace = True,
    lineterminator = '\r\n',
    quoting = csv.QUOTE_MINIMAL)

#region 데이터 저장용

def imwrite(image, path, **plugin_args):
    """Save a [-1.0, 1.0] image."""
    iio.imsave(path, dtype.im2uint(image), **plugin_args)


def mkdir(paths):
  if not isinstance(paths, (list, tuple)):
    paths = [paths]
  for path in paths:
    if not os.path.exists(path):
      os.makedirs(path)

def split(path):
  """Return dir, name, ext."""
  dir, name_ext = os.path.split(path)
  name, ext = os.path.splitext(name_ext)
  return dir, name, ext


def glob(dir, pats, recursive=False):  # faster than match, python3 only
    pats = pats if isinstance(pats, (list, tuple)) else [pats]
    matches = []
    for pat in pats:
        matches += _glob.glob(os.path.join(dir, pat), recursive=recursive)
    return matches
#endregion

#region csv 파일 처리 함수
#authentic DATASET load
def authentic_ds(csvname):
    register_ds=csv2list(csvname)
    for i,x in enumerate(register_ds):
        register_ds[i][0]=0
    return register_ds


#imposter DATASET load (본인클래스데이터 만 제외하고 나머지만 randomchoice)
def imposter_ds(csvname,path,numofcls,numofclsfile):
    ds = csv2list(csvname)
    files = glob(path, '*/*')
    files = [x.replace('\\','/') for x in files]
    ds_np_return= np.array(ds)
    ### 글자수 제한 풀기 ( numpy는 type에서 글자수 제한함 (이유 : 메모리 땀시)
    ### 아래의 U80은 80자 까지 받을수 있게 바꾸는 거임
    ds_np_return = ds_np_return.astype('U80')
    #같은 클래스 중복안되게 제거후 삽입  삽입
    for i in range(numofcls):
        fpfiles=copy.deepcopy(files)
        del fpfiles[numofclsfile*(i):numofclsfile*(i+1)]
        mask = np.random.choice(len(fpfiles), 900,replace=False)
        fpfiles = np.array(fpfiles)
        fpfiles = fpfiles[mask]
        ds_np_return[numofclsfile*(i)*30:numofclsfile*(i+1)*30,0]=1
        ds_np_return[numofclsfile*(i)*30:numofclsfile*(i+1)*30,2]=fpfiles

    return ds_np_return.tolist()


def imposter_test_ds(csvname, path, numofcls, numofclsfile):
    ds = csv2list(csvname)
    files = glob(path, '*/*')
    files = [x.replace('\\', '/') for x in files]
    ds_np = np.array(ds)
    ds_np = np.unique(ds_np[:, 1])
    ds_np = ds_np.tolist()
    ds_np_return = np.array(ds)
    # list에서 등록영상만 제거
    for x in ds_np:
        files.remove(x)
    # 같은 클래스 중복안되게 제거후 삽입  삽입
    for i in range(numofcls):
        fpfiles = copy.deepcopy(files)
        del fpfiles[numofclsfile * (i):numofclsfile * (i + 1)]
        ds_np_return[numofclsfile * (i) * (numofcls - 1):numofclsfile * (i + 1) * (numofcls - 1), 0] = 1
        ds_np_return[numofclsfile * (i) * (numofcls - 1):numofclsfile * (i + 1) * (numofcls - 1), 2] = fpfiles

    return ds_np_return.tolist()


def imposter_ds_for_gradcam(csvname):
    register_ds=csv2list(csvname)
    for i,x in enumerate(register_ds):
        register_ds[i][0]=1
    return register_ds

def csv2list(filename):
  lists=[]
  file=open(filename,"r")
  while True:
    line=file.readline().replace('\n','')
    if line:
      line=line.split(",")
      lists.append(line)
    else:
      break
  return lists

def writecsv(csvname,contents):
    f = open(csvname, 'a', newline='')
    wr = csv.writer(f)
    wr.writerow(contents)
    f.close()
#endregion

def split(path):
  """Return dir, name, ext."""
  dir, name_ext = os.path.split(path)
  name, ext = os.path.splitext(name_ext)
  return dir, name, ext

def make_composite_image(img1,img2):  # input shape (C, H, W)
    #이미지 사이즈 부터 체크
    if img1.shape[1]!=224 and img1.shape[2]!=224:
        img1 = TF.resize(img1,(224, 224))

    # 채널 체크(gray scale 이미지이면 reshape / channel이 3이면 1채널로
    if len(img1.shape) < 3:
        img1 = img1.unsqueeze(0)
    else:
        if img1.shape[0] > 1:
            img1 = TF.rgb_to_grayscale(img1)  # (1, 224, 224)

    # 이미지 사이즈 부터 체크
    if img2.shape[1] != 224 and img2.shape[2] != 224:
        img2 = TF.resize(img2, (224, 224))

    # 채널 체크(gray scale 이미지이면 reshape / channel이 3이면 1채널로
    if len(img2.shape) < 3:
        img2 = img2.unsqueeze(0)
    else:
        if img2.shape[0] > 1:
            img2 = TF.rgb_to_grayscale(img2)  # (1, 224, 224)

    # 3 채널 #height 기준 concatenation)
    img3_1 = TF.resize(img1, (112, 224))  # C, H, W
    img3_2 = TF.resize(img2, (112, 224))  # C, H, W
    img3 = torch.cat((img3_1, img3_2), 1)  # (1, 224, 224)

    # # 데이터 검증용
    # fig = plt.figure(figsize=(30, 30))
    # plt.subplot(2, 8, 1)
    # img1_show = np.concatenate([img1, img1, img1], axis=2)
    # plt.imshow(img1_show)
    #
    # plt.subplot(2, 8, 2)
    # img2_show = np.concatenate([img2, img2, img2], axis=2)
    # plt.imshow(img2_show)
    #
    # plt.subplot(2, 8, 3)
    # img3_show = np.concatenate([img3, img3, img3], axis=2)
    # plt.imshow(img3_show)
    #
    # plt.show()

    input_img = torch.cat((img1, img2, img3), 0)  # (3, 224, 224)
    return input_img


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    gcam[np.isnan(gcam)]=0
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    raw_image=raw_image*255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


#region pytorch Custom dataset
class FingerveinDataset(Dataset):
    def __init__(self,dslist,test_img_path,transform=None):
        self.dslist = dslist
        self.transform = transform
        self.test_img_path = test_img_path

    def __len__(self):
        return len(self.dslist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cls=int(self.dslist[idx][0])
        img_name1=self.dslist[idx][1]
        img_name2=self.test_img_path+'/'+self.dslist[idx][2].replace('\\','/').split('/')[-2]+'/'+self.dslist[idx][2].replace('\\','/').split('/')[-1]

        img1 = iio.imread(img_name1)
        img2 = iio.imread(img_name2)
        #img1 = skiT.resize(img1,(256,256))
        #img2 = skiT.resize(img2,(256,256))
        pixel_diff=(np.abs(img1.astype('float32')-img2.astype('float32'))-127.5)/127.5
        pixel_diff=self.transform(pixel_diff)

        return cls,pixel_diff

class FingerveinDataset_zeros(Dataset):
    def __init__(self, dslist, transform=None):
        self.dslist = dslist
        self.transform = transform

    def __len__(self):
        return len(self.dslist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cls=int(self.dslist[idx][0])  # labels. genuine 0, imposter 1
        img_name1=self.dslist[idx][1]
        targets_name=self.dslist[idx][2].replace('\\','/').split('/')

        img_name2=self.dslist[idx][2]
        # 70 * 180 (이미지 높이 x 이미지 너비)

        img1 = iio.imread(img_name1)
        img2 = iio.imread(img_name2)

        input_img = make_composite_image(img1, img2)

        input_img = self.transform(input_img)

        return cls, input_img, [img_name1, img_name2]

class FingerveinDataset_zeros_with_aug(Dataset):
    def __init__(self,dslist,transform=None):
        self.dslist=dslist
        self.transform=transform

    def __len__(self):
        return len(self.dslist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cls=int(self.dslist[idx][0])
        img_name1=self.dslist[idx][1]
        img_name2=self.dslist[idx][2]

        img1 = iio.imread(img_name1)
        img2 = iio.imread(img_name2)
        # img1 = skiT.resize(img1,(256,256))
        # img2 = skiT.resize(img2,(256,256))
        pixel_diff =np.abs(np.subtract(img1.astype('int16'),img2.astype('int16')))
        pixel_diff=Image.fromarray(pixel_diff.astype('uint8'))
        pixel_diff=self.transform(pixel_diff)

        return cls,pixel_diff

# -1 ~ 1로 정규화
class FingerveinDataset_test(Dataset):
    def __init__(self,dslist,transform=None):
        self.dslist=dslist
        self.transform=transform

    def __len__(self):
        return len(self.dslist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cls=int(self.dslist[idx][0])
        img_name1=self.dslist[idx][1]
        img_name2=self.dslist[idx][2]

        img1 = iio.imread(img_name1)
        img2 = iio.imread(img_name2)
        pixel_diff=(np.abs(img1.astype('float32')-img2.astype('float32'))-127.5)/127.5
        pixel_diff=self.transform(pixel_diff)

        return cls,pixel_diff,[img_name1,img_name2]

# 0 ~ 1로 정규화
class FingerveinDataset_test_zeros(Dataset):
    def __init__(self,dslist,path,transform=None,Use_blendset=False):
        self.dslist = dslist
        self.folder = path
        self.transform = transform
        self.Use_blendset = Use_blendset

    def __len__(self):
        return len(self.dslist)

    # shift matching 대상들 특정
    def make_Matching_files(self,filenames):
        paths_for_matching = split(filenames)
        directory = paths_for_matching[0][-3:]
        GB = paths_for_matching[1][1:]
        files = glob(self.folder + '/' + directory, '*')
        M_mask = np.where(np.char.find(files, GB) >= 0)

        return np.array(files)[M_mask]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cls=int(self.dslist[idx][0])  # class label
        img_name1=self.dslist[idx][1]
        img_name2=self.dslist[idx][2]

        # shift matching 대상들 특정

        matching_files= self.make_Matching_files(img_name2)

        #등록영상을 조도 조절한 것으로 쓰려면 False로
        if self.Use_blendset:
            paths_for_matching = split(img_name1)
            directory = paths_for_matching[0][-3:]
            img_name1=self.folder + '/' + directory+'/'+paths_for_matching[1]+'.bmp'

        img1 = iio.imread(img_name1)
        outputs = torch.zeros(1,3,224,224)

        for i,filepath in enumerate(matching_files):
            img2 = iio.imread(filepath)
            output=make_composite_image(img1,img2)
            output=self.transform(output)  # 텐서로 바꿈
            output=torch.reshape(output,(1,3,224,224))
            if i==0:
                outputs = outputs+output
            else:
                outputs = torch.cat((outputs,output),dim=0)

        return cls,outputs,[img_name1,img_name2],matching_files.tolist()


class FingerveinDataset_test_zeros_FOR_GRADCAM(Dataset):
    def __init__(self, path1, path2, save_path, GB_idx, transform=None):
        self.path1 = path1 # enrolled image list
        self.path2 = path2 # target image list
        self.save_path = save_path
        self.GB_idx = GB_idx ### authentic 0 / imposter 1 구분용
        self.transform = transform

    def __len__(self):
        return len(self.path1)

    def __getitem__(self, idx):
        img_name1=self.path1[idx]
        img_name2=self.path2[idx]

        img1 = iio.imread(img_name1)
        img2 = iio.imread(img_name2)

        output = make_composite_image(img1,img2)
        # iio.imsave(self.save_path+img_name1.replace('\\','/').split('/')[-1], output)
        output = self.transform(output)


        return output, self.GB_idx

class FingerveinDataset_test_zeros_forloss(Dataset):
    def __init__(self,dslist,path,valid_data,transform=None,Use_blendset=False):
        self.dslist=dslist
        self.folder = path
        self.transform=transform
        self.Use_blendset=Use_blendset
        self.valid_data=valid_data

    def __len__(self):
        return len(self.dslist)

    # shift matching 대상들 특정
    def make_Matching_files(self,filenames):
        paths_for_matching = split(filenames)
        directory = paths_for_matching[0][-3:]
        GB = paths_for_matching[1][1:]
        files = glob(self.folder + '/' + directory, '*')
        M_mask = np.where(np.char.find(files, GB) >= 0)

        return np.array(files)[M_mask]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cls=int(self.dslist[idx][0])
        img_name1=self.dslist[idx][1]
        img_name2=self.dslist[idx][2]
        # 데이터 구분자 문제로 list 체크 안되서 따로 매칭되도록 수정
        img_name1=img_name1.split('/')[0]+'/'+img_name1.split('/')[1]+'/'+img_name1.split('/')[2]+'\\'+img_name1.split('/')[3]+'\\'+img_name1.split('/')[4]
        if '\\' not in img_name2:
            img_name2 = img_name2.split('/')[0] + '/' + img_name2.split('/')[1] + '/' + img_name2.split('/')[2] + '\\' + img_name2.split('/')[3] + '\\' + img_name2.split('/')[4]


        if img_name1 in self.valid_data and img_name2 in self.valid_data:
            # shift matching 대상들 특정
            matching_files= self.make_Matching_files(img_name2)
            targetsindex=[0]
            matching_files=matching_files[targetsindex]

            #등록영상을 조도 조절한 것으로 쓰려면 False로
            if self.Use_blendset:
                paths_for_matching = split(img_name1)
                directory = paths_for_matching[0][-3:]
                img_name1=self.folder + '/' + directory+'/'+paths_for_matching[1]+'.bmp'

            img1 = iio.imread(img_name1)
            outputs = torch.zeros(1,3,224,224)

            for i,filepath in enumerate(matching_files):
                img2 = iio.imread(filepath)
                output=make_composite_image(img1,img2)
                output=self.transform(output)
                output=torch.reshape(output,(1,3,224,224))
                if i==0:
                    outputs = outputs+output
                else:
                    outputs = torch.cat((outputs,output),dim=0)


            return cls,outputs,[img_name1,img_name2],matching_files.tolist()
        else:
            return 1,1,1,1



class FingerveinDataset__savedata(Dataset):
    def __init__(self,dslist,transform=None):
        self.dslist=dslist
        self.transform=transform

    def __len__(self):
        return len(self.dslist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cls=int(self.dslist[idx][0])
        img_name1=self.dslist[idx][1]
        img_name2=self.dslist[idx][2]

        img1 = iio.imread(img_name1)
        img2 = iio.imread(img_name2)
        savaname1=split(img_name1)[-2]
        savename2=split(img_name2)[-2]

        if cls==0:
            fol='auth'
        else:
            fol='impo'

        pixel_diff=(np.abs(img1.astype('float32')-img2.astype('float32'))-127.5)/127.5
        imwrite(pixel_diff,'Output/diffimages_for_why/'+fol+'/'+savaname1+'-'+savename2+'.bmp')
        pixel_diff=self.transform(pixel_diff)

        return cls,pixel_diff,[img_name1,img_name2]

class FingerveinDataset_for_loss(Dataset):
    def __init__(self,originpath,ganpath,transform=None):
        self.transform=transform
        self.originpath=originpath
        self.ganpath=ganpath
    def __len__(self):
        return len(self.originpath)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name1=self.originpath[idx]

        img1 = iio.imread(img_name1)

        img1 = skiT.resize(img1,(256,256)).astype('float32')

        img1=self.transform(img1)

        img_name2=self.ganpath[idx]

        img2 = iio.imread(img_name2).astype('float32')/255.

        img2=self.transform(img2)


        return img1,img2,img_name1,img_name2


class FingerveinDataset_make_diff_images(Dataset):
    def __init__(self,dslist,transform=None):
        self.dslist=dslist
        self.transform=transform

    def __len__(self):
        return len(self.dslist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cls=int(self.dslist[idx][0])
        img_name1=self.dslist[idx][1]
        img_name2=self.dslist[idx][2]

        img1 = iio.imread(img_name1)
        img2 = iio.imread(img_name2)
        pixel_diff=(np.abs(img1.astype('float32')-img2.astype('float32'))-127.5)/127.5

        return cls,pixel_diff,[img_name1,img_name2]
#endregion