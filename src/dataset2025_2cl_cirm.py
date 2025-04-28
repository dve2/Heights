from math import nan
from torch.utils.data import Dataset
from glob import glob
import os
from pathlib import Path
import torch
import pandas as pd
import json
import cv2
import numpy as np
import warnings
import ast
import pickle
import torchvision.transforms.functional as F
from statistics import mode
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision.transforms import ToTensor

class BaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform = None, exclude = [], cache = None):
      images = glob(f"{root_dir}{os.sep}Images{os.sep}*")
      labels = glob(f"{root_dir}{os.sep}Masks{os.sep}_*") # read only continuous heigths
      # extract id from paths
      im = set(map(lambda x: Path(x).stem,images))
      lab = set(map(lambda x: Path(x).stem[1:],labels))
      img_without_masks = im - lab
      if len(img_without_masks) > 0:
        warn_text = f"Found images without masks {','.join(img_without_masks)}"
        warnings.warn(warn_text)
      self.items =  (list((im & lab) - set(exclude)))
      self.items.sort()
      self.root_dir = root_dir
      self.transform = transform
      self.target_transform = target_transform
      self.max_height = 100
      if cache == None:
          self.cache = {}
      else:
        self.cache = cache

      #self.load_cache()
      #self.scales = self.get_all_scales()

    def get_all_scales(self):
        scales = []
        for name in self.items:
            path = self.get_im_path(name)
            image, real_w = self.txt2pil(path)
            scales.append(self.get_scale(image, real_w))
        return np.array(scales)

    def get_scale(self, img, real_w):
      return  real_w / img.shape[1]   #  micron per pixel; "[0]" changed to "[1]"

    def get_im_path(self,name):
      path = f"{self.root_dir}{os.sep}Images{os.sep}{name}.txt"
      return path

    def get_mask_path(self,name):
      path = f"{self.root_dir}{os.sep}Masks{os.sep}_{name}.txt"
      return path

    def save_cache(self,filename = "cache.pickle"):#??
      with open(filename, 'wb') as f:
        pickle.dump(self.cache, f, pickle.HIGHEST_PROTOCOL)

    def load_cache(self,filename = "cache.pickle"):#??
      if os.path.isfile(filename):
        with open(filename, 'rb') as f:
          self.cache =  pickle.load(f)
        print(f"Loaded cache from {filename}")

    def __len__(self):
      return len(self.items)

    def line2tensor(self,line):
        txt = line.strip()
        parts = txt.split("\t")
        parts = list(filter(len,parts)) #remove empty
        if len(parts) <= 2:
          return None
        numbers = list(map(float, parts))
        t = torch.tensor(numbers)
        return t


    def txt2pil(self, filename):
      if filename in self.cache and "image" in self.cache[filename]:
        return self.cache[filename]["image"].copy(), self.cache[filename]["real_w"]
    
      # convert list of relative heights to image
      with open(filename, encoding='unicode_escape') as file:
        x_line = file.readline() # X
        x = self.line2tensor(x_line[6:]) # bypass X,nm "8:" was replaced by "5:"
        real_w = (x.max()-x.min()).item() #let it be in microns
        units = x_line[3:5]
        if units == "A°":
          real_w = real_w/10000
        if units == "nm":
          real_w = real_w/1000
        line = file.readline() # Y, Z skip it
        lines = []
        for line in file:
          if line != '\n':  #to exclude the last line
            pos = line.index('\t')#position of the first tabulation
            line2 = line[(pos + 2):]#exclude Y-coordinate and 2 tabulations after it
            t = self.line2tensor(line2)
            if t is not None:
              lines.append(t)
        t = torch.stack(lines)
        # Shift to zero
        # Because all heights just a difference between current and randomly sampled point
        t = t - t.min()
        t = t.numpy()
        self.cache[filename]= {"image": t, "real_w" : real_w}
      return t, real_w


    def load_heights(self, path):
      """
        get heights of some points marked by human
      """
      df = pd.read_excel(path)
      return self.fix_format(df)


    def get_height_map(self, path):
      if not (path in self.cache and "mask" in self.cache[path]):
          with open(path, 'r') as file:
            content = file.read()
          x = ast.literal_eval(content)
          x = np.array(x)
          self.cache[path] = { "mask" : x }
      return self.cache[path]["mask"].copy()

    def x_from_y(self, y, i, j, area):
        xmin = j - (area**2 - (y - i)**2)**0.5
        xmax = j + (area**2 - (y - i)**2)**0.5
        #return xmin, xmax
        return round(xmin), round(xmax)
     
    def create_one_circle(self, cir_mask, i,j, area):
        for y in range(max((i-area), 0), min((i+area+1), len(cir_mask))):
            xminmax=[]
            xminmax.append(self.x_from_y(y, i, j, area)[0])
            xminmax.append(self.x_from_y(y, i, j, area)[1])
            if y != max((i-area), 0):
                xminmax.append(self.x_from_y(y-0.5, i, j, area)[0])
                xminmax.append(self.x_from_y(y-0.5, i, j, area)[1])
            if y != min(i+area, len(cir_mask)-1):
                xminmax.append(self.x_from_y(y+0.5, i, j, area)[0])
                xminmax.append(self.x_from_y(y+0.5, i, j, area)[1])
            #for x in range(max(round(min(xminmax)),0), min(round(max(xminmax))+1, len(cir_mask[0]))):
            for x in range(max(min(xminmax),0), min((max(xminmax)+1), len(cir_mask[0]))):
                cir_mask[y][x] = 1
        return cir_mask
                
    def make_circles(self, mask, area = 2):
      cir_mask = np.zeros((mask.shape[0], mask.shape[1]))
      for i in range(len(mask)):
        for j in range(len(mask[i])):
          if mask[i][j] != 0:
            cir_mask = self.create_one_circle(cir_mask, i,j, area)
      return cir_mask
                       
    
    def __getitem__(self,n):
      """
        img - data(raw heights) from microscope
        masks - continious globules height map

        real_w - width of the image in microns
      """
      name = self.items[n]
      img = self.get_im_path(name)
      mask = self.get_mask_path(name)

      image, real_w = self.txt2pil(img)
      mask = self.get_height_map(mask) 
      mask[mask > 0] = 1 #binarizing mask (target) for segmentation
      mask = self.make_circles(mask)
      #image, orig_size, scale_factor = self.rescale(image,real_w)
      #mask, _, _ = self.rescale(mask,real_w)
      scale_factor = 0

      if self.transform:
        output = self.transform(image=image, mask=mask)
        image = output['image']
        mask = output['mask'] # here mask is cropped but not normalized
        # TODO resize to one scale
        #if self.target_transform:
        #  mask = self.target_transform(mask)
      meta = {"w": real_w, 'name' : name, "scale_factor": scale_factor}
      mask=mask.type(torch.LongTensor)
      return image, mask, meta  #normalized image (if tranform is true), binarized mask

    def rescale(self,img, real_w):
      resize_coeff = 1
      h,w = img.shape[:2]
      original_size = (h,w)
      #most_popular_scale = mode(self.scales)
      most_popular_scale = 0.00389862060546875
      scale = self.get_scale(img,real_w)
      if most_popular_scale != scale:
        resize_coeff = most_popular_scale/scale
      new_size = tuple((np.array(original_size) / resize_coeff).astype(int).tolist()) # '*' changed to '/'
      img = cv2.resize(img, new_size)
      return img, original_size, resize_coeff