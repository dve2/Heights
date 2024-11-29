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
'''
def load_cache(filename = "cache.pickle"):
      if os.path.isfile(filename):
        with open(filename, 'rb') as f:
           cache =  pickle.load(f)
        print(f"Loaded cache from {filename}")
        return cache
      return {}
'''

class BaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform = None, exclude = [], cache = None):
      images = glob(f"{root_dir}{os.sep}Images{os.sep}*")
      labels = glob(f"{root_dir}{os.sep}Masks{os.sep}!*") # read only continuous heigths
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
      path = f"{self.root_dir}{os.sep}Masks{os.sep}!{name}.txt"
      return path
    
    def save_cache(self,filename = "cache.pickle"):
      with open(filename, 'wb') as f:
        pickle.dump(self.cache, f, pickle.HIGHEST_PROTOCOL)

    def load_cache(self,filename = "cache.pickle"):
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
      #return x


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

      #image, orig_size, scale_factor = self.rescale(image,real_w)
      #mask, _, _ = self.rescale(mask,real_w)
      scale_factor = 0 

      if self.transform:
        output = self.transform(image=image, mask=mask)
        image = output['image']
        mask = output['mask'] # here mask is cropped but not normalized 
        # TODO resize to one scale
        if self.target_transform:
          mask = self.target_transform(mask) 
      meta = {"w": real_w, 'name' : name, "scale_factor": scale_factor} # , centers: [[x1,y1],[x2,y2]] 
      return image, mask, meta

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


class DoubleMaskDataset(BaseDataset):
    def get_mask2_path(self, name):
        path = f"{self.root_dir}{os.sep}Masks{os.sep}_{name}.txt"
        return path

    def __getitem__(self, n):
        """
                img - data(raw heights) from microscope
                masks - continious globules height map

                real_w - width of the image in microns
              """
        name = self.items[n]
        img = self.get_im_path(name)
        mask = self.get_mask_path(name)
        mask2 = self.get_mask2_path(name)

        image, real_w = self.txt2pil(img)
        mask = self.get_height_map(mask)
        mask2 = self.get_height_map(mask2)

        scale_factor = 0

        if self.transform:
            output = self.transform(image=image, mask=mask, mask2=mask2)
            #output2 = self.transform(image=image, mask=mask2)
            image = output['image']
            mask = output['mask']  # here mask is cropped but not normalized
            mask2 = output['mask2']
            if self.target_transform:
                mask = self.target_transform(mask)
                mask2 = self.target_transform(mask2)
        meta = {"w": real_w, 'name': name, "scale_factor": scale_factor}  # , centers: [[x1,y1],[x2,y2]]
        return image, mask, mask2, meta


class GroundDataset(BaseDataset):

    def get_label_path(self, name):
        path = f"{self.root_dir}{os.sep}Labels{os.sep}{name}.xlsx"
        return path

    def __getitem__(self, n):
        image, mask, meta = super().__getitem__(n)
        name = self.items[n]
        label_path = self.get_label_path(name)
        ground = pd.read_excel(label_path)
        xy = ground.iloc[:, 1:3].to_numpy()
        img = image.copy()
        h, w = image.shape[:2]
        ground_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        for x, y in xy:
            img_tmp, msk = self.flood_fill(img, (x, y))
            ground_mask = ground_mask + msk
            ground_mask[ground_mask > 0] = 1  # to avoid overflow
        ground_mask = self.smooth(ground_mask[1:-1, 1:-1])
        #cv2.imwrite('img.png', img_tmp)
        #cv2.imwrite('msk.png', ground_mask * 255)

        return image, mask, ground_mask,  meta

    def flood_fill(self, im, seed_point):
        """
        loDiff (Lower Difference)

            Definition: loDiff specifies the maximum lower brightness/color difference between the currently observed pixel and one of its neighbors belonging to the component, or the seed pixel being added to the component.

            Interpretation: If the difference between the current pixel and its neighbor is less than or equal to loDiff, the neighbor is considered part of the same component.

        upDiff (Upper Difference)

            Definition: upDiff specifies the maximum upper brightness/color difference between the currently observed pixel and one of its neighbors belonging to the component, or the seed pixel being added to the component.

            Interpretation: If the difference between the current pixel and its neighbor is less than or equal to upDiff, the neighbor is considered part of the same component.

        """
        image = im
        h, w = image.shape[:2]
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        fill_color = 255
        flags = cv2.FLOODFILL_FIXED_RANGE
        cv2.floodFill(image, mask, seed_point, fill_color, loDiff=0.02, upDiff=0.02, flags=flags )
        return image, mask

    def smooth(self, mask):
        kernel = np.ones((3, 3), np.float32)
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  kernel, iterations=2)
        return closed_mask

