import torch
from math import sqrt as sqrt
from itertools import product as product

# ssd300 config
voc = {
    'num_classes': 21,
    'min_dim': 300,  # image Size
    
    #grid sizes for the 6 Stations
    'feature_maps': [38, 19, 10, 5, 3, 1],
    
    # s_k (in pixels)
    'min_sizes': [30, 60, 111, 162, 213, 264], 
    
    # s'_k (in pixels)
    'max_sizes': [60, 111, 162, 213, 264, 315],
    
    #  size of one grid cell in original image pixels
    'steps': [8, 16, 32, 64, 100, 300],
    
    # aspect Ratios: Which shapes to generate at each station
    # (2 implies creating both 2:1 and 1:2)
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    
    'clip': True,
}

class PriorBox(object):
    def __init__(self, cfg):
        self.image_size = cfg['min_dim']
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratio = cfg['aspect_ratios']
        self.clip = cfg['clip']
    
    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            
            for i, j in product(range(f), range(f)):
                #center point
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                
                #width and height of SQUARE BOX
                s_k = self.min_sizes[k]
                s_k_ratio = s_k / self.image_size
                
                mean += [cx, cy, s_k_ratio, s_k_ratio]
                
                # width and height of MIDDLE SQUARE BOX
                s_k_prime = sqrt(self.min_sizes[k] * self.max_sizes[k])
                s_k_prime_ratio = s_k_prime / self.image_size

                
                mean += [cx, cy, s_k_prime_ratio, s_k_prime_ratio]
                
                # rectangular boxes
                for ar in self.aspect_ratio[k]:
                    #fat box
                    w = s_k * sqrt(ar)
                    h = s_k / sqrt(ar)
                    w_ratio = w / self.image_size
                    h_ratio = h / self.image_size
                    mean += [cx, cy, w_ratio, h_ratio]
                    
                    #tall box
                    w = s_k / sqrt(ar)
                    h = s_k * sqrt(ar)
                    w_ratio = w / self.image_size
                    h_ratio = h / self.image_size
                    mean += [cx, cy, w_ratio, h_ratio]
        
        output = torch.tensor(mean).view(-1, 4)
        
        if self.clip:
            output = output.clamp(min=0, max=1)
        
        return output
                

import numpy as np
def center_to_bottom_left(rectangles: np.array, image_size: float) -> np.array:
    """
        args:
            rectangles: 2D numpy array (cx, cy, w_ratio, h_ratio), ratio relative to image_size
            image_size: size of square image (float)
        return np.array of shape(rectangles[0], 4), (x_bottom_left, y_bottom_left, width, heigth)
    """
    w = rectangles[:, 2:3]
    h = rectangles[:, 3:]
    x_bl = (rectangles[:, 0:1] - w / 2) * image_size
    y_bl = (rectangles[:, 1:2] + h / 2) * image_size
    width = w * image_size
    height = h * image_size
    return np.concatenate((x_bl, y_bl, width, height), axis=1)
    