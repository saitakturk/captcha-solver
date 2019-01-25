import time
import urllib3
import requests
from bs4 import BeautifulSoup

from keras.models import load_model
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time 
#%matplotlib inline


model = load_model('/home/sait/Pictures/data/captcha_solver.hdf5')
np.load('/home/sait/Pictures/data/captcha_class_labels.npy')
print("[#]Program loaded...")

def img_solver(self):
    img = np.asarray(Image.open('screenshot.jpg'))

    imgs = []
    imgs.append(img[:,8:23].astype(float)/255.0)  
    imgs.append(img[:,23:38].astype(float)/255.0) 
    imgs.append(img[:,43:58].astype(float)/255.0)
    imgs.append(img[:,60:75].astype(float)/255.0)
    imgs.append(img[:,78:93].astype(float)/255.0)

    imgs[0] = (np.expand_dims(imgs[0], axis=0))
    imgs[1] = (np.expand_dims(imgs[1], axis=0))
    imgs[2] = (np.expand_dims(imgs[2], axis=0))
    imgs[3] = (np.expand_dims(imgs[3], axis=0))
    imgs[4] = (np.expand_dims(imgs[4], axis=0))

    result = ''
    for i in range(5):
        result += class_labels[int(model.predict_classes(imgs[i]))]
    
    print(result)
    return result