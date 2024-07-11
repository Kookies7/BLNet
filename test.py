import time
import os
import time
import torch
import ntpath
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
# from util.visualizer import Visualizer
# from pdb import set_trace as st
# from util import html
from models.test_model import TestModel
from PIL import Image
import numpy as np

opt = TestOptions().parse()
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = TestModel()
model.initialize(opt)
print("model [%s] was created" % (model.name()))
out_dir = os.path.join("./result/", opt.name)
out_dir5 = os.path.join("./result5/", opt.name)
out_dir4 = os.path.join("./result4/", opt.name)
out_dir3 = os.path.join("./result3/", opt.name)
out_dir2 = os.path.join("./result2/", opt.name)
out_dir1 = os.path.join("./result1/", opt.name)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
print('the number of test image:')
print(len(dataset))


def save_images(path, visuals, image_path):
    image_dir = path
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    for label, image_numpy in visuals.items():
        if label is 'gray' or 'fake_B'  or 'latent_real_A' or 'fea4' or 'fea3' or 'fea2' or 'fea1':
            image_name = '%s.png' % name
            #image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            print(image_numpy.shape)
            save_image(image_numpy, save_path)


def save_image(image_numpy, image_path):

    #image_numpy=(image_numpy[0]+1)/2.0*255.0
    #maxi=image_numpy.max()
    #image_numpy=image_numpy*255./maxi
    #image_numpy=image_numpy.astype(np.uint8)
    image_numpy=np.squeeze(image_numpy,axis=2)
    image_pil = Image.fromarray(image_numpy)
    #image_pil= image_pil.convert('L')
    image_pil.save(image_path)


with torch.no_grad():
    for i, data in enumerate(dataset):
        model.set_input(data)
        #visuals,latent_real_A,fea4,fea3,fea2,fea1 = model.predict()
        visuals = model.predict()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        save_images(out_dir,  visuals, img_path)
        #save_images(out_dir5, latent_real_A, img_path)
        #save_images(out_dir4, fea4, img_path)
        #save_images(out_dir3, fea3, img_path)
        #save_images(out_dir2, fea2, img_path)
        #save_images(out_dir1, fea1, img_path)


