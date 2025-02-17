import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import torch.nn.functional as F
from . import networks as networks
import time


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)
        self.input_A_I = self.Tensor(nb, 1, size, size)

        self.max = networks.max_operation()
        self.edge = networks.edge_operation()

        self.net = networks.define_network(opt.name, self.gpu_ids)
        which_epoch=opt.which_epoch

        #self.load_network(self.net)
        #self.load_network(self.net,'G_A',which_epoch)
        self.load_network(self.net, 'net', which_epoch)

        self.net.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_img = input['input_img']
        input_A_gray = input['A_gray']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths']

    def predict(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_A_gray = Variable(self.input_A_gray)
        self.real_A_gray_o = self.real_A_gray
        self.real_A_I = Variable(self.input_A_I)
        self.real_img = Variable(self.input_img)

        self.max_out = self.max(self.real_A_gray_o)
        self.edge_out = self.edge(self.real_A_gray_o)
        self.gadience = self.max_out + self.edge_out
        #self.real_A_gray = torch.cat([self.gadience, self.real_A_gray], 1)

        start_time = time.time()
        self.input,self.fake_B, self.latent_real_A, self.gray,self.fea4,self.fea3,self.fea2,self.fea1= self.net.forward(self.real_img, self.real_A_gray)
        print("using {}".format(time.time() - start_time))

        #fake_B = util.tensor2im(self.fake_B.data)

        #latent_real_A= util.tensor2im(self.latent_real_A)
        #fea4 = util.tensor2im(self.fea4.data)
        #fea3 = util.tensor2im(self.fea3.data)
        #fea2 = util.tensor2im(self.fea2.data)
        #fea1 = util.tensor2im(self.fea1.data)
        gray = util.tensor2im(self.gray)

        #return OrderedDict([('fake_B', fake_B)]),OrderedDict([('latent_real_A', latent_real_A)]),OrderedDict([('fea4', fea4)]),OrderedDict([('fea3', fea3)]),OrderedDict([('fea2', fea2)]),OrderedDict([('fea1', fea1)])

       # return OrderedDict([('fake_B', fake_B)]),OrderedDict([('img_grid ',self.img_grid )])
        return OrderedDict([('gray', gray)])


    # get image paths
    def get_image_paths(self):
        return self.image_paths
