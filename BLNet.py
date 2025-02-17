import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(DehazeBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res




class CSDNet(nn.Module):
    def __init__(self):
        super(CSDNet, self).__init__()

        p = 1

        self.downsample_1 = nn.MaxPool2d(2)
        self.downsample_2 = nn.MaxPool2d(2)
        self.downsample_3 = nn.MaxPool2d(2)
        self.downsample_4 = nn.MaxPool2d(2)

        self.attention_conv1_1 = nn.Conv2d(4, 32, 3, padding=p)
        self.attention_LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn1_1 = nn.BatchNorm2d(32)
        self.attention_conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.attention_LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn1_2 = nn.BatchNorm2d(32)
        self.attention_max_pool1 = nn.MaxPool2d(2)

        self.attention_conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.attention_LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn2_1 = nn.BatchNorm2d(64)
        self.attention_conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.attention_LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn2_2 = nn.BatchNorm2d(64)
        self.attention_max_pool2 = nn.MaxPool2d(2)

        self.attention_conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.attention_LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn3_1 = nn.BatchNorm2d(128)
        self.attention_conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.attention_LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn3_2 = nn.BatchNorm2d(128)
        self.attention_max_pool3 = nn.MaxPool2d(2)

        self.attention_conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.attention_LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn4_1 = nn.BatchNorm2d(256)
        self.attention_conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.attention_LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn4_2 = nn.BatchNorm2d(256)
        self.attention_max_pool4 = nn.MaxPool2d(2)

        self.attention_conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.attention_LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn5_1 = nn.BatchNorm2d(512)

        self.block = DehazeBlock(default_conv, 512, 3)

        self.attention_conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.attention_LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn5_2 = nn.BatchNorm2d(512)

        # self.deconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.attention_deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.attention_conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.attention_LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn6_1 = nn.BatchNorm2d(256)
        self.attention_conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.attention_LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn6_2 = nn.BatchNorm2d(256)

        # self.deconv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.attention_deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.attention_conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.attention_LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn7_1 = nn.BatchNorm2d(128)
        self.attention_conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.attention_LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn7_2 = nn.BatchNorm2d(128)

        # self.deconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.attention_deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.attention_conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.attention_LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn8_1 = nn.BatchNorm2d(64)
        self.attention_conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.attention_LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn8_2 = nn.BatchNorm2d(64)

        # self.deconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.attention_deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.attention_conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.attention_LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.attention_bn9_1 = nn.BatchNorm2d(32)
        self.attention_conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.attention_LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.attention_conv10 = nn.Conv2d(32, 1, 1)
        self.attention_sigmoid_5 = nn.Sigmoid()
        self.attention_sigmoid_4 = nn.Sigmoid()
        self.attention_sigmoid_3 = nn.Sigmoid()
        self.attention_sigmoid_2 = nn.Sigmoid()
        self.attention_sigmoid_1 = nn.Sigmoid()
        self.attention_sigmoid = nn.Sigmoid()


        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.max_pool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.feature5 = nn.Conv2d(512, 3, 1, padding=p)

        #self.ca = CALayer(512)
        #self.palayer = PALayer(512)
        self.block = DehazeBlock(default_conv, 512, 3)

        # self.deconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6_1 = nn.BatchNorm2d(256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6_2 = nn.BatchNorm2d(256)
        self.feature4 = nn.Conv2d(256, 3, 1, padding=p)

        # self.deconv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_2 = nn.BatchNorm2d(128)
        self.feature3 = nn.Conv2d(128, 3, 1, padding=p)

        # self.deconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_1 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2 = nn.BatchNorm2d(64)
        self.feature2 = nn.Conv2d(64, 3, 1)

        # self.deconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)
        self.feature1 = nn.Conv2d(32, 3, 1)

        self.conv10 = nn.Conv2d(32, 3, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, gray):
        flag = 0
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            gray = avg(gray)
            flag = 1
            # pass
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        gray, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gray)
        gray_22 = self.downsample_1(gray)
        gray_33 = self.downsample_2(gray_22)
        gray_44 = self.downsample_3(gray_33)
        gray_55 = self.downsample_4(gray_44)

        #first
        x = self.attention_bn1_1(self.attention_LReLU1_1(self.attention_conv1_1(torch.cat((input,gray),1))))
        conv1 = self.attention_bn1_2(self.attention_LReLU1_2(self.attention_conv1_2(x)))
        x = self.attention_max_pool1(conv1)

        x = self.attention_bn2_1(self.attention_LReLU2_1(self.attention_conv2_1(x)))
        conv2 = self.attention_bn2_2(self.attention_LReLU2_2(self.attention_conv2_2(x)))
        x = self.attention_max_pool2(conv2)

        x = self.attention_bn3_1(self.attention_LReLU3_1(self.attention_conv3_1(x)))
        conv3 = self.attention_bn3_2(self.attention_LReLU3_2(self.attention_conv3_2(x)))
        x = self.attention_max_pool3(conv3)

        x = self.attention_bn4_1(self.attention_LReLU4_1(self.attention_conv4_1(x)))
        conv4 = self.attention_bn4_2(self.attention_LReLU4_2(self.attention_conv4_2(x)))
        x = self.attention_max_pool4(conv4)

        x = self.attention_bn5_1(self.attention_LReLU5_1(self.attention_conv5_1(x)))
        x1 = self.block(x)
        x2 = self.block(x1)
        x3 = self.block(x2)
        x4 = self.block(x3)
        x5 = self.block(x4)
        x = self.block(x5)
        x = x * gray_55
        conv5 = self.attention_bn5_2(self.attention_LReLU5_2(self.attention_conv5_2(x)))
        self.gray_5 = self.attention_sigmoid(conv5)  # 512 c
        #print(self.gray_5.shape)
        #gray_t=torch.transpose(self.gray_5,0,1)
        #print(gray_t.shape)
        #writer = SummaryWriter()
        #img_grid=writer.add_image('512',make_grid(gray_t,nrow=10))

        #for i in (1,512):
          #writer.add_image(f'{i}_feature_maps',img_grid,global_step=0)

        conv5 = F.upsample(conv5, scale_factor=2, mode='bilinear')
        conv4 = conv4 * gray_44
        up6 = torch.cat([self.attention_deconv5(conv5), conv4], 1)
        x = self.attention_bn6_1(self.attention_LReLU6_1(self.attention_conv6_1(up6)))
        conv6 = self.attention_bn6_2(self.attention_LReLU6_2(self.attention_conv6_2(x)))
        self.gray_4 = self.attention_sigmoid_4(conv6)  # 256 c

        conv6 = F.upsample(conv6, scale_factor=2, mode='bilinear')
        conv3 = conv3 * gray_33
        up7 = torch.cat([self.attention_deconv6(conv6), conv3], 1)
        x = self.attention_bn7_1(self.attention_LReLU7_1(self.attention_conv7_1(up7)))
        conv7 = self.attention_bn7_2(self.attention_LReLU7_2(self.attention_conv7_2(x)))
        self.gray_3 = self.attention_sigmoid_3(conv7)

        conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
        conv2 = conv2 * gray_22
        up8 = torch.cat([self.attention_deconv7(conv7), conv2], 1)
        x = self.attention_bn8_1(self.attention_LReLU8_1(self.attention_conv8_1(up8)))
        conv8 = self.attention_bn8_2(self.attention_LReLU8_2(self.attention_conv8_2(x)))
        self.gray_2 = self.attention_sigmoid_2(conv8)


        conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
        conv1 = conv1 * gray
        up9 = torch.cat([self.attention_deconv8(conv8), conv1], 1)
        x = self.attention_bn9_1(self.attention_LReLU9_1(self.attenti           on_conv9_1(up9)))
        conv9 = self.attention_LReLU9_2(self.attention_conv9_2(x))
        self.gray_1 = self.attention_sigmoid_1(conv9)


        #gray_t = torch.transpose(conv9, 0, 1)
        # print(gray_t.shape)
        #writer = SummaryWriter()
        #img_grid = writer.add_image('32', make_grid(gray_t, nrow=8))

        #print(self.gray_1.shape)
        #gray_t = torch.transpose(conv9, 0, 1)
        #print(gray_t.shape)
        # writer = SummaryWriter()
        # img_grid= writer.add_image('32',make_grid(gray_t,nrow=10))

        latent = self.attention_conv10(conv9)
        self.gray = self.attention_sigmoid(latent)
        print(self.gray.shape)



        #second
        x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
        conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
        x = self.max_pool1(conv1)

        x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
        x = self.max_pool2(conv2)

        x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
        x = self.max_pool3(conv3)

        x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
        x = self.max_pool4(conv4)

        x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))

        #x = self.ca(x)
        #x = self.palayer(x)
        x1 = self.block(x)
        x2 = self.block(x1)
        x3 = self.block(x2)
        x4 = self.block(x3)
        x5 = self.block(x4)
        x6 = self.block(x5)
        x = x6 / self.gray_5
        #fea5 = self.tanh(self.feature5(x))
        conv5 = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))
        conv5_ = self.LReLU5_2(self.conv5_2(x))
        fea4 = self.tanh(self.feature5(conv5_))
        conv5 = F.upsample(conv5, scale_factor=2, mode='bilinear')
        conv5 = self.deconv5(conv5)
        conv5 = conv5 / self.gray_4
        #fea4 =self.tanh( self.feature4(conv5))
        conv4 = conv4 / self.gray_4
        up6 = torch.cat([conv5, conv4], 1)
        x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
        conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))
        conv6_ = self.LReLU6_2(self.conv6_2(x))
        fea3 = self.tanh(self.feature4(conv6_))

        conv6 = F.upsample(conv6, scale_factor=2, mode='bilinear')
        conv6 = self.deconv6(conv6)
        conv6 = conv6 / self.gray_3
        #fea3 = self.tanh(self.feature3(conv6))
        conv3 = conv3 / self.gray_3
        up7 = torch.cat([conv6, conv3], 1)
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))
        conv7_ = self.LReLU7_2(self.conv7_2(x))
        fea2 = self.tanh(self.feature3(conv7_))

        conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
        conv7 = self.deconv7(conv7)

        conv7 = conv7 / self.gray_2
        #fea2 = self.tanh(self.feature2(conv7))
        #gray_t = torch.transpose(conv7, 0, 1)
        #writer = SummaryWriter()
        #img_grid = writer.add_image('64', make_grid(gray_t, nrow=8))
        conv2 = conv2 / self.gray_2
        up8 = torch.cat([conv7, conv2], 1)
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))
        conv8_ = self.LReLU8_2(self.conv8_2(x))
        fea1 = self.tanh(self.feature2(conv8_))

        conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
        conv8 = self.deconv8(conv8)
        conv8 = conv8 / self.gray_1

        conv1 = conv1 / self.gray_1
        up9 = torch.cat([conv8, conv1], 1)
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))



        latent = self.tanh(self.conv10(conv9))

        output = latent / self.gray

        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        gray = pad_tensor_back(self.gray, pad_left, pad_right, pad_top, pad_bottom)
        #fea1= pad_tensor_back(fea1, pad_left, pad_right, pad_top, pad_bottom)
        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
            gray = F.upsample(gray, scale_factor=2, mode='bilinear')
            #fea1  = F.upsample(fea1 , scale_factor=2, mode='bilinear')
        return input, output, latent, gray,fea4,fea3,fea2,fea1
            #, img_grid
