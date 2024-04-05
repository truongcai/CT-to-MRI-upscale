import torch.nn as nn
import torch.nn.functional as F

# Generator
class Generator(nn.Module):
    def __init__(self, in_out_channel,res_block_count=9):
        super(Generator, self).__init__()
        #storing the values for the channel dimensions
        self.in_out_dim = 64
        self.output_dim_1 = 128
        self.input_dim_2 = 128
        self.output_dim_2 = 256
        self.residual_dim = 256
        self.upsample_in_dim_1 = 256
        self.upsample_out_dim_1 = 128
        self.upsample_in_dim_2 = 128
        self.upsample_out_dim_2 = 64
        #Initialize the first model layers and the downsampling layers with increasing filter channel values. 
        model_initialization_downsample = [
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(in_out_channel, self.in_out_dim, 7),
                    nn.BatchNorm2d(self.in_out_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.in_out_dim, self.output_dim_1, 3, stride=2, padding=1),
                    nn.BatchNorm2d(self.output_dim_1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.input_dim_2, self.output_dim_2, 3, stride=2, padding=1),
                    nn.BatchNorm2d(self.output_dim_2),
                    nn.ReLU(inplace=True)

                ]       
        #Making the layers for the residual block and repeating it multiple times as based on architecture. 
        model_res = []
        for x in range(res_block_count):
            model_res += [  nn.Sequential(* [   nn.ReflectionPad2d(1),
                                                nn.Conv2d(self.residual_dim, self.residual_dim, 3),
                                                nn.BatchNorm2d(self.residual_dim),
                                                nn.ReLU(inplace=True),
                                                nn.ReflectionPad2d(1),
                                                nn.Conv2d(self.residual_dim, self.residual_dim, 3),
                                                nn.BatchNorm2d(self.residual_dim)
                                            ])
                        ]
        model_res = nn.ModuleList(model_res)

        #Adding the upsample portion of model to recreate the original image.
        model_upsample = [
            nn.ConvTranspose2d(self.upsample_in_dim_1, self.upsample_out_dim_1, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.upsample_out_dim_1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.upsample_in_dim_2, self.upsample_out_dim_2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.upsample_out_dim_2),
            nn.ReLU(inplace=True)
        ]
        #Output
        model_output = []
        model_output += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(self.in_out_dim, in_out_channel, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model_initialization_downsample)
        self.model_res = model_res
        self.model_upsample = nn.Sequential(*model_upsample)
        self.model_output = nn.Sequential(*model_output)

    def forward(self, x):
        x = self.model(x)
        for block in self.model_res:
            x = block(x)+x
        x = self.model_upsample(x)
        x = self.model_output(x)
        return x

# The discriminator will classify the outputted images as part of the cycle gan implementation of this project approach. 
class Discriminator(nn.Module):
    def __init__(self, image):
        super(Discriminator, self).__init__()
        self.layer_1 = 64
        self.layer_2 = 128
        self.layer_3 = 256
        self.layer_4 = 512
        model = [   nn.Conv2d(image, self.layer_1, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(self.layer_1, self.layer_2, 4, stride=2, padding=1),
                    nn.BatchNorm2d(self.layer_2), 
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(self.layer_2, self.layer_3, 4, stride=2, padding=1),
                    nn.BatchNorm2d(self.layer_3), 
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(self.layer_3, self.layer_4, 4, padding=1),
                    nn.BatchNorm2d(self.layer_4), 
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(self.layer_4, image, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).reshape(-1)








