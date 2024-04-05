#!/usr/bin/python3

#######################
### IMPORT PACKAGES ###
#######################
import argparse
from torchsummary import summary
from itertools import chain
import torch
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from model import Generator
from model import Discriminator
from utility import Buffer
from utility import Logging
from datasetup import CT2MRDataset


class Model():
    def __init__(self, options):
        self.options = options
        self.learning_rate = 0.0002
        self.input_output_channel = 1
        self.gen_a2b = Generator(self.input_output_channel)
        self.gen_b2a = Generator(self.input_output_channel)
        self.Dis_A = Discriminator(self.input_output_channel)
        self.Dis_B = Discriminator(self.input_output_channel)
        #making sure it works on gpu computation
        self.gen_a2b.cuda()
        self.gen_b2a.cuda()
        self.Dis_A.cuda()
        self.Dis_B.cuda()
        # print(summary(self.gen_a2b,(1, 256, 256)))
        # print(summary(self.Dis_A, (1,256,256)))
        self.gan_loss = torch.nn.MSELoss()
        self.cycle_loss = torch.nn.HuberLoss()
        self.identity_loss = torch.nn.HuberLoss()

        # Defining the optimizers that we will be using for the whole model itself for both generators and discriminators. ####
        self.Gen_optimizer = torch.optim.Adam(chain(self.gen_a2b.parameters(), self.gen_b2a.parameters()),
                                        lr=self.learning_rate, betas=(0.5, 0.999))
        self.Dis_optimizer_A = torch.optim.Adam(self.Dis_A.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.Dis_optimizer_B = torch.optim.Adam(self.Dis_B.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

    #initializing the weights of the diffeerent layers for both the convolution layers and the batch norm layers.
    def weights_initialization(self, layer):
        classname = layer.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(layer.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(layer.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(layer.bias.data, 0.0)

    def train(self, data_gen):
        self.gen_a2b.apply(self.weights_initialization)
        self.gen_b2a.apply(self.weights_initialization)
        self.Dis_A.apply(self.weights_initialization)
        self.Dis_B.apply(self.weights_initialization)

        #if self.options.cuda else torch.Tensor
        Tensor = torch.cuda.FloatTensor 
        input_CT = Tensor(self.options.batchSize, 1, 256, 256)
        input_MR = Tensor(self.options.batchSize, 1, 256, 256)
        target_real = Variable(Tensor(self.options.batchSize).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(self.options.batchSize).fill_(0.0), requires_grad=False)
        #Storing the generated images. 
        buffer_a = Buffer()
        buffer_b = Buffer()
        log_loss = Logging(self.options.num_epochs, len(data_gen))
        for epoch in range(self.options.num_epochs):
            for i, batch in enumerate(data_gen):
                # A is represented as the CT and B is represented as the MR result
                # Set model input by copying into the cuda tensors
                true_CT = input_CT.copy_(batch['CT'])
                true_MR = input_MR.copy_(batch['MR'])

                # Generating the fake images from the true CT and MR images. 
                fake_B = self.gen_a2b(true_CT)
                fake_A = self.gen_b2a(true_MR)

                # Generators that represent the CT to MRI conversion and the MRI to CT conversion. 
                self.Gen_optimizer.zero_grad()
                # GAN loss
                loss_GAN_A2B = self.gan_loss(self.Dis_B(fake_B), target_real)
                loss_GAN_B2A = self.gan_loss(self.Dis_A(fake_A), target_real)
                # Cycle loss
                loss_cycle_ABA = self.cycle_loss(self.gen_b2a(fake_B), true_CT)
                loss_cycle_BAB = self.cycle_loss(self.gen_a2b(fake_A), true_MR)
                # Identity loss
                identity_loss_B = self.identity_loss(self.gen_a2b(true_MR), true_MR)
                identity_loss_A = self.identity_loss(self.gen_b2a(true_CT), true_CT)
                # Total loss
                loss_G = identity_loss_A*5.0 + identity_loss_B*5.0 + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA*10.0 + loss_cycle_BAB*10.0
                loss_G.backward()
                
                self.Gen_optimizer.step()

                # Discriminator A start
                self.Dis_optimizer_A.zero_grad()

                # calcuatting the real loss
                loss_D_real = self.gan_loss(self.Dis_A(true_CT), target_real)

                # calculating the fake loss. 
                fake_A = buffer_a.retrieve(fake_A)
                loss_D_fake = self.gan_loss(self.Dis_A(fake_A.detach()), target_fake)

                # calculating the total loss. 
                loss_D_A = (loss_D_real + loss_D_fake)*0.5
                loss_D_A.backward()

                self.Dis_optimizer_A.step()
                ###################################

                # Discriminator B portion of the training
                self.Dis_optimizer_B.zero_grad()

                # Calculating the real loss. 
                loss_D_real = self.gan_loss(self.Dis_B(true_MR), target_real)
                
                # Fake loss calculation
                fake_B = buffer_b.retrieve(fake_B)
                loss_D_fake = self.gan_loss(self.Dis_B(fake_B.detach()), target_fake)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake)*0.5
                loss_D_B.backward()

                self.Dis_optimizer_B.step()

                log_loss.log(epoch + 1, i+1, './output.txt', {'loss_G': loss_G.item(), 'loss_G_identity': (identity_loss_A.item() + identity_loss_B.item()), 'loss_G_GAN': (loss_GAN_A2B.item() + loss_GAN_B2A.item()),
                            'loss_G_cycle': (loss_cycle_ABA.item() + loss_cycle_BAB.item()), 'loss_D': (loss_D_A.item() + loss_D_B.item())}) 

            # Save models checkpoints for loading in testing state and potential checkpoint training. 
            torch.save(self.gen_a2b.state_dict(), '../gen_a2b.pth')
            torch.save(self.gen_b2a.state_dict(), '../gen_b2a.pth')
            torch.save(self.Dis_A.state_dict(), '../discriminator_a.pth')
            torch.save(self.Dis_B.state_dict(), '../discriminator_b.pth')

    def test(self, data_gen):
        #if self.options.cuda else torch.Tensor
        #Making the tensors for the input values being stored
        Tensor = torch.cuda.FloatTensor 
        input_CT = Tensor(self.options.batchSize, self.input_output_channel, self.options.size, self.options.size)
        input_MR = Tensor(self.options.batchSize, self.input_output_channel, self.options.size, self.options.size)

        # Load the states for the generators from the checkpoints so we can run the model on the test data. 
        self.gen_a2b.load_state_dict(torch.load(self.options.gen_a2b))
        self.gen_b2a.load_state_dict(torch.load(self.options.gen_b2a))

        # Setting the models to evaluation mode so that we can do testing. 
        self.gen_a2b.eval()
        self.gen_b2a.eval()

        # Create the output dirs for where the testing images will be stored if they don't exist
        if not os.path.exists(self.options.base_path +'C2M_out/output/CT'):
            os.makedirs(self.options.base_path + 'C2M_out/output/CT')
        if not os.path.exists(self.options.base_path+'C2M_out/output/MRI'):
            os.makedirs(self.options.base_path + 'C2M_out/output/MRI')
        if not os.path.exists(self.options.base_path+'C2M_out/output/remadeCT'):
            os.makedirs(self.options.base_path + 'C2M_out/output/remadeCT')
        for i, batch in enumerate(data_gen):
            real_A = input_CT.copy_(batch['CT'])
            real_B = input_MR.copy_(batch['MR'])

            # generating the output images
            fake_B = 0.5*(self.gen_a2b(real_A).data + 1.0)
            fake_A = 0.5*(self.gen_b2a(real_B).data + 1.0)
            remade_ct = 0.5*(self.gen_b2a(fake_B).data + 1.0)

            # saving the image files. 
            for x in range(self.options.batchSize):
                print((i)*4 + x)
                save_image(fake_A[x], self.options.base_path +'C2M_out/output/CT/%04d.png' % ((i)*4 + x))
                save_image(fake_B[x], self.options.base_path +'C2M_out/output/MRI/%04d.png' % ((i)*4 + x))
                save_image(remade_ct[x], self.options.base_path +'C2M_out/output/remadeCT/%04d.png' % ((i)*4 + x))




if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs we will be training for')
    arg_parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
    arg_parser.add_argument('--dataroot', type=str, default='/home/nasheath_ahmed/C2MSample/', help='root directory of the dataset')
    arg_parser.add_argument('--mode', type=str, default='train', help='train, validate, or test')
    arg_parser.add_argument('--size', type=int, default=256, help='size of the images for the model')
    arg_parser.add_argument('--cuda', action='store_true', help='use the GPU')
    arg_parser.add_argument('--gen_a2b', type=str, default='./A2B_new_model.pth', help='A2B generator checkpoint file')
    arg_parser.add_argument('--gen_b2a', type=str, default='./B2A_new_model.pth', help='B2A generator checkpoint file')
    arg_parser.add_argument('--base_path', type=str, default='/home/nasheath_ahmed/', help='B2A generator checkpoint file')
    options = arg_parser.parse_args()

    model = Model(options)

    if options.mode  == 'train':
        data_gen = DataLoader(
            CT2MRDataset(options.dataroot,
            transformations = [
            transforms.Resize((int(options.size), int(options.size))), 
            transforms.RandomResizedCrop(256), 
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()]),
            batch_size=options.batchSize, shuffle=True, num_workers=4)

        model.train(data_gen)
    if options.mode == 'test':
        data_gen = DataLoader(CT2MRDataset(options.dataroot, transformations = [ 
            transforms.Resize((int(options.size), int(options.size))),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
            ], mode='test'),
            batch_size=options.batchSize,
            shuffle=False,
            num_workers=4)

        model.test(data_gen)
    
