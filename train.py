# from __future__ import print_function
# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torchvision
import torchvision.utils
from torch.autograd import Variable
import torch.nn as nn
import sys, os, time
sys.path.append('utils')
sys.path.append('models')
from utils.data import CelebA, RandomNoiseGenerator, Cityscape_img, Cityscape_label
from utils import helper
from models.model import Generator, Discriminator, Encoder
import argparse
import numpy as np
from scipy.misc import imsave
from utils.logger import Logger


class PGGAN():
    def __init__(self, G, D, E, data, noise, opts):
        self.G = G
        self.D = D
        self.E = E
        self.data = data
        self.noise = noise
        self.opts = opts
        self.current_time = time.strftime('%Y-%m-%d %H%M%S')
        self.logger = Logger('./logs/' + self.current_time + "/")
        gpu = self.opts['gpu']
        self.use_cuda = len(gpu) > 0
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        current_time = time.strftime('%Y-%m-%d %H%M%S')
        self.opts['sample_dir'] = os.path.join(os.path.join(self.opts['exp_dir'], current_time), 'samples')
        self.opts['ckpt_dir'] = os.path.join(os.path.join(self.opts['exp_dir'], current_time), 'ckpts')
        os.makedirs(self.opts['sample_dir'])
        os.makedirs(self.opts['ckpt_dir'])

        self.bs_map = {2**R: self.get_bs(2**R) for R in range(2, 11)}
        self.rows_map = {32: 8, 16: 4, 8: 4, 4: 2, 2: 2}
        self.alpha = 0.9

        # save opts
        with open(os.path.join(os.path.join(self.opts['exp_dir'], current_time), 'options.txt'), 'w') as f:
            for k, v in self.opts.items():
                print('%s: %s' % (k, v), file=f)
            print('batch_size_map: %s' % self.bs_map, file=f)

    def get_bs(self, resolution):
        R = int(np.log2(resolution))
        if R < 7:
            bs = 32 / 2**(max(0, R-4))
        else:
            bs = 8 / 2**(min(2, R-7))
        return int(bs)

    def register_on_gpu(self):
        if self.use_cuda:
            self.E.cuda()
            self.G.cuda()
            self.D.cuda()

    def create_optimizer(self):
        self.optim_G = optim.Adam(self.G.parameters(), lr=self.opts['g_lr_max'], betas=(self.opts['beta1'], self.opts['beta2']))
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.opts['d_lr_max'], betas=(self.opts['beta1'], self.opts['beta2']))

    def create_criterion(self):
        # w is for gan
        if self.opts['gan'] == 'lsgan':
            self.adv_criterion = lambda p,t,w: torch.mean((p-t)**2)  # sigmoid is applied here
        elif self.opts['gan'] == 'wgan_gp':
            self.adv_criterion = lambda p,t,w: (-2*t+1) * torch.mean(p)
        elif self.opts['gan'] == 'gan':
            self.adv_criterion = lambda p,t,w: -w*(torch.mean(t*torch.log(p+1e-8)) + torch.mean((1-t)*torch.log(1-p+1e-8)))
        else:
            raise ValueError('Invalid/Unsupported GAN: %s.' % self.opts['gan'])

    def compute_adv_loss(self, prediction, target, w):
        return self.adv_criterion(prediction, target, w)

    def compute_additional_g_loss(self, image1, image2):
        criterionMSE = nn.MSELoss()
        criterionMSE.cuda()
        l2_loss = criterionMSE(image1, image2)
        return l2_loss

    def compute_additional_d_loss(self):  # drifting loss and gradient penalty, weighting inside this function
        return 0.0

    def _get_data(self, d):
        return d.data[0] if isinstance(d, Variable) else d

    def compute_G_loss(self):
        # print(self.d_fake.view(-1))
        g_adv_loss = self.compute_adv_loss(self.d_fake, True, 1)
        g_l2_loss = self.compute_additional_g_loss(self.hole_real, self.hole_fake)
        self.g_adv_loss = self._get_data(g_adv_loss)
        self.g_l2_loss = self._get_data(g_l2_loss)
        return (1-self.alpha)*g_adv_loss + self.alpha*g_l2_loss

    def compute_D_loss(self):
        self.d_adv_loss_real = self.compute_adv_loss(self.d_real, True, 0.5)
        self.d_adv_loss_fake = self.compute_adv_loss(self.d_fake, False, 0.5) * 0.1
        d_adv_loss = self.d_adv_loss_real + self.d_adv_loss_fake
        # d_add_loss = self.compute_additional_d_loss()
        self.d_adv_loss = self._get_data(d_adv_loss)
        # self.d_add_loss = self._get_data(d_add_loss)

        return d_adv_loss

    def postprocess(self):
        # TODO: weight cliping or others
        pass

    def _numpy2var(self, x):
        var = Variable(torch.from_numpy(x).float())
        # var = Variable(torchvision.transforms.ToTensor(x))
        if self.use_cuda:
            var = var.cuda()
        return var

    def _var2numpy(self, var):
        if self.use_cuda:
            return var.cpu().data.numpy()
        return var.data.numpy()

    def add_noise(self, x):
        # TODO: support more method of adding noise.
        if self.opts.get('no_noise', False):
            return x

        if hasattr(self, '_d_'):
            self._d_ = self._d_ * 0.9 + torch.mean(self.d_real).data[0] * 0.1
        else:
            self._d_ = 0.0
        strength = 0.2 * max(0, self._d_ - 0.5)**2
        noise = self._numpy2var(np.random.randn(*x.size()).astype(np.float32) * strength)
        return x + noise

    def preprocess(self, hole_image, real, hole_real):
        self.hole_image = self._numpy2var(hole_image)
        self.real = self._numpy2var(real)
        # self.hole_fake = self._numpy2var(hole_fake)
        self.hole_real = self._numpy2var(hole_real)
        # print (torch.min(self.real.data), torch.max(self.real.data))

    def forward_G(self, cur_level):
        self.d_fake = self.D(self.fake, cur_level=cur_level,starth=self.starth, startw=self.startw, hole_h=self.hole_h, hole_w=self.hole_w)

    def forward_D(self, cur_level, detach=True):
        self.encoded_feature = self.E(self.hole_image, cur_level = cur_level)
        self.fake = self.G(self.encoded_feature, cur_level=cur_level)
        # self.d_real = self.D(self.add_noise(self.real), cur_level=cur_level)
        self.d_real = self.D(self.real, cur_level=cur_level,starth=self.starth, startw=self.startw, hole_h=self.hole_h, hole_w=self.hole_w)
        self.d_fake = self.D(self.fake.detach() if detach else self.fake, cur_level=cur_level,starth=self.starth, startw=self.startw, hole_h=self.hole_h, hole_w=self.hole_w)

        hole_fake = self.fake.data[:,:,self.starth:self.starth+self.hole_h,self.startw:self.startw + self.hole_w]
        self.hole_fake = Variable(hole_fake.cuda())
        # print('d_real', self.d_real.view(-1))
        # print('d_fake', self.d_fake.view(-1))
        # print(self.fake[0].view(-1))
        # stop

    def backward_G(self):
        g_loss = self.compute_G_loss()
        g_loss.backward()
        self.optim_G.step()
        self.g_loss = self._get_data(g_loss)

    def backward_D(self, retain_graph=False):
        d_loss = self.compute_D_loss()
        d_loss.backward(retain_graph=retain_graph)
        self.optim_D.step()
        self.d_loss = self._get_data(d_loss)

    def report(self, it, num_it, phase, resol):
        formation = 'Iter[%d|%d], %s, %s, G: %.3f, D: %.3f, G_adv: %.3f, G_add: %.3f, D_adv: %.3f'
        values = (it, num_it, phase, resol, self.g_loss, self.d_loss, self.g_adv_loss, self.g_l2_loss, self.d_adv_loss)
        print(formation % values)

    def tensorboard(self, it, num_it, phase, resol, samples):
        # (1) Log the scalar values
        prefix = str(resol)+'/'+phase+'/'
        info = {prefix + 'G_loss': self.g_loss,
                prefix + 'G_adv_loss': self.g_adv_loss,
                prefix + 'g_l2_loss': self.g_l2_loss,
                prefix + 'D_loss': self.d_loss,
                prefix + 'D_adv_loss': self.d_adv_loss,
                prefix + 'D_adv_loss_fake': self._get_data(self.d_adv_loss_fake),
                prefix + 'D_adv_loss_real': self._get_data(self.d_adv_loss_real)}

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, it)

        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in self.G.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary('G/' + prefix +tag, self._var2numpy(value), it)
            if value.grad is not None:
                self.logger.histo_summary('G/' + prefix +tag + '/grad', self._var2numpy(value.grad), it)

        for tag, value in self.D.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary('D/' + prefix + tag, self._var2numpy(value), it)
            if value.grad is not None:
                self.logger.histo_summary('D/' + prefix + tag + '/grad',
                                          self._var2numpy(value.grad), it)

        # (3) Log the images
        # info = {'images': samples[:10]}
        # for tag, images in info.items():
        #     logger.image_summary(tag, images, it)

    def train(self):
        # prepare
        self.create_optimizer()
        self.create_criterion()
        self.register_on_gpu()

        to_level = int(np.log2(self.opts['target_resol']))
        from_level = int(np.log2(self.opts['first_resol']))
        assert 2**to_level == self.opts['target_resol'] and 2**from_level == self.opts['first_resol'] and to_level >= from_level >= 2
        cur_level = from_level # cur_level = 3

        for R in range(from_level-1, to_level-1):
            batch_size = self.bs_map[2 ** (R+1)]
            train_kimg = int(self.opts['train_kimg'] * 1000)
            transition_kimg = int(self.opts['transition_kimg'] * 1000)
            if R == to_level-1:
                transition_kimg = 0
            cur_nimg = 0
            _len = len(str(train_kimg + transition_kimg))
            _num_it = (train_kimg + transition_kimg) // batch_size
            for it in range(_num_it):
                # determined current level: int for stabilizing and float for fading in
                cur_level = R + float(max(cur_nimg-train_kimg, 0)) / transition_kimg
                cur_resol = 2 ** int(np.ceil(cur_level+1))
                phase = 'stabilize' if int(cur_level) == cur_level else 'fade_in'

                # get a batch noise and real images
                x = self.data(batch_size, cur_resol, cur_level)
                x = x[:,:,0:cur_resol//2,:]
                ins = np.zeros((batch_size, 35, x.shape[2], x.shape[3]))
                for b in range(batch_size):
                    for h in range(x.shape[2]):
                        for w in range(x.shape[3]):
                            channel = x[b,0,h,w]
                            ins[b,int(channel),h,w] = 1
                hole_image = ins.copy()
                # self.hole_h = self.hole_w = cur_resol // 2 - cur_resol // self.opts['first_resol']
                # self.starth = int((cur_resol - self.hole_h) / 2)
                # self.startw = int((cur_resol - self.hole_h) / 2)
                # self.hole_h = self.hole_w = cur_resol // 4 #- cur_resol // self.opts['first_resol']
                self.hole_h = self.hole_w = 3 * (cur_resol // 16) #- cur_resol // self.opts['first_resol']
                self.starth = int((cur_resol // 2 - self.hole_h) / 2)
                self.startw = int((cur_resol - self.hole_h) / 2)
                hole_real = hole_image[:,:,self.starth:self.starth+self.hole_h,self.startw:self.startw + self.hole_w].copy()
                hole_image[:,:,self.starth:self.starth+self.hole_h,self.startw:self.startw + self.hole_w] = 0

                # preprocess
                self.preprocess(hole_image, ins, hole_real)

                # update D
                self.optim_D.zero_grad()
                self.forward_D(cur_level, detach=True)  # TODO: feed gdrop_strength
                self.backward_D()
                # update G
                self.optim_G.zero_grad()
                self.forward_G(cur_level)
                self.backward_G()

                # report
                self.report(it, _num_it, phase, cur_resol)

                cur_nimg += batch_size

                # sampling
                samples = []
                if (it % self.opts['sample_freq'] == 0) or it == _num_it-1:
                    output = torch.FloatTensor(35, 1, self.real.size(2), self.real.size(3))
                    layer_rep = self.real[0].cpu().data
                    real = layer_rep.clone()
                    tmp = real.numpy()
                    tmp = tmp[:,np.newaxis,:,:]
                    real = torch.from_numpy(tmp)

                    fake = self.fake[0].cpu().data
                    layer_rep[:,self.starth:self.starth+self.hole_h,self.startw:self.startw + self.hole_w] = fake[:,self.starth:self.starth+self.hole_h,self.startw:self.startw + self.hole_w]
                    tmp = layer_rep.numpy()
                    tmp = tmp[:,np.newaxis,:,:]
                    layer_rep = torch.from_numpy(tmp)
                    torchvision.utils.save_image(layer_rep, os.path.join(self.opts['sample_dir'],
                                        '%dx%d-%s-%s_layer.png' % (cur_resol, cur_resol, phase, str(it).zfill(6))))
                    torchvision.utils.save_image(real, os.path.join(self.opts['sample_dir'],
                                        '%dx%d-%s-%s_real.png' % (cur_resol, cur_resol, phase, str(it).zfill(6))))

                    samples = self.sample()
                    imsave(os.path.join(self.opts['sample_dir'],
                                        '%dx%d-%s-%s.png' % (cur_resol, cur_resol, phase, str(it).zfill(6))), samples)

                # ===tensorboard visualization===
                if (it % self.opts['sample_freq'] == 0) or it == _num_it - 1:
                    self.tensorboard(it, _num_it, phase, cur_resol, samples)

                # save model
                # if (it % self.opts['save_freq'] == 0 and it > 0) or it == _num_it-1:
                #     self.save(os.path.join(self.opts['ckpt_dir'], '%dx%d-%s-%s' % (cur_resol, cur_resol, phase, str(it).zfill(6))))

    def sample(self):
        batch_size = self.hole_image.size(0)
        n_row = self.rows_map[batch_size]
        n_col = int(np.ceil(batch_size / float(n_row)))
        samples = []
        i = j = 0
        for row in range(n_row):
            one_row = []
            # fake
            for col in range(n_col):
                fake = self.fake[i].cpu().data
                fake[fake<=0] = 0 #clipping negative
                hole_img = self.hole_image[i].cpu().data
                hole_img[:,self.starth:self.starth+self.hole_h,self.startw:self.startw + self.hole_w] = fake[:,self.starth:self.starth+self.hole_h,self.startw:self.startw + self.hole_w]
                hole_img = np.argmax(hole_img, axis=0)
                hole_img = hole_img[np.newaxis,:]
                fake_color = helper.tensor2label(hole_img, 35)
                fake_color = fake_color.transpose([2,0,1])
                # print (np.min(hole_img), np.max(hole_img))
                # print (np.min(fake), np.max(fake))
                # print (fake[:,self.starth:self.starth+self.hole_h,self.startw:self.startw + self.hole_w])
                one_row.append(fake_color)
                # one_row.append(self.fake[i].cpu().data.numpy())
                i += 1
            # real
            for col in range(n_col):
                real = self.hole_image[j].cpu().data
                real = np.argmax(real, axis=0)
                real = real[np.newaxis,:]
                real_color = helper.tensor2label(real, 35)
                real_color = real_color.transpose([2,0,1])
                one_row.append(real_color)
                j += 1
            samples += [np.concatenate(one_row, axis=2)]
        samples = np.concatenate(samples, axis=1).transpose([1, 2, 0])

        half = samples.shape[1] // 2
        # samples[:,:half,:] = samples[:,:half,:] - np.min(samples[:,:half,:])
        # samples[:,:half,:] = samples[:,:half,:] / np.max(samples[:,:half,:])
        # samples[:,half:,:] = samples[:,half:,:] - np.min(samples[:,half:,:])
        # samples[:,half:,:] = samples[:,half:,:] / np.max(samples[:,half:,:])
        return samples

    def save(self, file_name):
        g_file = file_name + '-G.pth'
        d_file = file_name + '-D.pth'
        torch.save(self.G.state_dict(), g_file)
        torch.save(self.D.state_dict(), d_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str, help='gpu(s) to use.')
    parser.add_argument('--train_kimg', default=600, type=float, help='# * 1000 real samples for each stabilizing training phase.')
    parser.add_argument('--transition_kimg', default=600, type=float, help='# * 1000 real samples for each fading in phase.')
    parser.add_argument('--g_lr_max', default=1e-3, type=float, help='Generator learning rate')
    parser.add_argument('--d_lr_max', default=1e-4, type=float, help='Discriminator learning rate')
    parser.add_argument('--beta1', default=0, type=float, help='beta1 for adam')
    parser.add_argument('--beta2', default=0.99, type=float, help='beta2 for adam')
    parser.add_argument('--gan', default='lsgan', type=str, help='model: lsgan/wgan_gp/gan, currently only support lsgan or gan with no_noise option.')
    parser.add_argument('--first_resol', default=16, type=int, help='first resolution')
    parser.add_argument('--target_resol', default=512, type=int, help='target resolution')
    parser.add_argument('--drift', default=1e-3, type=float, help='drift, only available for wgan_gp.')
    parser.add_argument('--sample_freq', default=600, type=int, help='sampling frequency.')
    parser.add_argument('--save_freq', default=5000, type=int, help='save model frequency.')
    parser.add_argument('--exp_dir', default='./exp', type=str, help='experiment dir.')
    parser.add_argument('--no_noise', action='store_true', help='do not add noise to real data.')

    # TODO: support conditional inputs

    args = parser.parse_args()
    opts = {k:v for k,v in args._get_kwargs()}

    latent_size = 512
    sigmoid_at_end = args.gan in ['lsgan', 'gan']

    E = Encoder(num_channels=35, resolution=args.target_resol, fmap_max=latent_size, fmap_base=8192, sigmoid_at_end=sigmoid_at_end)
    G = Generator(num_channels=35, latent_size=latent_size, resolution=args.target_resol, fmap_max=latent_size, fmap_base=8192, tanh_at_end=False)
    D = Discriminator(num_channels=35, resolution=args.target_resol, fmap_max=latent_size, fmap_base=8192, sigmoid_at_end=sigmoid_at_end)
    # print(E)
    # print(G)
    # print(D)
    # stop
    data = Cityscape_label()
    # data = CelebA()
    noise = RandomNoiseGenerator(latent_size, 'gaussian')
    pggan = PGGAN(G, D, E, data, noise, opts)
    pggan.train()
