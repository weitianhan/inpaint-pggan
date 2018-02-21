# -*- coding: utf-8 -*-
from models.base_model import *
conv_kernel_size = 3
padding_size = int(conv_kernel_size / 2)

def G_conv(incoming, in_channels, out_channels, kernel_size, padding, nonlinearity, init, param=None,
        to_sequential=True, use_wscale=True, use_batchnorm=False, use_pixelnorm=True):
    layers = incoming
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
    he_init(layers[-1], init, param)  # init layers
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    layers += [nonlinearity]
    if use_batchnorm:
        layers += [nn.BatchNorm2d(out_channels)]
    if use_pixelnorm:
        layers += [PixelNormLayer()]
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers

def E_conv(incoming, in_channels, out_channels, kernel_size, stride, padding, nonlinearity, init, param=None,
        to_sequential=True, use_wscale=True, use_gdrop=True, use_layernorm=False, gdrop_param=dict()):
    layers = incoming
    # if use_gdrop:
    #     layers += [GDropLayer(**gdrop_param)]
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
    he_init(layers[-1], init, param)  # init layers
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    layers += [nonlinearity]
    # if use_layernorm:
    #     layers += [LayerNormLayer()]  # TODO: requires incoming layer
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers

def NINLayer(incoming, in_channels, out_channels, nonlinearity, init, param=None,
            to_sequential=True, use_wscale=True):
    layers = incoming
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)]  # NINLayer in lasagne
    he_init(layers[-1], init, param)  # init layers
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    if not (nonlinearity == 'linear'):
        layers += [nonlinearity]
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers

class Encoder(nn.Module):
    def __init__(self,
                num_channels    = 1,        # Overridden based on dataset.
                resolution      = 32,       # Overridden based on dataset.
                label_size      = 0,        # Overridden based on dataset.
                fmap_base       = 4096,
                fmap_decay      = 1.0,
                fmap_max        = 256,
                mbstat_avg      = 'all',
                mbdisc_kernels  = None,
                use_wscale      = True,
                use_gdrop       = False,
                use_layernorm   = False,
                sigmoid_at_end  = False):
        super(Encoder, self).__init__()
        self.num_channels = num_channels
        self.resolution = resolution #256
        self.label_size = label_size
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max
        self.mbstat_avg = mbstat_avg
        self.mbdisc_kernels = mbdisc_kernels
        self.use_wscale = use_wscale
        self.use_gdrop = use_gdrop
        self.use_layernorm = use_layernorm
        self.sigmoid_at_end = sigmoid_at_end

        R = int(np.log2(resolution)) # r = 8
        assert resolution == 2**R and resolution >= 4
        gdrop_strength = 0.0

        negative_slope = 0.2
        act = nn.LeakyReLU(negative_slope=negative_slope)
        # input activation
        iact = 'leaky_relu'
        # output activation
        output_act = nn.Sigmoid() if self.sigmoid_at_end else 'linear'
        output_iact = 'sigmoid' if self.sigmoid_at_end else 'linear'
        gdrop_param = {'mode': 'prop', 'strength': gdrop_strength}

        nins = nn.ModuleList()
        lods = nn.ModuleList()
        pre = None
        # pre = nn.ModuleList()
        #
        # net = []
        # net = D_conv(net, 512, 512, 1, 0, act, iact, negative_slope, True,
        #             self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
        # pre.append(net)

        # nins.append(NINLayer([], self.num_channels, self.get_nf(R-1), act, iact, negative_slope, True, self.use_wscale))
        net = []
        ic, oc = self.get_nf(R), self.get_nf(R)
        net = D_conv(net, ic, oc, conv_kernel_size, 1, padding_size, act, iact, negative_slope, False,
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
        # net += [nn.MaxPool2d(2)]
        net = D_conv(net, oc, oc, conv_kernel_size, 2, padding_size, act, iact, negative_slope, False,
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
        lods.append(nn.Sequential(*net))
        nin = []
        nin = NINLayer(nin, self.num_channels, oc, act, iact, negative_slope, True, self.use_wscale)
        nins.append(nin)
        for I in range(R-1, 3, -1):
            ic, oc = self.get_nf(I+1), self.get_nf(I)
            net = []
            net = D_conv(net, ic, oc, conv_kernel_size, 1, padding_size, act, iact, negative_slope, False,
                        self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            net = D_conv(net, oc, oc, conv_kernel_size, 2, padding_size, act, iact, negative_slope, False,
                        self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            lods.append(nn.Sequential(*net))
            # nin = [nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
            nin = []
            nin = NINLayer(nin, self.num_channels, oc//2, act, iact, negative_slope, True, self.use_wscale)
            nins.append(nin)

        net = []
        lods.append(E_conv(net, self.get_nf(4), self.get_nf(4), conv_kernel_size, 1, padding_size, act, iact, negative_slope, True, self.use_wscale,self.use_gdrop, self.use_layernorm, gdrop_param))
        nin = []
        nin = NINLayer(nin, self.num_channels, self.get_nf(4), act, iact, negative_slope, True, self.use_wscale)
        nins.append(nin)
        # print (lods)
        # print (nins)
        # stop
        self.output_layer = ESelectLayer(pre, lods, nins)

    def get_nf(self, stage):
        return int(self.fmap_base / (2.0 ** (stage * self.fmap_decay)))

    def forward(self, x, y=None, cur_level=None, insert_y_at=None, gdrop_strength=0.0):
        for module in self.modules():
            if hasattr(module, 'strength'):
                module.strength = gdrop_strength
        return self.output_layer(x, y, cur_level, insert_y_at)

class Generator(nn.Module):
    def __init__(self,
                num_channels        = 1,        # Overridden based on dataset.
                resolution          = 32,       # Overridden based on dataset.
                label_size          = 0,        # Overridden based on dataset.
                fmap_base           = 4096,
                fmap_decay          = 1.0,
                fmap_max            = 256,
                latent_size         = None,
                normalize_latents   = False,
                use_wscale          = True,
                use_pixelnorm       = False,
                use_leakyrelu       = True,
                use_batchnorm       = False,
                tanh_at_end         = None):
        super(Generator, self).__init__()
        self.num_channels = num_channels
        self.resolution = resolution
        self.label_size = label_size
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max
        self.latent_size = latent_size
        self.normalize_latents = normalize_latents
        self.use_wscale = use_wscale
        self.use_pixelnorm = use_pixelnorm
        self.use_leakyrelu = use_leakyrelu
        self.use_batchnorm = use_batchnorm
        self.tanh_at_end = tanh_at_end

        R = int(np.log2(resolution)) # r = 8
        assert resolution == 2**R and resolution >= 4
        # if latent_size is None:
        #     latent_size = self.get_nf(0)

        negative_slope = 0.2
        act = nn.LeakyReLU(negative_slope=negative_slope) if self.use_leakyrelu else nn.ReLU()
        iact = 'leaky_relu' if self.use_leakyrelu else 'relu'
        output_act = nn.Tanh() if self.tanh_at_end else 'linear'
        output_iact = 'tanh' if self.tanh_at_end else 'linear'

        pre = None
        lods = nn.ModuleList()
        nins = nn.ModuleList()
        layers = []

        # if self.normalize_latents:
        #     pre = PixelNormLayer()
        #
        # if self.label_size:
        #     layers += [ConcatLayer()]

        # layers += [ReshapeLayer([latent_size, 1, 1])]
        # layers = G_conv(layers, self.get_nf(4), self.get_nf(4), 1, 1, act, iact, negative_slope,
        #             False, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
        net = G_conv(layers, self.get_nf(4), self.get_nf(4), 1, 0, act, iact, negative_slope,
                    True, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)  # first block

        lods.append(net)
        nins.append(NINLayer([], self.get_nf(4), self.num_channels, 'linear', iact, negative_slope, True, self.use_wscale))  # to_rgb layer

        for I in range(3, R):  # following blocks
            ic, oc = self.get_nf(I+1), self.get_nf(I+2)
            layers = [nn.Upsample(scale_factor=2, mode='nearest')]  # upsample
            # layers = G_conv(layers, ic, oc, 3, 1, act, iact, negative_slope, False, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
            net = G_conv(layers, ic, oc, conv_kernel_size, padding_size, act, iact, negative_slope, True, self.use_wscale, self.use_batchnorm, self.use_pixelnorm)
            lods.append(net)
            nins.append(NINLayer([], oc, self.num_channels, 'linear', iact, negative_slope, True, self.use_wscale))  # to_rgb layer

        # print (lods)
        # print (nins)
        # stop
        self.output_layer = GSelectLayer(pre, lods, nins)

    def get_nf(self, stage):
        return int(self.fmap_base / (2.0 ** (stage * self.fmap_decay)))

    def forward(self, x, y=None, cur_level=None, insert_y_at=None):
        return self.output_layer(x, y, cur_level, insert_y_at)


def D_conv(incoming, in_channels, out_channels, kernel_size, stride, padding, nonlinearity, init, param=None,
        to_sequential=True, use_wscale=True, use_gdrop=True, use_layernorm=False, gdrop_param=dict()):
    layers = incoming
    if use_gdrop:
        layers += [GDropLayer(**gdrop_param)]
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
    he_init(layers[-1], init, param)  # init layers
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    layers += [nonlinearity]
    if use_layernorm:
        layers += [LayerNormLayer()]  # TODO: requires incoming layer
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers


class Discriminator(nn.Module):
    def __init__(self,
                num_channels    = 1,        # Overridden based on dataset.
                resolution      = 32,       # Overridden based on dataset.
                label_size      = 0,        # Overridden based on dataset.
                fmap_base       = 4096,
                fmap_decay      = 1.0,
                fmap_max        = 256,
                mbstat_avg      = 'all',
                mbdisc_kernels  = None,
                use_wscale      = True,
                use_gdrop       = False,
                use_layernorm   = False,
                sigmoid_at_end  = False):
        super(Discriminator, self).__init__()
        self.num_channels = num_channels
        self.resolution = resolution
        self.label_size = label_size
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max
        self.mbstat_avg = mbstat_avg
        self.mbdisc_kernels = mbdisc_kernels
        self.use_wscale = use_wscale
        self.use_gdrop = use_gdrop
        self.use_layernorm = use_layernorm
        self.sigmoid_at_end = sigmoid_at_end
        self.final = nn.Linear(2048,1)

        R = int(np.log2(resolution))
        assert resolution == 2**R and resolution >= 4
        gdrop_strength = 0.0

        negative_slope = 0.2
        act = nn.LeakyReLU(negative_slope=negative_slope)
        # input activation
        iact = 'leaky_relu'
        # output activation
        output_act = nn.Sigmoid() if self.sigmoid_at_end else 'linear'
        output_iact = 'sigmoid' if self.sigmoid_at_end else 'linear'
        gdrop_param = {'mode': 'prop', 'strength': gdrop_strength}

        nins = nn.ModuleList()
        lods = nn.ModuleList()
        local_nins = nn.ModuleList()
        local_lods = nn.ModuleList()
        pre = None

        # global D
        net = []
        ic, oc = self.get_nf(R), self.get_nf(R)
        # net = D_conv(net, ic, oc, 3, 1, 1, act, iact, negative_slope, False,
        #             self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
        net = D_conv(net, oc, oc, conv_kernel_size, 2, padding_size, act, iact, negative_slope, False,
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
        lods.append(nn.Sequential(*net))
        nin = []
        nin = NINLayer(nin, self.num_channels, oc, act, iact, negative_slope, True, self.use_wscale)
        nins.append(nin)
        for I in range(R, 3, -1):
            ic, oc = self.get_nf(I+1), self.get_nf(I)
            net = []
            # net = D_conv(net, ic, oc, 3, 1, 1, act, iact, negative_slope, False,
            #             self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            net = D_conv(net, ic, oc, conv_kernel_size, 2, padding_size, act, iact, negative_slope, False,
                        self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            lods.append(nn.Sequential(*net))
            nin = []
            nin = NINLayer(nin, self.num_channels, ic, act, iact, negative_slope, True, self.use_wscale)
            nins.append(nin)

        net = []
        lods.append(E_conv(net, self.get_nf(4), self.get_nf(4), 1, 1, 0, act, iact, negative_slope, True, self.use_wscale,self.use_gdrop, self.use_layernorm, gdrop_param))
        nin = []
        nin = NINLayer(nin, self.num_channels, self.get_nf(4), act, iact, negative_slope, True, self.use_wscale)
        nins.append(nin)
        # print (lods)
        # print (nins)
        # stop
        self.global_output_layer = globalDSelectLayer(pre, lods, nins)

        # local D
        net = []
        ic, oc = self.get_nf(R), self.get_nf(R)
        # net = D_conv(net, ic, oc, 3, 1, 1, act, iact, negative_slope, False,
        #             self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
        net = D_conv(net, oc, oc, conv_kernel_size, 2, padding_size, act, iact, negative_slope, False,
                    self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
        local_lods.append(nn.Sequential(*net))
        nin = []
        nin = NINLayer(nin, self.num_channels, oc, act, iact, negative_slope, True, self.use_wscale)
        local_nins.append(nin)
        for I in range(R-1, 3, -1):
            ic, oc = self.get_nf(I+1), self.get_nf(I)
            net = []
            # net = D_conv(net, ic, oc, 3, 1, 1, act, iact, negative_slope, False,
            #             self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            net = D_conv(net, ic, oc, conv_kernel_size, 2, padding_size, act, iact, negative_slope, False,
                        self.use_wscale, self.use_gdrop, self.use_layernorm, gdrop_param)
            local_lods.append(nn.Sequential(*net))
            nin = []
            nin = NINLayer(nin, self.num_channels, oc//2, act, iact, negative_slope, True, self.use_wscale)
            local_nins.append(nin)

        net = []
        local_lods.append(E_conv(net, self.get_nf(4), self.get_nf(4), 1, 1, 0, act, iact, negative_slope, True, self.use_wscale,self.use_gdrop, self.use_layernorm, gdrop_param))
        nin = []
        nin = NINLayer(nin, self.num_channels, self.get_nf(4), act, iact, negative_slope, True, self.use_wscale)
        local_nins.append(nin)
        # print (local_lods)
        # print (local_nins)
        # stop
        self.local_output_layer = localDSelectLayer(pre, local_lods, local_nins)


    def get_nf(self, stage):
        result = int(self.fmap_base / (2.0 ** (stage * self.fmap_decay)))
        if result <= 16:
            result = 16
        return result

    def forward(self, x, y=None, cur_level=None, insert_y_at=None, gdrop_strength=0.0,starth=0, startw=0, hole_h=0, hole_w=0):
        for module in self.modules():
            if hasattr(module, 'strength'):
                module.strength = gdrop_strength
        hole = x[:,:,starth:starth+hole_h,startw:startw + hole_w]
        local_feature = self.local_output_layer(hole, y, cur_level, insert_y_at)
        global_feature = self.global_output_layer(x, y, cur_level, insert_y_at)
        concat_feature = torch.cat((global_feature,local_feature), 1)
        x = concat_feature.view(-1, 2048)
        x = self.final(x)
        return x

# class AutoencodingDiscriminator(nn.Module):
#     def __init__(self,
#                 num_channels    = 1,        # Overridden based on dataset.
#                 resolution      = 32,       # Overridden based on dataset.
#                 fmap_base       = 4096,
#                 fmap_decay      = 1.0,
#                 fmap_max        = 256,
#                 tanh_at_end     = False):
#         super(AutoencodingDiscriminator, self).__init__()
#         self.num_channels = num_channels
#         self.resolution = resolution
#         self.fmap_base = fmap_base
#         self.fmap_decay = fmap_decay
#         self.fmap_max = fmap_max
#         self.tanh_at_end = tanh_at_end

#         R = int(np.log2(resolution))
#         assert resolution == 2**R and resolution >= 4

#         negative_slope = 0.2
#         act = nn.LeakyReLU(negative_slope=negative_slope)
#         iact = 'leaky_relu'
#         output_act = nn.Tanh() if self.tanh_at_end else 'linear'
#         output_iact = 'tanh' if self.tanh_at_end else 'linear'

#         nins = nn.ModuleList()
#         lods = nn.ModuleList()
#         pre = None

#         for I in range(R, 1, -1):
#             ic, oc = self.get_nf(I), self.get_nf(I-1)
#             nins.append(NINLayer([], self.num_channels, ic, act, iact, negative_slope, True, True))  # from_rgb layer

#             net = [nn.Conv2d(ic, oc, 3, 1, 1), act]
#             net += [nn.BatchNorm2d(oc), nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)]
#             he_init(net[0], iact, negative_slope)
#             lods.append(nn.Sequential(*net))

#         for I in range(2, R+1):
#             ic, oc = self.get_nf(I-1), self.get_nf(I)
#             net = [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(ic, oc, 3, 1, 1), act, nn.BatchNorm2d(oc)]
#             he_init(net[1], iact, negative_slope)
#             lods.append(nn.Sequential(*net))
#             nins.append(NINLayer([], oc, self.num_channels, output_act, output_iact, None, True, True))  # to_rgb layer

#         self.output_layer = AEDSelectLayer(pre, lods, nins)

#     def get_nf(self, stage):
#         return min(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_max)

#     def forward(self, x, cur_level=None):
#         return self.output_layer(x, cur_level)
