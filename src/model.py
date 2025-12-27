import torch
import torch.nn as nn
vgg_config = [
    64, 64, 'M',
    128, 128, 'M',
    256, 256, 256, 'C',
    512, 512, 512, 'M',
    512, 512, 512,
]
def make_vgg_base(config=vgg_config, batch_norm=False):
    """
    parameters
        config: config of vgg model,
        batch_norm: True or False
    """
    layers = []
    in_channels=3
    for v in config:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v

    layers += [nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6), nn.ReLU(inplace=True)]

    layers += [nn.Conv2d(1024, 1024, kernel_size=1), nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)

def extractor_layers(config, i=1024, batch_norm=False):
    layers = []
    input_channels=i

    #hard code
    layers += [nn.Conv2d(input_channels, 256, kernel_size=1),]
    layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),]

    layers += [nn.Conv2d(512, 128, kernel_size=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    layers += [nn.Conv2d(256, 128, kernel_size=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3)]

    layers += [nn.Conv2d(256, 128, kernel_size=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3)]

    return nn.ModuleList(layers)

def make_multibox_layer(vgg, extra_layers, cfg, num_classes):

    """
    Creates the prediction heads (loc and conf layers) for SSD.

    Args:
        vgg: The base VGG network (ModuleList).
        extra_layers: The extra feature layers (ModuleList).
        cfg: List of number of boxes per layer (e.g., [4, 6, 6, 6, 4, 4]).
        num_classes: Number of classes (including background).
    """

    loc_layers = []
    conf_layers = []

    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):

        input_channels = vgg[v].out_channels
        boxes_per_loc = cfg[k]

        loc_layers += [nn.Conv2d(input_channels, 4 * boxes_per_loc, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(input_channels, num_classes * boxes_per_loc, kernel_size=3, padding=1)]

    all_conv = [x for x in extra_layers if isinstance(x, nn.Conv2d)]

    for k, v in enumerate(all_conv[1::2], 2):
        input_channels = v.out_channels
        boxes_per_loc = cfg[k]

        loc_layers += [nn.Conv2d(input_channels, 4 * boxes_per_loc, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(input_channels, num_classes * boxes_per_loc, kernel_size=3, padding=1)]

    return (nn.ModuleList(loc_layers), nn.ModuleList(conf_layers))


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or 10
        self.eps = 1e-10

        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):

        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps

        x = torch.div(x, norm)

        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x

        return out

import torch.nn.functional as F
class SSD(nn.Module):
    def __init__(self, phase, base, extras, head, num_classes):
        super().__init__()
        self.phase = phase
        self.vgg = base
        self.extras = extras #only conv layers not ReLU
        self.loc = head[0]
        self.conf = head[1]
        self.num_classes = num_classes

        self.l2_norm = L2Norm(512, 20)

        self.vgg_src = [21, -2]

    def forward(self, x):

        sources = [] # 6 features map here
        loc = []
        conf = []

        # conv4_3
        for k in range(self.vgg_src[0] + 1):
            x = self.vgg[k](x)

        s =  self.l2_norm(x)
        sources.append(s)
        #conv 7
        for k in range(self.vgg_src[0] + 1, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # for feature extractors
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # last layer
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # end shape: (N, H, W, C)
            l_out = l(x).permute(0, 2, 3, 1).contiguous()
            c_out = c(x).permute(0, 2, 3, 1).contiguous()

            loc.append(l_out.view(l_out.size(0), -1, 4))
            conf.append(c_out.view(c_out.size(0), -1, self.num_classes))

        loc = torch.cat(loc, dim=1) #loc now: (batch, 8732, 4)
        conf = torch.cat(conf, dim=1)#conf now (batch, 8732, num_classes)

        if self.phase == 'train':
            output = (loc, conf)
        else:
            output = (loc, F.softmax(conf, dim=-1))

        return output


