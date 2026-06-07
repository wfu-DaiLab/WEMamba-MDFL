import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, use_sigmoid=False):
        super(PatchDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, padding=1)
        )
        
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None

    def forward(self, img):
        result = self.model(img)
        if self.sigmoid:
            return self.sigmoid(result)
        return result
