import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


def gram_matrix(x):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        
    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        style_loss = 0.0
        style_loss += self.criterion(gram_matrix(x_vgg[0]), gram_matrix(y_vgg[0]).detach())
        style_loss += self.criterion(gram_matrix(x_vgg[1]), gram_matrix(y_vgg[1]).detach())
        style_loss += self.criterion(gram_matrix(x_vgg[2]), gram_matrix(y_vgg[2]).detach())
        style_loss += self.criterion(gram_matrix(x_vgg[3]), gram_matrix(y_vgg[3]).detach())
        style_loss += self.criterion(gram_matrix(x_vgg[4]), gram_matrix(y_vgg[4]).detach())
        
        return style_loss

class AdversarialLoss(nn.Module):
    def __init__(self, use_bce=True):
        super(AdversarialLoss, self).__init__()
        if use_bce:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

    def forward(self, logits, target_is_real):
        if target_is_real:
            target_tensor = torch.ones_like(logits)
        else:
            target_tensor = torch.zeros_like(logits)
        return self.criterion(logits, target_tensor)
