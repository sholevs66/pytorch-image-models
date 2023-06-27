""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""

from torch import nn as nn

from .helpers import to_2tuple


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

    '''
    def fuse_bn(self):

        if not self.flatten:
            return self

        embed_dim, in_chans, _, _ = self.proj.weight.size()
        new_layer = PatchEmbedAsConv1x1(
            self.img_size[0], self.patch_size, in_chans, embed_dim)
        new_layer.norm = self.norm

        weights = self.proj.weight.view(
            embed_dim, self.patch_size[0]*self.patch_size[1]*in_chans, 1, 1)
        bias = self.proj.bias

        new_layer.proj.weight = nn.Parameter(weights)
        new_layer.proj.bias = nn.Parameter(bias)

        return new_layer
    '''

class PatchEmbedAsConv1x1(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = x.view(B, C, self.grid_size[0], self.patch_size[0],
                   self.grid_size[1], self.patch_size[1])
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B,
                      C * self.patch_size[0] * self.patch_size[1],
                      self.num_patches,
                      1)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class PatchConv(nn.Module):
    #def __init__(self, *, embed_dim=768, channels=3, norm_layer=None, flatten=True):
    def __init__(self, embed_dim=768, channels=3, norm_layer=None, flatten=True, **kwargs):
        """
        3x3 conv, stride 1, 5 conv layers per https://arxiv.org/pdf/2106.14881v2.pdf
        """
        super().__init__()

        n_filter_list = (channels, 64, 128, 256, 512)  # hardcoding for now because that's what the paper used
        #n_filter_list = (channels, 64, 128, 128, 256, 256, 512)
        self.num_patches = 196 # hardcoding for now, replacing patch16 -> 14x14=196 patches
        self.flatten = flatten

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels=n_filter_list[i],
                        out_channels=n_filter_list[i + 1],
                        kernel_size=3,  # hardcoding for now because that's what the paper used
                        stride=2,  # hardcoding for now because that's what the paper used
                        padding=1),  # hardcoding for now because that's what the paper used
                nn.BatchNorm2d(num_features=n_filter_list[i + 1]),
                nn.ReLU()
            )
                for i in range(len(n_filter_list)-1)
            ])


        self.conv_layers.add_module("conv_1x1", nn.Conv2d(in_channels=n_filter_list[-1],
                                    out_channels=embed_dim,
                                    stride=1,  # hardcoding for now because that's what the paper used
                                    kernel_size=1,  # hardcoding for now because that's what the paper used
                                    padding=0))  # hardcoding for now because that's what the paper used

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv_layers(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2) # BCHW -> BNC
        x = self.norm(x)
        return x
