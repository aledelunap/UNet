import torch
import torch.nn as nn
import torchvision


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):

        x = self.block(x)

        return x


class Encoder(nn.Module):
    def __init__(self, channels=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.convolution_blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        self.enconding_block = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):

        feature_maps = []
        for block in self.convolution_blocks:
            x = block(x)
            feature_maps.append(x)
            x = self.enconding_block(x)

        return feature_maps


import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.channels = channels
        self.convolution_blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        self.decoding_block = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=2,
                    stride=2,
                )
                for i in range(len(channels) - 1)
            ]
        )

    def forward(self, x, encoder_features):
        for i in range(len(self.channels) - 1):
            x = self.decoding_block[i](x)
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.convolution_blocks[i](x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        enconding_channels=(3, 64, 128, 256, 512, 1024),
        decoding_channels=(1024, 512, 256, 128, 64),
        num_class=1,
    ):
        super().__init__()
        self.encoder = Encoder(enconding_channels)
        self.decoder = Decoder(decoding_channels)
        self.final_conv = nn.Conv2d(decoding_channels[-1], num_class, 1)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.final_conv(out)
        return out
