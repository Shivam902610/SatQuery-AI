
import torch
import torch.nn as nn


# Basic convolution block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# Siamese U-Net
class SiameseUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared encoder for both satellite images
        self.encoder1 = ConvBlock(3, 32)
        self.encoder2 = ConvBlock(32, 64)
        self.encoder3 = ConvBlock(64, 128)
        self.encoder4 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder1 = ConvBlock(64, 32)

        # Final change prediction
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, image_a, image_b):

        # Encode first image
        a1 = self.encoder1(image_a)
        a2 = self.encoder2(self.pool(a1))
        a3 = self.encoder3(self.pool(a2))
        a4 = self.encoder4(self.pool(a3))

        # Encode second image using the same encoder
        b1 = self.encoder1(image_b)
        b2 = self.encoder2(self.pool(b1))
        b3 = self.encoder3(self.pool(b2))
        b4 = self.encoder4(self.pool(b3))

        # Compare feature maps
        f1 = torch.abs(a1 - b1)
        f2 = torch.abs(a2 - b2)
        f3 = torch.abs(a3 - b3)
        f4 = torch.abs(a4 - b4)

        # Decode and combine multi-scale differences
        x = self.up3(f4)
        x = self.decoder3(torch.cat([x, f3], dim=1))

        x = self.up2(x)
        x = self.decoder2(torch.cat([x, f2], dim=1))

        x = self.up1(x)
        x = self.decoder1(torch.cat([x, f1], dim=1))

        return self.output(x)
