import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(4, 8, 3, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
        nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
        nn.ReLU(True),
        nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),
        nn.ReLU(True),
        nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1),
        nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class UNet(nn.Module):
    def __init__(self, cl=[8, 8, 16, 32], bnorm=True):
        super(UNet, self).__init__()
        
        self.c1 = nn.Sequential(
            nn.Conv2d(1, cl[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cl[0]) if bnorm else nn.Identity()
        )
        self.p1 = nn.MaxPool2d(kernel_size=2)
        
        self.c2 = nn.Sequential(
            nn.Conv2d(cl[0], cl[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cl[1]) if bnorm else nn.Identity()
        )
        self.p2 = nn.MaxPool2d(kernel_size=2)
        
        self.c3 = nn.Sequential(
            nn.Conv2d(cl[1], cl[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cl[2]) if bnorm else nn.Identity()
        )
        self.p3 = nn.MaxPool2d(kernel_size=2)
        
        self.mid = nn.Sequential(
            nn.Conv2d(cl[2], cl[3], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cl[3]) if bnorm else nn.Identity()
        )
        
        self.u10 = nn.ConvTranspose2d(cl[3], cl[3], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c10 = nn.Sequential(
            nn.Conv2d(cl[3]+cl[2], cl[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cl[2]) if bnorm else nn.Identity()
        )
        
        self.u11 = nn.ConvTranspose2d(cl[2], cl[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c11 = nn.Sequential(
            nn.Conv2d(cl[2]+cl[1], cl[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cl[1]) if bnorm else nn.Identity()
        )
        
        self.u12 = nn.ConvTranspose2d(cl[1], cl[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c12 = nn.Sequential(
            nn.Conv2d(cl[1]+cl[0], cl[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cl[0]) if bnorm else nn.Identity()
        )
        
        self.c13 = nn.Sequential(
            nn.Conv2d(cl[0]+1, cl[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cl[0]) if bnorm else nn.Identity(),
            nn.Conv2d(cl[0], 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        c1 = self.c1(x)
        p1 = self.p1(c1)
        c2 = self.c2(p1)
        p2 = self.p2(c2)
        c3 = self.c3(p2)
        p3 = self.p3(c3)
        mid = self.mid(p3)
        u10 = self.u10(mid)
        c10 = self.c10(torch.cat([u10, c3], dim=1))
        u11 = self.u11(c10)
        c11 = self.c11(torch.cat([u11, c2], dim=1))
        u12 = self.u12(c11)
        c12 = self.c12(torch.cat([u12, c1], dim=1))
        c13 = self.c13(torch.cat([c12, x], dim=1))
        return c13
    
                          

class Stacked(nn.Module):
    def __init__(self):
        super(Stacked, self).__init__()
        self.s = nn.Sequential(
        nn.Conv2d(1, 8, 5, padding=2),
        nn.ReLU(True),
        nn.Conv2d(8, 16, 5, padding=2),
        nn.ReLU(True),
        nn.Conv2d(16, 32, 5, padding=2),
        nn.ReLU(True),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 1, 3, padding=1),
        nn.Sigmoid()
        )        
    
    def forward(self, x):
        return self.s(x)
    
class Pointwise(nn.Module):
    def __init__(self):
        super(Pointwise, self).__init__()
        self.s = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1), nn.ReLU(),# replicator
        nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1, groups=2), nn.ReLU(),# depthwise
        nn.Conv2d(in_channels=2, out_channels=4, kernel_size=1, padding=1, stride=2), nn.ReLU(),# pointwise ?

        nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1), nn.ReLU(),# replicator
        nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, groups=8), nn.ReLU(),# depthwise
        nn.Conv2d(in_channels=8, out_channels=5, kernel_size=1, padding=1, stride=2), nn.ReLU(),# pointwise ?

        nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1), nn.ReLU(),# replicator
        nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1, groups=10), nn.ReLU(),# depthwise
        nn.Conv2d(in_channels=10, out_channels=5, kernel_size=1, padding=1, stride=2), nn.ReLU(),# pointwise ?

        nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1), nn.ReLU(),# replicator
        nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1, groups=10), nn.ReLU(),# depthwise
        nn.ConvTranspose2d(in_channels=10, out_channels=4, kernel_size=1, padding=1, stride=2), nn.ReLU(),# pointwise ?

        nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1), nn.ReLU(),# replicator
        nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, groups=8), nn.ReLU(),# depthwise
        nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=1, padding=1, stride=2, output_padding=1), nn.ReLU(),# pointwise ?

        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1), nn.ReLU(),# replicator
        nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1, groups=6), nn.ReLU(),# depthwise
        nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=1, padding=1, stride=2, output_padding=1), nn.ReLU(),# pointwise ?

        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1), nn.ReLU(),# replicator
        nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1, groups=6), nn.ReLU(),# depthwise
        nn.Conv2d(in_channels=6, out_channels=1, kernel_size=1, padding=1), nn.ReLU(),# pointwise ?
        )

    def forward(self, x):
        return self.s(x)
