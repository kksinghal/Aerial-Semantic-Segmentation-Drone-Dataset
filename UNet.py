import torch 
import torch.nn as nn

def doubleConv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        torch.manual_seed(0)
        self.maxpool_2x2 = nn.MaxPool2d(kernel_size=2)
                
        self.doubleConv1 = doubleConv(3, 64)
        self.doubleConv2 = doubleConv(64, 128)
        self.doubleConv3 = doubleConv(128, 256)
        self.doubleConv4 = doubleConv(256, 512)
        self.doubleConv5 = doubleConv(512, 1024)
        
        self.convT1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.doubleConv6 = doubleConv(1024, 512)
        
        self.convT2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.doubleConv7 = doubleConv(512, 256)
        
        self.convT3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.doubleConv8 = doubleConv(256, 128)
        
        self.convT4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.doubleConv9 = doubleConv(128, 64)
        
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

        
    def forward(self, X):
        #Encoder
        x1 = self.doubleConv1(X)
        x2 = self.maxpool_2x2(x1)
        
        x2 = self.doubleConv2(x2)
        x3 = self.maxpool_2x2(x2)
        
        x3 = self.doubleConv3(x3)
        x4 = self.maxpool_2x2(x3)
        
        x4 = self.doubleConv4(x4)
        out = self.maxpool_2x2(x4)
        
        out = self.doubleConv5(out)
        
        #Decoder
        out = self.convT1(out)
        out = torch.cat((x4, out), dim=1)
        out = self.doubleConv6(out)
        
        out = self.convT2(out)
        out = torch.cat((x3, out), dim=1)
        out = self.doubleConv7(out)
        
        out = self.convT3(out)
        out = torch.cat((x2, out), dim=1)
        out = self.doubleConv8(out)
        
        out = self.convT4(out)
        out = torch.cat((x1, out), dim=1)
        out = self.doubleConv9(out)
        
        out = self.final_conv(out)
        
        return out
