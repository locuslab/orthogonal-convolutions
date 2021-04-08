from layers import *

class KWLarge(nn.Module):
    def __init__(self, conv=CayleyConv, linear=CayleyLinear, w=1):
        super().__init__()
        self.model = nn.Sequential(
            conv(3, 32 * w, 3), GroupSort(),
            conv(32 * w, 32 * w, 3, stride=2), GroupSort(),
            conv(32 * w, 64 * w, 3), GroupSort(),
            conv(64 * w, 64 * w, 3, stride=2), GroupSort(),
            nn.Flatten(),
            linear(4096 * w, 512 * w), GroupSort(),
            linear(512 * w, 512), GroupSort(),
            linear(512, 10)
        )
    def forward(self, x):
        return self.model(x)
    
class ResNet9(nn.Module):
    def __init__(self, conv=CayleyConv, linear=CayleyLinear):
        super().__init__()
        self.block1 = nn.Sequential(
            conv(3, 64, 3), GroupSort(),
            conv(64, 128, 3), GroupSort(),
            nn.AvgPool2d(2, divisor_override=2)
        )
        self.res1 = nn.Sequential(
            conv(128, 128, 3), GroupSort(),
            conv(128, 128, 3), GroupSort()
        )
        self.combo1 = ConvexCombo()
        self.block2 = nn.Sequential(
            conv(128, 256, 3), GroupSort(),
            nn.AvgPool2d(2, divisor_override=2),
            conv(256, 512, 3), GroupSort(),
            nn.AvgPool2d(2, divisor_override=2)
        )
        self.res2 = nn.Sequential(
            conv(512, 512, 3), GroupSort(),
            conv(512, 512, 3), GroupSort()
        )
        self.combo2 = ConvexCombo()
        self.linear1 = nn.Sequential(
            nn.AvgPool2d(2, divisor_override=2),
            nn.Flatten(),
            linear(2048, 512), GroupSort(),
            linear(512, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        res = self.res1(x)
        x = self.combo1(x, res)
        x = self.block2(x)
        res = self.res2(x)
        x = self.combo2(x, res)
        x = self.linear1(x)
        return x

class PooledConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, conv=CayleyConv):
        super().__init__()
        self.stride = stride
        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=bias)

    def forward(self, X):
        if self.stride == 1:
            return self.conv(X)
        if self.stride == 2:
            return 2.0 * F.avg_pool2d(self.conv(X), 2)

###################################################################
# WideResNet from: https://github.com/xternalz/WideResNet-pytorch
# With some modifications
###################################################################

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, conv=CayleyConv):
        super(BasicBlock, self).__init__()
        self.relu1 = GroupSort()
        self.conv1 = PooledConv(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=True, conv=conv)
        self.relu2 = GroupSort()
        self.conv2 = PooledConv(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True, conv=conv)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and PooledConv(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=0, bias=True, conv=conv) or None
        self.a1 = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(x)
        else:
            out = self.relu1(x)
        out = self.relu2(self.conv1(out if self.equalInOut else x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(torch.sigmoid(self.a1) * (x if self.equalInOut else self.convShortcut(x)),
                (1.0 - torch.sigmoid(self.a1)) * out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, conv=CayleyConv):
        super(NetworkBlock, self).__init__()
        self.conv = conv
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, self.conv))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    # WideResNet10-10 by default
    def __init__(self, depth=10, num_classes=10, widen_factor=10, dropRate=0.0, conv=CayleyConv, linear=CayleyLinear):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.conv1 = PooledConv(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=True, conv=conv)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, conv)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, conv)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, conv)
        self.relu = GroupSort()
        self.fc1 = linear(4*nChannels[3], nChannels[3], bias=True)
        self.fc2 = linear(nChannels[3], num_classes, bias=True)
        self.nChannels = nChannels[3]

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(out)
        out = 4 * F.avg_pool2d(out, 4)
        out = out.view(-1, self.nChannels*4)
        out = self.relu(self.fc1(out))
        return self.fc2(out)
    
