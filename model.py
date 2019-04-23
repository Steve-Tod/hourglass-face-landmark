import torch
import torch.nn as nn

class StackedHourGlass(nn.Module):
    def __init__(self, opt):
        super(StackedHourGlass, self).__init__()
        self.opt = opt
        self.num_feature = opt['num_feature']
        self.num_stack = opt['num_stack']
        self.pre_conv_block = nn.Sequential(
            nn.Conv2d(3, self.num_feature // 4, 7, 2, 3),
            nn.BatchNorm2d(self.num_feature // 4),
            nn.ReLU(inplace=True),
            ResidualBlock(self.num_feature // 4, self.num_feature // 2),
            nn.MaxPool2d(2, 2),
            ResidualBlock(self.num_feature // 2, self.num_feature // 2),
            ResidualBlock(self.num_feature // 2, self.num_feature),
        )
        self._init_stacked_hourglass()

    def _init_stacked_hourglass(self):
        for i in range(self.num_stack):
            setattr(self, 'hg' + str(i), HourGlass(self.opt['num_layer'], self.num_feature))
            setattr(self, 'hg' + str(i) + '_res1',
                    ResidualBlock(self.num_feature, self.num_feature))
            setattr(self, 'hg' + str(i) + '_lin1',
                    Lin(self.num_feature, self.num_feature))
            setattr(self, 'hg' + str(i) + '_conv_pred',
                    nn.Conv2d(self.num_feature, self.opt['num_keypoints'], 1))
            if i < self.num_stack - 1:
                setattr(self, 'hg' + str(i) + '_conv1',
                        nn.Conv2d(self.num_feature, self.num_feature, 1))
                setattr(self, 'hg' + str(i) + '_conv2',
                        nn.Conv2d(self.opt['num_keypoints'], self.num_feature, 1))
                
    def forward(self,x):
        x = self.pre_conv_block(x) #(n,256,32,32)

        out = []
        inter = x

        for i in range(self.num_stack):
            hg = eval('self.hg'+str(i))(inter)
            # Residual layers at output resolution
            ll = hg
            ll = eval('self.hg'+str(i)+'_res1')(ll)
            # Linear layer to produce first set of predictions
            ll = eval('self.hg'+str(i)+'_lin1')(ll)
            # Predicted heatmaps
            tmpOut = eval('self.hg'+str(i)+'_conv_pred')(ll)
            out.append(tmpOut)
            # Add predictions back
            if i < self.num_stack - 1:
                ll_ = eval('self.hg'+str(i)+'_conv1')(ll)
                tmpOut_ = eval('self.hg'+str(i)+'_conv2')(tmpOut)
                inter = inter + ll_ + tmpOut_
        return out

    
class GHCU(nn.Module):
    def __init__(self, opt):
        super(GHCU, self).__init__()
        self.feature = nn.Sequential(
            self.conv_block(opt['in_channel'], 64, 5, 2),  # B*64*16*16
            self.conv_block(64, 32, 5, 2),  # B*32*8*8
            self.conv_block(32, 16, 3, 2),  # B*16*4*4
        )
        self.regressor = nn.Sequential(
            nn.Linear(opt['feature_length'], 256),
            nn.Dropout2d(opt['drop_rate']), nn.ReLU(inplace=True),
            nn.Linear(256, opt['output_dim']))

    def forward(self, x):
        x = self.feature(x)
        x = self.regressor(x.view(x.size()[0], -1))
        return x

    def conv_block(self, num_in, num_out, kernel_size, stride, padding=None):
        if not padding:
            padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(num_in, num_out, kernel_size, stride, padding),
            nn.BatchNorm2d(num_out), nn.ReLU(inplace=True))
    
    
class HourGlass(nn.Module):
    def __init__(self, num_layer, num_feature):
        super(HourGlass, self).__init__()
        self._n = num_layer
        self._f = num_feature
        self._init_layers(self._n, self._f)

    def _init_layers(self, n, f):
        setattr(self, 'res' + str(n) + '_1', ResidualBlock(f, f))
        setattr(self, 'pool' + str(n) + '_1', nn.MaxPool2d(2, 2))
        setattr(self, 'res' + str(n) + '_2', ResidualBlock(f, f))
        if n > 1:
            self._init_layers(n - 1, f)
        else:
            self.res_center = ResidualBlock(f, f)
        setattr(self, 'res' + str(n) + '_3', ResidualBlock(f, f))

    def _forward(self, x, n, f):
        up1 = eval('self.res' + str(n) + '_1')(x)

        low1 = eval('self.pool' + str(n) + '_1')(x)
        low1 = eval('self.res' + str(n) + '_2')(low1)
        if n > 1:
            low2 = self._forward(low1, n - 1, f)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.' + 'res' + str(n) + '_3')(low3)
        up2 = nn.functional.interpolate(low3, scale_factor=2, mode='bilinear', align_corners=True)

        return up1 + up2

    def forward(self, x):
        return self._forward(x, self._n, self._f)
    

class Lin(nn.Module):
    def __init__(self,numIn,numout):
        super(Lin,self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(numIn,numout,1), 
            nn.BatchNorm2d(numout),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv_block(x)
    
    
class ResidualBlock(nn.Module):
    def __init__(self, num_in, num_out):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_in, num_out // 2, 1), nn.BatchNorm2d(num_out // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_out // 2, num_out // 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_out // 2), nn.ReLU(inplace=True),
            nn.Conv2d(num_out // 2, num_out, 1), nn.BatchNorm2d(num_out))
        self.skip_layer = None if num_in == num_out else nn.Sequential(
            nn.Conv2d(num_in, num_out, 1), nn.BatchNorm2d(num_out))
    
    def forward(self, x):
        residual = self.conv_block(x)
        if self.skip_layer:
            x = self.skip_layer(x)
        return x + residual