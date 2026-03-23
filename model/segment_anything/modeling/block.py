import torch
from torch import nn
from .common import LayerNorm2d
from torch.nn import functional as F


class ModifyPPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(ModifyPPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(reduction_dim, reduction_dim, kernel_size=3, bias=False, groups = reduction_dim),
                nn.GELU()
            ))
        self.features = nn.ModuleList(self.features)
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding = 1, bias=False, groups = in_dim),
            nn.GELU(),
        )
        

    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class LMSA(nn.Module):
    def __init__(self, in_dim, hidden_dim, patch_num, require_feature):
        super().__init__()
        skipdim = 32
        self.down_project = nn.Linear(in_dim,hidden_dim) 
        self.act = nn.GELU()
        self.mppm = ModifyPPM(hidden_dim, hidden_dim //4,  [5,11,17,23])
        self.patch_num = patch_num
        self.up_project = nn.Linear(hidden_dim, in_dim)
        self.down_conv = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim, 1),
                                       nn.GELU())
        self.require_feature = require_feature

        if require_feature:
            self.dproject = nn.Linear(in_dim, hidden_dim // 4) 
            self.fuseproj = nn.Linear(hidden_dim // 4 + skipdim, hidden_dim) 
            self.act2 = nn.GELU()
            self.convconcat = BasicConv2d(1, 1, 3, padding = 1)
            self.SA1 = SpatialAttention()
            self.SA2 = SpatialAttention()
            self.SA3 = SpatialAttention()
            self.SA4 = SpatialAttention()
            self.weight = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
            self.sa_fusion = nn.Sequential(BasicConv2d(1, 1, 3, padding=1),
                                        nn.Sigmoid()
                                        )

    def forward(self, x, x_skip):
        #x:32 * 32 * 1024
        #x_skip: 32 * 8 * 8 要先调整维度
        if self.require_feature:
            # return x
            x_size = x.size()
            x_skipused = F.interpolate(x_skip, x_size[1:3], mode='bilinear', align_corners=True)
            # x_skipused = x_skipused.permute(0, 2, 3, 1).contiguous()

            down_x = self.dproject(x) # 128 * 32 * 32
            down_x = self.act(down_x)
            down_x = down_x.permute(0, 3, 1, 2).contiguous()

            x1, x2, x3, x4 = torch.split(down_x, 32, dim = 1)
            s1 = self.SA1(x1)
            s2 = self.SA2(x2)
            s3 = self.SA3(x3)
            s4 = self.SA4(x4)
            nor_weights = F.softmax(self.weight, dim=0)
            s_all = s1 * nor_weights[0] + s2 * nor_weights[1] + s3 * nor_weights[2] + s4 * nor_weights[3]
            x_tmp = self.sa_fusion(s_all) * down_x + down_x

            x_new = torch.cat((x_skipused, x_tmp), dim = 1).permute(0, 2, 3, 1).contiguous()
            x_out = self.fuseproj(x_new)
            down_x = self.act2(x_out)


        else:
            down_x = self.down_project(x)
            down_x = self.act(down_x)

            down_x = down_x.permute(0, 3, 1, 2).contiguous()
            down_x = self.mppm(down_x).contiguous()
            down_x = self.down_conv(down_x)
            down_x = down_x.permute(0, 2, 3, 1).contiguous()

        up_x = self.up_project(down_x)
        x_out = x + up_x

        return x_out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x