import torch
import torch.nn as nn
import torch.nn.functional as F

class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False)
        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

        self.conv1_bn = nn.BatchNorm2d(mid_channel)

        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

        self.conv2_bn = nn.BatchNorm2d(out_channel)

        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.att_conv = None

    def forward(self, x, y=None, shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)

        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")  # student와 teacher의 feature size를 맞추기 위함. y : residual feature
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w))

        y = self.conv2(x)  # teacher의 feature와 맞추기 위함?
        return y, x  # y는 teacher의 feature와 같아야함.


class ReviewKD(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channel,
                 shapes=[24, 48, 96, 192],
                 hcl_mode="avg"):
        super().__init__()
        self.shapes = shapes

        abfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels) - 1)
            )

        self.abfs = abfs[::-1]

        self.hcl = HCL(mode=hcl_mode)

    def forward(self, student_features, teacher_features):
        '''
        student_features: list of tensor, low-level -> high-level
        student_logit: tensor, N x class_num
        '''
        # merge students' feature
        # x is from high-level to low-level
        x = student_features[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0])
        results.append(out_features)

        for idx in range(1, len(x)):
            features, abf, shape = x[idx], self.abfs[idx], self.shapes[idx]
            out_features, res_features = abf(features, res_features, shape)
            results.insert(0, out_features)  # student의 out_features가 HCL로 들어감

        return self.hcl(results, teacher_features)


class HCL(nn.Module):
    def __init__(self, mode="avg"):
        super(HCL, self).__init__()
        assert mode in ["max", "avg"]
        self.mode = mode

    def forward(self, fstudent, fteacher):
        loss_all = 0.0
        for fs, ft in zip(fstudent, fteacher):
            h = fs.shape[2]
            loss = F.mse_loss(fs, ft)
            cnt = 1.0
            tot = 1.0
            for l in [4, 2, 1]:
                if l >= h:
                    continue
                if self.mode == "max":
                    tmpfs = F.adaptive_max_pool2d(fs, (l, l))
                    tmpft = F.adaptive_max_pool2d(ft, (l, l))
                else:
                    tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
                    tmpft = F.adaptive_avg_pool2d(ft, (l, l))

                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft) * cnt
                tot += cnt

            loss = loss / tot
            loss_all = loss_all + loss
        return loss_all