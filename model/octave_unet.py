import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


import math

from torch import nn


class OctaveConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        alpha_in=0.5,
        alpha_out=0.5,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(OctaveConv, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        assert stride == 1 or stride == 2, 'Stride should be 1 or 2'
        self.stride = stride

        self.is_dw = groups == in_channels

        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, (
            'Alpha should be in the interval from 0 to 1'
        )
        self.alpha_in, self.alpha_out = alpha_in, alpha_out

        # print("alpha_in: ", alpha_in, " alpha_out: ", alpha_out)
        self.conv_l2l = (
            None
            if alpha_in == 0 or alpha_out == 0
            else nn.Conv2d(
                in_channels=int(alpha_in * in_channels),
                out_channels=int(alpha_out * out_channels),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=math.ceil(alpha_in * groups),
                bias=bias,
            )
        )

        self.conv_l2h = (
            None
            if alpha_in == 0 or alpha_out == 1 or self.is_dw
            else nn.Conv2d(
                in_channels=int(alpha_in * in_channels),
                out_channels=out_channels - int(alpha_out * out_channels),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        )

        self.conv_h2l = (
            None
            if alpha_in == 1 or alpha_out == 0 or self.is_dw
            else nn.Conv2d(
                in_channels=in_channels - int(alpha_in * in_channels),
                out_channels=int(alpha_out * out_channels),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        )

        self.conv_h2h = (
            None
            if alpha_in == 1 or alpha_out == 1
            else nn.Conv2d(
                in_channels=in_channels - int(alpha_in * in_channels),
                out_channels=out_channels - int(alpha_out * out_channels),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=math.ceil(groups - alpha_in * groups),
                bias=bias,
            )
        )

    def forward(self, x):

        x_h, x_l = x if type(x) is tuple else (x, None)

        # fig, axs = plt.subplots

        x_h = self.downsample(x_h) if self.stride == 2 else x_h

        x_h2h = self.conv_h2h(x_h) if (self.conv_h2h is not None) else None
        x_h2l = (
            self.conv_h2l(self.downsample(x_h))
            if self.alpha_out > 0 and not self.is_dw
            else None
        )

        if x_l is not None:
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None

            if self.is_dw:
                return x_h2h, x_l2l
            else:
                x_l2h = self.conv_l2h(x_l)
                x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
                # print("Testes ", x_l2h.size(), x_h2h.size())
                if x_l2h.size()[-1] != x_h2h.size()[-1]:
                    x_l2h = nn.functional.pad(x_l2h, (0, 1), mode='constant', value=0)
                if x_l2h.size()[-2] != x_h2h.size()[-2]:
                    x_l2h = nn.functional.pad(
                        x_l2h, (0, 0, 0, 1), mode='constant', value=0
                    )

                x_h = x_l2h + x_h2h
                x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None

                return x_h, x_l

        else:
            return x_h2h, x_h2l


class Conv_BN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        alpha_in=0.5,
        alpha_out=0.5,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        norm_layer=nn.BatchNorm2d,
    ):
        print('CONV_BN')
        super(Conv_BN, self).__init__()
        self.conv = OctaveConv(
            in_channels,
            out_channels,
            kernel_size,
            alpha_in,
            alpha_out,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        self.bn_h = (
            None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        )
        self.bn_l = (
            None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))
        )

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_l


class OctaveConv_ACT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        alpha_in=0.5,
        alpha_out=0.5,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        activation_layer=nn.ReLU,
    ):
        super(OctaveConv_ACT, self).__init__()
        self.conv = OctaveConv(
            in_channels,
            out_channels,
            kernel_size,
            alpha_in,
            alpha_out,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        self.act = activation_layer(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.act(x_h) if x_h is not None else None
        x_l = self.act(x_l) if x_l is not None else None
        return x_h, x_l


class Conv_BN_ACT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        alpha_in=0.5,
        alpha_out=0.5,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
    ):
        super(Conv_BN_ACT, self).__init__()
        print('CONV_BN_ACT')
        self.conv = OctaveConv(
            in_channels,
            out_channels,
            kernel_size,
            alpha_in,
            alpha_out,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        self.bn_h = (
            None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        )
        self.bn_l = (
            None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))
        )
        self.act = activation_layer(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.act(self.bn_h(x_h))
        x_l = self.act(self.bn_l(x_l)) if x_l is not None else None
        return x_h, x_l



class TransposeOctConv(nn.Module):
    """This is the implementation of Octave Transpose Conv from paper https://arxiv.org/abs/1906.12193"""

    def __init__(
        self,
        in_chn,
        out_chn,
        alphas=[0.5, 0.5],
        kernel=2,
    ):

        super(TransposeOctConv, self).__init__()

        (self.alpha_in, self.alpha_out) = alphas

        assert 1 > self.alpha_in >= 0 and 1 > self.alpha_out >= 0, (
            'alphas values must be bound between 0 and 1, it could be 0 but not 1'
        )

        self.htoh = nn.ConvTranspose2d(
            in_chn - int(self.alpha_in * in_chn),
            out_chn - int(self.alpha_out * out_chn),
            kernel,
            2,
        )
        self.htol = (
            nn.ConvTranspose2d(
                in_chn - int(self.alpha_in * in_chn),
                int(self.alpha_out * out_chn),
                kernel,
                2,
            )
            if self.alpha_out > 0
            else None
        )
        self.ltol = (
            nn.ConvTranspose2d(
                int(self.alpha_in * in_chn), int(self.alpha_out * out_chn), kernel, 2
            )
            if self.alpha_out > 0 and self.alpha_in > 0
            else None
        )
        self.ltoh = (
            nn.ConvTranspose2d(
                int(self.alpha_in * in_chn),
                out_chn - int(self.alpha_out * out_chn),
                kernel,
                2,
            )
            if self.alpha_in > 0
            else None
        )

    def forward(self, x):
        (high, low) = x if isinstance(x, tuple) else (x, None)

        if self.htoh is not None:
            htoh = self.htoh(high)
        if self.htol is not None:
            htol = self.htol(F.avg_pool2d(high, 2, 2))
        if self.ltol is not None and low is not None:
            ltol = self.ltol(low)
        if self.ltoh is not None and low is not None:
            ltoh = F.interpolate(self.ltoh(low), scale_factor=2, mode='nearest')

        # it will behave as normal Transpose Conv operation

        if self.alpha_in == 0 and self.alpha_out == 0:
            return (htoh, None)

        # case where we don't want a low frequency map as output

        if self.alpha_out == 0:
            return (htoh.add_(ltoh), None)

        # otherwise add feature maps and return both high and low freq maps

        htoh.add_(ltoh)
        ltol.add_(htol)

        return (htoh, ltol)



class Encoder_block(nn.Module):
    def __init__(
        self,
        input_chn,
        output_chn,
        alphas=[0.5, 0.5],
        num_layers=2,
        kernel=3,
        padding=1,
        last_block=False,
    ):
        super(Encoder_block, self).__init__()

        layers_chn = [input_chn, output_chn] + [output_chn for i in range(num_layers)]

        if len(alphas) == 4:
            # means alphas for both blocks are given

            alphas1 = alphas[0:2]
            alphas2 = alphas[2:]
        else:
            # use same alphas for both blocks

            (alphas1, alphas2) = (alphas, alphas)

        self.pool = None
        self.block1 = OctaveBnAct(layers_chn[0], layers_chn[1], alphas1)
        self.block2 = Oct(layers_chn[2], layers_chn[3], alphas2)

        if last_block is False:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        if self.pool:
            pool = (self.pool(x[0]), self.pool(x[1]))

            # here both x and pool are tuples consisting of high and low freq maps

            return (x, pool)
        return (x, None)


class Decoder_block(nn.Module):
    def __init__(
        self,
        input_chn,
        output_chn,
        alphas=[0.5, 0.5],
        bilinear=False,
    ):
        super(Decoder_block, self).__init__()

        # upsample feature maps size using either way

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = TransposeOctConv(input_chn, output_chn)

        if len(alphas) == 4:
            # means alphas for both blocks are given

            alphas1 = alphas[0:2]
            alphas2 = alphas[2:]
        else:
            # use same alphas for both blocks

            (alphas1, alphas2) = (alphas, alphas)

        # use two conv operation to process over upsample feature maps

        self.block1 = OctaveBnAct(input_chn, output_chn, alphas1)
        self.block2 = OctaveBnAct(output_chn, output_chn, alphas2)

    def forward(self, x, encoder_feat):
        x = self.up(x)
        x = (F.relu(x[0]), F.relu(x[1]))

        # concatenate high freq maps with corresponding encoder map

        assert x[0].shape[1] == encoder_feat[0].shape[1], (
            'High freq maps channels should be same'
        )
        high = torch.cat([x[0], encoder_feat[0]], dim=1)

        # concatenate low freq maps with corresponding encoder map

        if x[1] is not None and encoder_feat[1] is not None:
            assert x[1].shape[1] == encoder_feat[1].shape[1], (
                'Low freq maps channels should be same'
            )
            low = torch.cat([x[1], encoder_feat[1]], dim=1)

        x = (high, low)
        x = self.block1(x)
        x = self.block2(x)

        return x


class OctaveUnet(nn.Module):
    """This is Unet architecture implementation using Oactave and Transpose Ovctave Convolution from the paper https://arxiv.org/abs/1906.12193.
    However, in the paper feature maps retain same spatial size by using octave conv after max pool operation, here feature maps size is reduceed
    as per the Unet paper."""

    def __init__(self, num_classes):
        super(OctaveUnet, self).__init__()
        self.blocks = nn.ModuleDict()

        down_chn = [
            3,
            64,
            128,
            256,
            512,
            1024,
        ]
        up_chn = down_chn[::-1]

        # initial block which take in original image (only high freq map)

        self.blocks.update(
            {'down_block1': Encoder_block(down_chn[0], down_chn[1], [0, 0.5, 0.5, 0.5])}
        )

        # rest of the encoder blocks

        for i in range(1, 5):
            if i == 4:
                # this makes sure that max pooling is not applied to the last block of encoder side

                self.blocks.update(
                    {
                        'down_block' + str(i + 1): Encoder_block(
                            down_chn[i], down_chn[i + 1], last_block=True
                        )
                    }
                )
            else:
                # max pooling should be applied to rest of the blocks on encoder side

                self.blocks.update(
                    {
                        'down_block' + str(i + 1): Encoder_block(
                            down_chn[i], down_chn[i + 1]
                        )
                    }
                )

        # decoder blocks

        for i in range(3):
            self.blocks.update(
                {'up_block' + str(i + 1): Decoder_block(up_chn[i], up_chn[i + 1])}
            )

        # final block of decoder only outputs high freq map

        self.blocks.update(
            {
                'up_block4': Decoder_block(
                    up_chn[i + 1], up_chn[i + 2], [0.5, 0.5, 0.5, 0]
                )
            }
        )

        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):

        # save the encoder blocks output

        encoder_results = list()

        # iterate over encoder blocks

        for i in range(5):
            (feat, x) = self.blocks['down_block' + str(i + 1)](x)
            encoder_results.append([feat, x])

        index = 3
        for i in range(4):
            feat = self.blocks['up_block' + str(i + 1)](
                feat, encoder_results[index - i][0]
            )

        # classifier only gets the high freq map in the end

        return self.classifier(feat[0])

if __name__ == '__main__':
    model = OctaveUnet(20)
    print(model)
    im_random = torch.rand(1, 3, 300, 300)
    exit = model(im_random)