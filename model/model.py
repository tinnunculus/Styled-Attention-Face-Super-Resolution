import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Function

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise

class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None

class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None

blur = BlurFunction.apply

class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, image_size, kernel_size=3, padding=1, style_dim = 256):
        super().__init__()

        self.size =image_size

        self.conv1 = nn.Sequential(
            FusedUpsample(in_channel, out_channel, kernel_size, padding=padding),
            Blur(out_channel)
        )
        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.lrelu = nn.LeakyReLU(0.2)

        self.conv2 = ConvLayer(out_channel, out_channel, kernel_size)
        self.noise2 = equal_lr(NoiseInjection(out_channel))

        self.conv3 = ConvLayer(out_channel, out_channel, kernel_size)
        self.noise3 = equal_lr(NoiseInjection(out_channel))

        self.conv4 = ConvLayer(out_channel, out_channel, kernel_size)
        self.noise4 = equal_lr(NoiseInjection(out_channel))

        self.conv5 = ConvLayer(out_channel, out_channel, kernel_size)
        self.noise5 = equal_lr(NoiseInjection(out_channel))

        self.to_style_1 = nn.Linear(style_dim, style_dim)
        self.to_style_2 = nn.Linear(style_dim, style_dim)
        self.to_style_3 = nn.Linear(style_dim, style_dim)
        self.to_style_4 = nn.Linear(style_dim, style_dim)
        self.to_style_5 = nn.Linear(style_dim, out_channel)
        self.ca_layer = CALayer(out_channel)

    def forward(self, input, style):
        batch = input.size(0)
        noise_1 = torch.randn(batch, 1, self.size, self.size, device=input[0].device)
        noise_2 = torch.randn(batch, 1, self.size, self.size, device=input[0].device)
        noise_3 = torch.randn(batch, 1, self.size, self.size, device=input[0].device)
        noise_4 = torch.randn(batch, 1, self.size, self.size, device=input[0].device)
        noise_5 = torch.randn(batch, 1, self.size, self.size, device=input[0].device)

        out_1 = self.conv1(input)
        out = self.noise1(out_1, noise_1)
        to_style = self.to_style_1(style)
        to_style = self.to_style_2(to_style)
        to_style = self.to_style_3(to_style)
        to_style = self.to_style_4(to_style)
        to_style = self.to_style_5(to_style)
        to_style = to_style.unsqueeze(2)
        to_style = to_style.unsqueeze(3)
        out = out * to_style
        out = self.lrelu(out)
        out = self.ca_layer(out)

        out = self.conv2(out)
        out = self.noise2(out, noise_2)
        out = self.lrelu(out)

        out = self.conv3(out)
        out = self.noise3(out, noise_3)
        out = self.lrelu(out)

        out = self.conv4(out)
        out = self.noise4(out,noise_4)
        out = self.lrelu(out)

        out = self.conv5(out)
        out = self.noise5(out,noise_5)
        out = self.lrelu(out)

        return out + out_1

class generator(nn.Module):
    def __init__(self, input_size = 32, latent_dim = 256,scale = 4):
        super(generator,self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.scale = scale

        self.initial = ConvLayer(in_channels=3, out_channels=512, kernel_size=3, stride=1)
        self.upsample_1 = BasicBlock(512, 512, image_size=8)

        self.origin_2 = ConvLayer(in_channels=3, out_channels=512, kernel_size=3, stride=1)
        self.upsample_2 = BasicBlock(512, 256, image_size=16)

        self.origin_3 = ConvLayer(in_channels=3, out_channels=256, kernel_size=3, stride=1)
        self.upsample_3 = BasicBlock(256, 256, image_size=32)

        self.origin_4 = ConvLayer(in_channels=3, out_channels=256, kernel_size=3, stride=1)
        self.upsample_4 = BasicBlock(256, 128, image_size=64)
        self.upsample_5 = BasicBlock(128, 64, image_size=128)
        self.upsample_6 = BasicBlock(64, 32, image_size=256)
        self.finish = ConvLayer(in_channels=32, out_channels=3, kernel_size=3, stride=1)

    def forward(self, input, style):
        # 4 x 4
        out = F.upsample(input, scale_factor=1/8, mode='bilinear')
        out = self.initial(out)
        out = self.upsample_1(out, style)

        # 8 x 8
        image = F.upsample(input, scale_factor=1/4, mode='bilinear')
        image = self.origin_2(image)
        out = out + image
        out = self.upsample_2(out, style)

        # 16 x 16
        image = F.upsample(input, scale_factor=1/2, mode='bilinear')
        image = self.origin_3(image)
        out = out + image
        out = self.upsample_3(out, style)

        # 32 x 32
        image = self.origin_4(input)
        out = out + image
        out = self.upsample_4(out, style)

        # 64 x 64
        out = self.upsample_5(out, style)

        # 128 x 128
        out = self.upsample_6(out, style)

        # 256 x 256
        out = self.finish(out)

        return out

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))