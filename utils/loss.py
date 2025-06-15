import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def mean_squared_error(output, target):
    return F.mse_loss(output, target)

def cross_entropy(output, target):
    return F.cross_entropy(output, target)

def frequency_mean_squared_error(output, target):

    output_fft = torch.fft.fft(output, dim=-1)
    target_fft = torch.fft.fft(target, dim=-1)

    # 勾配計算がうまくいくよう，FFTの実部と虚部を別々に計算
    loss_real = F.mse_loss(output_fft.real, target_fft.real)
    loss_imag = F.mse_loss(output_fft.imag, target_fft.imag)

    loss = loss_real + loss_imag
    return loss

def frequency_absolute_mean_squared_error(output, target):

    output_fft = torch.fft.fft(output, dim=-1)
    target_fft = torch.fft.fft(target, dim=-1)

    return F.mse_loss(output_fft.abs(), target_fft.abs())