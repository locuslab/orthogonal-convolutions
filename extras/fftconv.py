import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FFTConv(nn.Conv2d):
    def fft_shift_matrix(self, n, shift_amount):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * shift_amount * shift / n)

    def forward(self, x):
        cout, cin, _, _ = self.weight.shape
        batches, _, n, _ = x.shape  # assume square images.
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = self.fft_shift_matrix(n, -s)[:, :(n//2 + 1)][None, None].to(x.device)

        xfft = torch.fft.rfft2(x, (n, n))
        wfft = torch.fft.rfft2(self.weight, (n, n)).conj()
        yfft = torch.einsum('bchw, bchw, dchw -> hwdb', self.shift_matrix, xfft, wfft)

        # Striding based on the downsampling/aliasing theorem
        if self.stride[0] == 2:
            g = n // 2 // 2 + 1
            yfftr = torch.roll(torch.flip(yfft[:, -g:(n//2 + 1)], (0, 1)), 1, 0).conj()
            yfft = 0.25 * (yfft[0:n//2, 0:g] + yfftr[0:n//2] + \
                           yfft[n//2:n, 0:g] + yfftr[n//2:n])

        # More naive but almost as efficient version of 2-striding
        #if self.stride[0] == 2:
            #y = y[..., ::2, ::2]
            
        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1), (n, n))

        if self.bias is not None:
            y += self.bias[:, None, None]

        return y