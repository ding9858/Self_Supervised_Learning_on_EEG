import torch
from random import choice
from numbers import Real
import numpy as np
from torch.nn.functional import pad
from sklearn.utils import check_random_state
class Augment():
    def __init__(self, data, preaug=None):
        self.X = data  # crop data to avoid overflow
        self.preaug = preaug  # name of previous augmentation

    def _new_random_fft_phase_odd(self, batch_size, c, n, device, random_state):
        rng = check_random_state(random_state)
        random_phase = torch.from_numpy(
            2j * np.pi * rng.random((batch_size, c, (n - 1) // 2))
        ).to(device)

        return torch.cat([
            torch.zeros((batch_size, c, 1), device=device),
            random_phase,
            -torch.flip(random_phase, [-1])
        ], dim=-1)

    def _new_random_fft_phase_even(self, batch_size, c, n, device, random_state):
        rng = check_random_state(random_state)

        # print(rng.random((batch_size, c, n // 2 - 1)).shape)
        random_phase = torch.from_numpy(
            2j * np.pi * rng.random((batch_size, c, n // 2 - 1))
        ).to(device)

        return torch.cat([
            torch.zeros((batch_size, c, 1), device=device),
            random_phase,
            torch.zeros((batch_size, c, 1), device=device),
            -torch.flip(random_phase, [-1])
        ], dim=-1)

    def fourier(self,
                channel_indep=True,
                random_state=None
                ):
        phase_noise_magnitude = torch.distributions.Uniform(0, 1).sample()
        if len(self.X.shape) == 3:
            batch_size = self.X.shape[0]
        else:
            batch_size = 1

        assert isinstance(
            phase_noise_magnitude,
            (Real, torch.FloatTensor, torch.cuda.FloatTensor)
        ) and 0 <= phase_noise_magnitude <= 1, (
            f"eps must be a float beween 0 and 1. Got {phase_noise_magnitude}.")

        f = torch.fft.fft(self.X.double(), dim=-1)
        device = self.X.device

        n = f.shape[-1]
        if n % 2 == 0:
            random_phase = self._new_random_fft_phase_even(
                batch_size,
                f.shape[-2] if channel_indep else 1,
                n,
                device=device,
                random_state=random_state
            )
        else:
            random_phase = self._new_random_fft_phase_odd(
                batch_size,
                f.shape[-2] if channel_indep else 1,
                n,
                device=device,
                random_state=random_state
            )
        if not channel_indep:
            random_phase = torch.tile(random_phase, (1, f.shape[-2], 1))
        if isinstance(phase_noise_magnitude, torch.Tensor):
            phase_noise_magnitude = phase_noise_magnitude.to(device)
        f_shifted = f * torch.exp(phase_noise_magnitude * random_phase)
        shifted = torch.fft.ifft(f_shifted, dim=-1)
        transformed_X = shifted.real.float()

        return transformed_X

    def _analytic_transform(self, x):
        if torch.is_complex(x):
            raise ValueError("x must be real.")

        N = x.shape[-1]
        f = torch.fft.fft(x, N, dim=-1)
        h = torch.zeros_like(f)
        if N % 2 == 0:
            h[..., 0] = h[..., N // 2] = 1
            h[..., 1:N // 2] = 2
        else:
            h[..., 0] = 1
            h[..., 1:(N + 1) // 2] = 2

        return torch.fft.ifft(f * h, dim=-1)

    def _nextpow2(self, n):

        return int(np.ceil(np.log2(np.abs(n))))

    def _frequency_shift(self, X, fs, f_shift):

        # Pad the signal with zeros to prevent the FFT invoked by the transform
        # from slowing down the computation:
        n_channels, N_orig = X.shape[-2:]
        N_padded = 2 ** self._nextpow2(N_orig)
        t = torch.arange(N_padded, device=X.device) / fs
        padded = pad(X, (0, N_padded - N_orig))
        analytical = self._analytic_transform(padded)
        if isinstance(f_shift, (float, int, np.ndarray, list)):
            f_shift = torch.as_tensor(f_shift).float()
        reshaped_f_shift = f_shift.repeat(
            N_padded, n_channels, 1).T
        shifted = analytical * torch.exp(2j * np.pi * reshaped_f_shift * t)
        return shifted[..., :N_orig].real.float()

    def frequency_shift(self, delta_freq=0.004, sfreq=15.0):

        transformed_X = self._frequency_shift(
            X=self.X,
            fs=sfreq,
            f_shift=delta_freq,
        )
        return transformed_X

    def gaussian(self, std=0.001, random_state=None):
        X = self.X
        rng = check_random_state(random_state)
        if isinstance(std, torch.Tensor):
            std = std.to(X.device)
        noise = torch.from_numpy(
            rng.normal(
                loc=np.zeros(X.shape),
                scale=1
            ),
        ).float().to(X.device) * std
        transformed_X = X + noise
        return transformed_X

    def sign_flip(self):
        data = self.X

        return -data

    def time_flip(self):
        data = self.X
        return torch.flip(data, [-1])

    def sselct(self):
        # this function will random use one augmentation algorithm below
        ll = [self.fourier, self.frequency_shift, self.gaussian, self.sign_flip, self.time_flip]
        if self.preaug != None:
            # print("no", ll[self.preaug].__name__,"next time")  # delete the preaugmentation from list
            ll.pop(self.preaug)
        res = choice(range(len(ll)))
        return ll[res](), res, ll
