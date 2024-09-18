import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchaudio.transforms import MelSpectrogram

import librosa
import scipy.signal
import numpy as np
from typing import List, Any


def apply_reduction(losses, reduction="none"):
    """Apply reduction to collection of losses."""
    if reduction == "mean":
        losses = losses.mean()
    elif reduction == "sum":
        losses = losses.sum()
    return losses


class FIRFilter(torch.nn.Module):
    """FIR pre-emphasis filtering module.

    Args:
        filter_type (str): Shape of the desired FIR filter ("hp", "fd", "aw"). Default: "hp"
        coef (float): Coefficient value for the filter tap (only applicable for "hp" and "fd"). Default: 0.85
        ntaps (int): Number of FIR filter taps for constructing A-weighting filters. Default: 101
        plot (bool): Plot the magnitude respond of the filter. Default: False

    Based upon the perceptual loss pre-empahsis filters proposed by
    [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).

    A-weighting filter - "aw"
    First-order highpass - "hp"
    Folded differentiator - "fd"

    Note that the default coefficeint value of 0.85 is optimized for
    a sampling rate of 44.1 kHz, considering adjusting this value at differnt sampling rates.
    """

    def __init__(self, filter_type="hp", coef=0.85, fs=44100, ntaps=101):
        """Initilize FIR pre-emphasis filtering module."""
        super(FIRFilter, self).__init__()
        self.filter_type = filter_type
        self.coef = coef
        self.fs = fs
        self.ntaps = ntaps

        import scipy.signal

        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

        if filter_type == "hp":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, -coef, 0]).view(1, 1, -1)
        elif filter_type == "fd":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, 0, -coef]).view(1, 1, -1)
        elif filter_type == "aw":
            # Definition of analog A-weighting filter according to IEC/CD 1672.
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217
            A1000 = 1.9997

            NUMs = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
            DENs = np.polymul(
                [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2],
            )
            DENs = np.polymul(
                np.polymul(DENs, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2]
            )

            # convert analog filter to digital filter
            b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)

            # compute the digital filter frequency response
            w_iir, h_iir = scipy.signal.freqz(b, a, worN=512, fs=fs)

            # then we fit to 101 tap FIR filter with least squares
            taps = scipy.signal.firls(ntaps, w_iir, abs(h_iir), fs=fs)

            # now implement this digital FIR filter as a Conv1d layer
            self.fir = torch.nn.Conv1d(
                1, 1, kernel_size=ntaps, bias=False, padding=ntaps // 2
            )
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor(taps.astype("float32")).view(1, 1, -1)

    def forward(self, input, target):
        """Calculate forward propagation.
        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor): Groundtruth signal (B, #channels, #samples).
        Returns:
            Tensor: Filtered signal.
        """
        input = torch.nn.functional.conv1d(
            input, self.fir.weight.data, padding=self.ntaps // 2
        )
        target = torch.nn.functional.conv1d(
            target, self.fir.weight.data, padding=self.ntaps // 2
        )
        return input, target


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719).
    """

    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        return (
            torch.norm(y_mag - x_mag, p="fro", dim=[-1, -2])
            / torch.norm(y_mag, p="fro", dim=[-1, -2])
        ).mean()


def adversarial_g_loss(y_disc_gen):
    loss = 0.0
    for i in range(len(y_disc_gen)):
        # print(f"Adversarial Generator Loss; y_disc_gen[{i}] shape: ", y_disc_gen[i].shape)
        # assert 1==2
        stft_loss = F.relu(1 - y_disc_gen[i]).mean().squeeze()
        loss += stft_loss
    return loss / len(y_disc_gen)


def feature_loss(fmap_r, fmap_gen):
    loss = 0.0
    for i in range(len(fmap_r)):
        for j in range(len(fmap_r[i])):
            stft_loss = (
                (fmap_r[i][j] - fmap_gen[i][j]).abs() / (fmap_r[i][j].abs().mean())
            ).mean()
            loss += stft_loss
    return loss / (len(fmap_r) * len(fmap_r[0]))


def sim_loss(y_disc_r, y_disc_gen):
    loss = 0.0
    for i in range(len(y_disc_r)):
        loss += F.mse_loss(y_disc_r[i], y_disc_gen[i])
    return loss / len(y_disc_r)


def sisnr_loss(x, s, eps=1e-8):
    """
    SI-SNR stands for Scale Invariant Signal-to-Noise Ratio

    calculate training loss
    input:
          x: separated signal, N x S tensor, estimate value
          s: reference signal, N x S tensor, True value
    Return:
          sisnr: N tensor
    """
    if x.shape != s.shape:
        if x.shape[-1] > s.shape[-1]:
            x = x[:, : s.shape[-1]]
        else:
            s = s[:, : x.shape[-1]]

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape
            )
        )
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = (
        torch.sum(x_zm * s_zm, dim=-1, keepdim=True)
        * s_zm
        / (l2norm(s_zm, keepdim=True) ** 2 + eps)
    )
    loss = -20.0 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))
    return torch.sum(loss) / x.shape[0]


def get_window(win_type: str, win_length: int):
    """Return a window function.

    Args:
        win_type (str): Window type. Can either be one of the window function provided in PyTorch
            ['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            or any of the windows provided by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html).
        win_length (int): Window length

    Returns:
        win: The window as a 1D torch tensor
    """

    try:
        win = getattr(torch, win_type)(win_length)
    except:
        win = torch.from_numpy(scipy.signal.windows.get_window(win_type, win_length))

    return win


def stft(x, fft_size, hop_size, win_length, window, phs_used, eps=1e-7):  # 1
    """Perform STFT.
    Args:
        x (Tensor): Input signal tensor (B, T).

    Returns:
        Tensor: x_mag, x_phs
            Magnitude and phase spectra (B, fft_size // 2 + 1, frames).
    """
    x_stft = torch.stft(
        x,
        fft_size,
        hop_size,
        win_length,
        window,
        return_complex=True,
    )
    x_mag = torch.sqrt(torch.clamp((x_stft.real**2) + (x_stft.imag**2), min=eps))
    if phs_used:
        x_phs = torch.angle(x_stft)
    else:
        x_phs = None

    return x_mag, x_phs


def reconstruction_loss(x, G_x, args, eps=1e-7, perceptual_weighting=True):
    spectralconv = SpectralConvergenceLoss()

    # wav L1 loss; LAMBDA_WAV = 100 (loss weight for time domain comparison)
    L = 100 * F.mse_loss(x, G_x)

    # n_mels = 128 # set for instrumentals  
    n_mels = 80 # set for vocals

    # added 2048, 4096, 8192 n_fft values too for improving frequency resolution
    for i in range(7, 14):
        # apply optional A-weighting via FIR filter
        if perceptual_weighting:
            prefilter = FIRFilter(filter_type="aw", fs=args.sr)
            bs, chs, seq_len = x.size()

            # since FIRFilter only support mono audio we will move channels to batch dim
            # however our audio is mono only
            x = x.view(bs * chs, 1, -1)
            G_x = G_x.view(bs * chs, 1, -1)

            # now apply the filter to both
            prefilter.to(args.device)
            x, G_x = prefilter(x, G_x)

            # now move the channels back
            x = x.view(bs, chs, -1)
            G_x = G_x.view(bs, chs, -1)  # e.g. 10, 1, 16000

        n_fft = 2**i
        hop_length = n_fft // 4
        assert n_mels <= n_fft

        melspec = MelSpectrogram(
            sample_rate=args.sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            wkwargs={"device": args.device},
        ).to(args.device)

        S_x = melspec(x)
        S_G_x = melspec(G_x)

        spec_conv_loss = spectralconv(S_x, S_G_x)

        lin_mel_loss = (S_x - S_G_x).abs().mean()
        log_mel_loss = (
            ((torch.log(S_x.abs() + eps) - torch.log(S_G_x.abs() + eps)) ** 2).mean(
                dim=-2
            )
            ** 0.5
        ).mean()

        tot_mel_loss = (lin_mel_loss + log_mel_loss + spec_conv_loss) / (i)

        L += tot_mel_loss
        # print(f"mel loss for frame size {n_fft}", tot_mel_loss)
    return L


def criterion_d(y_disc_r, y_disc_gen, fmap_r_det, fmap_gen_det):
    loss = 0.0
    loss_f = feature_loss(fmap_r_det, fmap_gen_det)
    for i in range(len(y_disc_r)):
        loss += F.relu(1 - y_disc_r[i]).mean() + F.relu(1 + y_disc_gen[i]).mean()
    return loss / len(y_disc_gen) + 0.0 * loss_f


def criterion_g(commit_loss, x, G_x, fmap_r, fmap_gen, y_disc_r, y_disc_gen, args):
    adv_g_loss = adversarial_g_loss(y_disc_gen)
    feat_loss = feature_loss(fmap_r, fmap_gen) + sim_loss(
        y_disc_r, y_disc_gen
    )  # 预测结果也应该尽可能相似
    rec_loss = reconstruction_loss(x.contiguous(), G_x.contiguous(), args)
    total_loss = (
        args.LAMBDA_COM * commit_loss
        + args.LAMBDA_ADV * adv_g_loss
        + args.LAMBDA_FEAT * feat_loss
        + args.LAMBDA_REC * rec_loss
    )
    return total_loss, adv_g_loss, feat_loss, rec_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def adopt_dis_weight(weight, global_step, threshold=0, value=0.0):
    if global_step % 3 == 0:  # 0,3,6,9,13....这些时间步，不更新dis
        weight = value
    return weight


def calculate_adaptive_weight(nll_loss, g_loss, last_layer, args):
    if last_layer is not None:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    else:
        print("last_layer cannot be none")
        assert 1 == 2
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.1, 10.0).detach()
    d_weight = d_weight * args.LAMBDA_ADV
    return d_weight


def loss_g(
    codebook_loss,
    inputs,
    reconstructions,
    fmap_r,
    fmap_gen,
    y_disc_r,
    y_disc_gen,
    global_step,
    last_layer=None,
    is_training=True,
    args=None,
):
    rec_loss = reconstruction_loss(
        inputs.contiguous(), reconstructions.contiguous(), args
    )
    adv_g_loss = adversarial_g_loss(y_disc_gen)
    feat_loss = feature_loss(fmap_r, fmap_gen) + sim_loss(y_disc_r, y_disc_gen)  #
    d_weight = torch.tensor(1.0)
    # try:
    #     d_weight = calculate_adaptive_weight(rec_loss, adv_g_loss, last_layer, args) # 动态调整重构损失和对抗损失
    # except RuntimeError:
    #     assert not is_training
    #     d_weight = torch.tensor(0.0)
    disc_factor = adopt_weight(
        args.LAMBDA_ADV, global_step, threshold=args.discriminator_iter_start
    )
    # feat_factor = adopt_weight(args.LAMBDA_FEAT, global_step, threshold=args.discriminator_iter_start)
    loss = (
        args.LAMBDA_REC * rec_loss
        + d_weight * disc_factor * adv_g_loss
        + args.LAMBDA_FEAT * feat_loss
        + args.LAMBDA_COM * codebook_loss
    )
    return loss, rec_loss, adv_g_loss, feat_loss, d_weight


def loss_dis(y_disc_r_det, y_disc_gen_det, fmap_r_det, fmap_gen_det, global_step, args):
    disc_factor = adopt_weight(
        args.LAMBDA_ADV, global_step, threshold=args.discriminator_iter_start
    )
    d_loss = disc_factor * criterion_d(
        y_disc_r_det, y_disc_gen_det, fmap_r_det, fmap_gen_det
    )
    return d_loss
