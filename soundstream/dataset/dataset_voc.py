import os

import torch
import torchaudio
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio.transforms import Resample


class SoundDataset(Dataset):
    """Dataset to load the audio data"""

    def __init__(self, audio_data, audio_dir):
        super().__init__()
        self.filenames = []
        data = pd.read_csv(audio_data)
        fnames = list(data["name"])

        # create the path from fnames
        self.filenames = list(map(lambda x: os.path.join(audio_dir, str(x) + ".wav"), fnames))
        self.start_samples = list(data["start"])
        self.end_samples = list(data["end"])

        print(f"Total files: {len(self.filenames)}")

        self.max_len = 24000  # corresponds to 1.5 second of audio
        self.target_sr = 16000

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        start_sample = self.start_samples[index]
        end_sample = self.end_samples[index]

        wav, sr = torchaudio.load(filename, backend="ffmpeg")

        assert (
            wav.numel() > 0  # number of elements
        ), f"one of your audio file ({filename}) is empty. please remove it from your folder"

        # convert to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # resample data to the target_sr
        transform = Resample(orig_freq=sr, new_freq=self.target_sr)
        wav = transform(wav)

        # slice the audio based on start and end samples
        wav = wav[:, start_sample:end_sample]

        wav_len = wav.size(1)

        # select a random clip from the audio
        if wav_len > self.max_len:
            max_start = wav_len - self.max_len
            start = torch.randint(0, max_start, (1,))
            wav = wav[:, start : start + self.max_len]
        else:
            wav = F.pad(wav, (0, self.max_len - wav_len), "constant")
        return wav
