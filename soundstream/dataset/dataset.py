import os

import torch
import torchaudio
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio.functional import resample

# from torchaudio.transforms import Resample


class SoundDataset(Dataset):
    """Dataset to load the audio data"""

    def __init__(self, audio_type, audio_data, audio_dir):
        super().__init__()
        self.audio_type = audio_type
        self.filenames = []
        data = pd.read_csv(audio_data)
        filenames = list(data["name"])

        # create the path from fnames
        self.filenames = list(map(lambda x: os.path.join(audio_dir, x), filenames))
        if self.audio_type == "vocals":
            self.start_samples = list(data["start"])
            self.end_samples = list(data["end"])

        print(f"Total files: {len(self.filenames)}")

        self.max_len = 24000  # corresponds to 1.5 second of audio
        self.target_sr = 16000

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        wav, sr = torchaudio.load(filename, backend="ffmpeg")

        assert (
            wav.numel() > 0  # number of elements
        ), f"one of your audio file ({filename}) is empty. please remove it from your folder"

        # mean and resample operations on full audios are time expensive
        # randomly select 30 seconds of instrumental audio, and then perform the mean and resample ops
        # the same can't be don for vocals because we have a start and end sample specified there
        # we can directly select the 1.5 second clip too
        if self.audio_type == "instrumentals":
            wav_dur = wav.size(1) // sr
            if wav_dur > 30:
                max_start = wav_dur - 30
                start = torch.randint(0, max_start, (1,))
                wav = wav[:, start * sr : (start + 30) * sr]

        # convert to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # resample data to the target_sr
        wav = resample(wav, orig_freq=sr, new_freq=self.target_sr)
        # OR
        # transform = Resample(orig_freq=sr, new_freq=self.target_sr)
        # wav = transform(wav)

        # slice the audio based on start and end samples
        # if audio_type is vocals
        if self.audio_type == "vocals":
            start_sample = self.start_samples[index]
            end_sample = self.end_samples[index]
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
