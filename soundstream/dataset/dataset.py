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
        # we are directly selecting the 1.5 second clip, and then perform the mean and resample ops
        # the same can be done for vocals too but first we need to convert the start and end sample at 16kHz(target_sr) to 44.1kHz(sr)
        if self.audio_type == "instrumentals":
            wav_len = wav.size(1)  # 44100*n
            seg_dur = self.max_len / self.target_sr  # 1.5
            seg_size = int(sr * seg_dur)  # 66150
            if wav_len > seg_size:
                max_start = wav_len - seg_size
                start = torch.randint(0, max_start, (1,))
                wav = wav[:, start : start + seg_size]
            else:
                wav = F.pad(wav, (0, seg_size - wav_len), "constant")

        elif self.audio_type == "vocals":
            start_sample = self.start_samples[index]
            start_sample = (start_sample * sr) // 16000

            end_sample = self.end_samples[index]
            end_sample = (end_sample * sr) // 16000

            wav = wav[:, start_sample:end_sample]

            wav_len = wav.size(1)  # 44100*n
            seg_dur = self.max_len / self.target_sr  # 1.5
            seg_size = int(sr * seg_dur)  # 66150
            if wav_len > seg_size:
                max_start = wav_len - seg_size
                start = torch.randint(0, max_start, (1,))
                wav = wav[:, start : start + seg_size]
            else:
                wav = F.pad(wav, (0, seg_size - wav_len), "constant")

        # convert to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # resample data to the target_sr
        wav = resample(wav, orig_freq=sr, new_freq=self.target_sr) # this offers more speed
        # OR
        # transform = Resample(orig_freq=sr, new_freq=self.target_sr)
        # wav = transform(wav)
        return wav
