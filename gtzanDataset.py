import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import os

ANNOTATIONS_FILE = "/home/rthr/Documentos/UFMG/2024-1/TransferMusic/Data/GTZAN/features_30_sec.csv"
AUDIO_DIR = "/home/rthr/Documentos/UFMG/2024-1/TransferMusic/Data/GTZAN/genres_original"
SAMPLE_RATE = 16000
NUM_SAMPLES = 696319

lbl2idx = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9,
}

idx2lbl = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock",
}


class GtzanDataset(Dataset):

    def __init__(
        self,
        annotations_file,
        audio_dir,
        transfromation,
        target_sample_rate,
        target_num_samples,
        device
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transfromation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.target_num_samples = target_num_samples
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self.transformation(signal)

        return signal, label
    
    def _get_audio_sample_path(self, index):
        fold = f"{self.annotations.iloc[index, 59]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return lbl2idx[self.annotations.iloc[index, 59]]

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)

        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.target_num_samples:
            num_missing_samples = self.target_num_samples - length_signal
            last_dim_padding  = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.target_num_samples:
            signal = signal[:, :self.target_num_samples]
        
        return signal

if __name__ == "__main__":
    # USAGE EXAMPLE

    # ANNOTATIONS_FILE = "/path/to/annotation/file.csv"
    # AUDIO_DIR = "path/to/audio_files/direct"
    # SAMPLE_RATE = 16000
    # NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    mel_spactrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=96
    )


    dataset = GtzanDataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR, 
        mel_spactrogram, 
        SAMPLE_RATE, 
        NUM_SAMPLES,
        device)

    print(f"There are {len(dataset)} samples in the dataset.")

    signal, label = dataset[0]
    print(dataset[0][0].shape)