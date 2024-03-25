import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from gtzanDataset import GtzanDataset
from customConvNet import CustomConvNet

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001
ANNOTATIONS_FILE = "/home/rthr/Documentos/UFMG/2024-1/TransferMusic/Data/GTZAN/features_30_sec.csv"
AUDIO_DIR = "/home/rthr/Documentos/UFMG/2024-1/TransferMusic/Data/GTZAN/genres_original"
SAMPLE_RATE = 16000
NUM_SAMPLES = 696319

def create_data_loader(train_data, batch_size):
  train_dataloader = DataLoader(train_data, batch_size=batch_size)
  return train_dataloader

def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
  for inputs, targets in data_loader:
    # print(f"inputs {inputs.shape}")
    inputs = inputs.to(device)
    targets = targets.to(device)
    # inputs, targets = inputs.to(device), targets.to(device)
    # Calculate loss
    predictions = model(inputs)
    loss = loss_fn(predictions, targets)
    # Backpropagate loss and update weights
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
  
  print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device, epochs):
  for i in range(epochs):
    print(f"Epoch: {i}")
    train_one_epoch(model, data_loader, loss_fn, optimiser, device)
    print("--------------------")

if __name__ == "__main__":
  # Select Device
  if torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"
  print(f"Using device: {device}")
  
  # Instantiating dataset object and creation data loader
  mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=2048,
    hop_length=512,
    n_mels=96
  )
  
  dataset = GtzanDataset(
    ANNOTATIONS_FILE,
    AUDIO_DIR,
    mel_spectrogram,
    SAMPLE_RATE,
    NUM_SAMPLES,
    device
  )
  train_dataloader = create_data_loader(dataset, BATCH_SIZE)
  
  # Build Model
  customCNN = CustomConvNet().to(device)
  print(customCNN)
  # Initiate loss_fn and optimiser
  loss_fn = nn.CrossEntropyLoss()
  optimiser = torch.optim.Adam(customCNN.parameters(), lr=LEARNING_RATE)
  
  # Train Model
  train(customCNN, train_dataloader, loss_fn, optimiser, device, EPOCHS)
  # Save Model
  torch.save(customCNN.state_dict(), "customCNN.pth")
  print("Model trained and stored at customCNN.pth")