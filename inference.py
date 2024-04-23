import torch
import torchaudio
from customConvNet import CustomConvNet
from gtzanDataset import idx2lbl, GtzanDataset
from train import ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES

# def mapping_classes(num_classes, classes):
#   return None

def predict(model, input, target, class_mapping):
  model.eval()
  with torch.no_grad():
    predictions = model(input)
    predicted_index = predictions[0].argmax(0)
    predicted = class_mapping[predicted_index]
    expected = class_mapping[target]
  return predicted, expected

if __name__ == "__main__":
  # load back the model
  customCNN = CustomConvNet()
  state_dict = torch.load("./CNNCUSTOM.pth")
  customCNN.load_state_dict(state_dict)
  
  if torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"
  print(f"Using device: {device}")
  
  # loading dataset
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
  
  
  input, target = dataset[0][0], dataset[0][1]
  input.unsqueeze_(0)

  predicted, expected = predict(customCNN, input, target, idx2lbl)
  print(f"Predicted: {predicted}, expected:{expected}")
  