import os
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import torchaudio
torchaudio.set_audio_backend("soundfile")
# Set seed for reproducibility
torch.manual_seed(42)

# ---------------------------
# 1. Dataset
# ---------------------------
class GrammarDataset(Dataset):
    def __init__(self, csv_file, audio_dir, is_train=True):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.iloc[idx, 0]
        audio_path = os.path.join(self.audio_dir, filename)

        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0).unsqueeze(0)  # convert to mono

        # Resample to 16000 Hz
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

        # Pad or truncate to 4 sec (64000 samples)
        target_len = 64000
        if waveform.size(1) < target_len:
            pad_len = target_len - waveform.size(1)
            waveform = nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :target_len]

        if self.is_train:
            label = self.data.iloc[idx, 1]
            return waveform, torch.tensor(label, dtype=torch.float)
        else:
            return filename, waveform


# ---------------------------
# 2. Model
# ---------------------------
class GrammarScoreModel(nn.Module):
    def __init__(self):
        super(GrammarScoreModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(-1)
        return self.fc(x).squeeze(-1)


# ---------------------------
# 3. Train Function
# ---------------------------
def train_model(model, dataloader, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(10):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")


# ---------------------------
# 4. Evaluate Function
# ---------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            preds = model(inputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    corr, _ = pearsonr(all_labels, all_preds)
    print(f"Train RMSE: {rmse:.4f}")
    print(f"Train Pearson Correlation: {corr:.4f}")
    return all_preds, all_labels


# ---------------------------
# 5. Predict and Save Submission
# ---------------------------
def predict_and_save(model, dataloader, device, output_file):
    model.eval()
    results = []

    with torch.no_grad():
        for filenames, inputs in dataloader:
            inputs = inputs.to(device)
            preds = model(inputs).cpu().numpy()
            for fname, pred in zip(filenames, preds):
                results.append((fname, float(pred)))

    submission = pd.DataFrame(results, columns=["filename", "score"])
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")


# ---------------------------
# 6. Main
# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    train_csv = "train.csv"
    test_csv = "test.csv"
    train_audio = "audios/train"
    test_audio = "audios/test"

    # Datasets & Dataloaders
    train_dataset = GrammarDataset(train_csv, train_audio, is_train=True)
    test_dataset = GrammarDataset(test_csv, test_audio, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model
    model = GrammarScoreModel().to(device)

    # Train
    train_model(model, train_loader, device)

    # Evaluate
    evaluate_model(model, train_loader, device)

    # Save model
    torch.save(model.state_dict(), "best_model.pt")
    print("Model saved as best_model.pt")

    # Predict on test set
    predict_and_save(model, test_loader, device, "submission.csv")


if __name__ == "__main__":
    main()
