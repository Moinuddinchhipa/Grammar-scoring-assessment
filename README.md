# 🎧 Grammar Scoring Engine

A deep learning-based Grammar Scoring Engine built using PyTorch and Torchaudio. This project aims to evaluate spoken audio files (e.g., .wav) and assign grammar accuracy scores, useful in applications such as spoken English assessment, language learning apps, or interview feedback systems.

---

## 📌 Features

- 🧠 Deep learning model using PyTorch
- 🔊 Audio feature extraction with Torchaudio
- 📈 Regression-based scoring (outputs grammar scores)
- 📁 Dataset handling and preprocessing
- ⚙️ Training, Evaluation, and Prediction pipelines
- ✅ Extensible architecture for integration into larger systems

---

## 🛠 Tech Stack

- Python 3.8+
- PyTorch
- Torchaudio (with SoundFile backend)
- NumPy / Pandas / Scikit-learn
- SoundFile (for .wav file loading)
- Custom Dataset & DataLoader for audio fi

---

1. Clone the repository
git clone https://github.com/Moinuddinchhipa/Grammar-scoring-assessment/edit/main
cd GrammarScoringEngine

2. Set up a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Train the model
python main.py

How It Works
Audio Loading: Loads .wav files using Torchaudio with the soundfile backend.
Feature Extraction: Extracts MFCC features or raw waveforms.
Model: A neural network trained to regress grammar scores from audio features.
Evaluation: Uses RMSE and Pearson correlation to evaluate performance.

Model Architecture
Example ( main.py defines):
Input: MFCC features or waveform
Layers: Linear -> ReLU -> Dropout -> Linear
Output: Single score (float)
Modify create_model() function in main.py to experiment with architectures.

2. Outputs
Trained model: best_model.pt

Predictions on test data: submission.csv
🧪 Example Output
Epoch 1, Loss: 14.0984
...
Epoch 10, Loss: 1.1113
Train RMSE: 1.0478
Train Pearson Correlation: -0.0008
Model saved as best_model.pt
Submission saved to submission.csv

📊 Evaluation Metrics
RMSE (Root Mean Squared Error) for loss

Pearson Correlation Coefficient between predicted and actual scores

⚙️ Dependencies
Python 3.8+

PyTorch

torchaudio

pandas

numpy

scikit-learn



