# Speech-to-Text with Wav2Vec2 and Greedy CTC Decoder

This project demonstrates speech-to-text transcription using the Wav2Vec2 model from Hugging Face and a Greedy CTC Decoder. It loads audio files and corresponding text transcripts, computes the Word Error Rate (WER) for each transcription, and calculates the average WER across all samples.

## Setup

### Requirements

- Python 3.6+
- PyTorch
- torchaudio
- jiwer


Install dependencies using pip:

```bash
pip install torch torchaudio jiwer 

git clone <repository_url>
cd <repository_folder>

python speech_to_text.py
