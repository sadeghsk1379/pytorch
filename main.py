import torch
import torchaudio
import os
from jiwer import wer
from torchaudio.models.decoder import ctc_decoder

Data = get_data_set()
# Load the model and set the device
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_100H
model = bundle.get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Create the decoder for WER calculation
decoder = ctc_decoder(labels=bundle.get_labels())
# Iterate over each audio-text pair
total_wer = 0.0
total_items = 0
for item in Data:
    audio_path = item["audio"]
    text = item["text"]

    # Load the waveform
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device)

    #  Perform classification using the model
    with torch.inference_mode():
        emission, _ = model(waveform)

    # Get the transcription from the emission using the decoder
    transcript = decoder(emission[0])
    # Calculate the Word Error Rate (WER)

    wer_score = wer(text.lower(), transcript.lower())
    total_wer += wer_score
    total_items += 1

    # Compute the average WER

    average_wer = (total_wer / total_items) * 100
    print(
        "File: ",
        os.path.basename(audio_path),
        "Transcript: ",
        transcript,
        " Reference Text: ",
        text,
        f"WER : {average_wer}",
    )
    # Compute the average WER
average_wer = total_wer / total_items
print("Average Word Error Rate:", average_wer)
