import torch
import torchaudio
import os
from jiwer import wer

from greedyctc import GreedyCTCDecoder


def get_data(data_path):
    # This function loads the audio files and corresponding transcripts from the specified data path

    # Define an empty list to store the data
    data_list = []

    # Iterate over all files in the data path
    for i in range(63):
        # Construct the paths for the audio and transcript files
        audio_file_path = os.path.join(data_path, f"61-70968-{i:04d}.flac")
        transcript_file_path = os.path.join(data_path, f"{i:04d}.txt")

        # Check if the audio file exists
        if os.path.exists(audio_file_path):
            # Read the transcript file content
            with open(transcript_file_path, "r", encoding="utf-8") as transcript_file:
                transcript_content = transcript_file.read()

            # Create a dictionary containing the audio and transcript data
            data = {"audio": audio_file_path, "text": transcript_content}

            # Append the data dictionary to the data list
            data_list.append(data)
        else:
            print(f"File {audio_file_path} not found")

    return data_list


def evaluate(data_list, model):
    # This function evaluates the given model on the specified data list and computes the average Word Error Rate (WER)

    # Initialize variables to store the total WER and total items
    total_wer = 0.0
    total_items = 0

    # Iterate over each data item
    for item in data_list:
        # Extract the audio file path and transcript
        audio_path = item["audio"]
        text = item["text"]

        # Load the audio waveform and convert it to tensors
        waveform, sample_rate = torchaudio.load(audio_path, format="flac")
        waveform = waveform.to(device)

        # Perform speech recognition using the model
        with torch.inference_mode():
            emission, _ = model(waveform)

        # Decode the CTC output to obtain the transcription
        transcript = Decoder(emission[0])
        transcript = transcript.replace(
            "|", " "
        )  # Replace the CTC separator with a space

        # Calculate the WER between the predicted and reference transcripts
        wer_score = wer(text.lower(), transcript.lower())

        # Update the total WER and total items
        total_wer += wer_score
        total_items += 1

        # Print the file information, transcript, and WER
        print(
            "File:",
            os.path.basename(audio_path),
            "Transcript:",
            transcript,
            "Reference Text:",
            text,
            f"WER: {wer_score:.2f}",
        )

    # Compute the average WER
    average_wer = (total_wer / total_items) * 100
    print("Average Word Error Rate:", average_wer)


if __name__ == "__main__":
    DATA_PATH = (
        "D:\\Vscode_Projects\\python-tutorial\\70968"  # Replace with your data path
    )

    # Load the pre-trained speech recognition model
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    Decoder = GreedyCTCDecoder(labels=bundle.get_labels())

    # Load the data
    data_list = get_data(DATA_PATH)

    # Evaluate the model on the loaded data
    evaluate(data_list, model)
