import torch
import torchaudio
import os
from jiwer import wer

from greedyctc import GreedyCTCDecoder


def get_data():
    # تعریف مسیر پوشه حاوی تمام فایل‌ها
    folder_path = "D:\\Vscode_Projects\\python-tutorial\\70968"

    # لیستی برای ذخیره اطلاعات فایل‌ها
    data_list = []

    # گردش در تمام فایل‌ها
    for i in range(0, 63):
        # ساخت مسیر فایل تکست
        text_file_path = os.path.join(folder_path, f"{i:04d}.txt")

        # ساخت مسیر فایل صوتی
        audio_file_path = os.path.join(folder_path, f"61-70968-{i:04d}.flac")

        # بررسی وجود فایل تکست
        if os.path.exists(text_file_path):
            # خواندن محتوای فایل تکست
            with open(text_file_path, "r", encoding="utf-8") as text_file:
                text_content = text_file.read()

            # افزودن اطلاعات به لیست
            data_list.append(
                {
                    "text": text_content,
                    "audio": audio_file_path,
                }
            )
        else:
            print(f"فایل تکست p340{i:03d}.txt یافت نشد.")

    return data_list


data_list = get_data()

# Load the model and set the device
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
Decoder = GreedyCTCDecoder(labels=bundle.get_labels())

# Iterate over each audio-text pair
total_wer = 0.0
total_items = 0
for item in data_list:
    audio_path = item["audio"]
    text = item["text"]

    # Load the waveform
    waveform, sample_rate = torchaudio.load(audio_path, format="flac")
    waveform = waveform.to(device)

    # Perform classification using the model
    with torch.inference_mode():
        emission, _ = model(waveform)

    # Get the transcription from the CTC decoder
    transcript = Decoder(emission[0])
    transcript = transcript.replace("|", " ")
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
# Compute the final average WER
average_wer = (total_wer / total_items) * 100
print("Average Word Error Rate:", average_wer)
