import speech_recognition as sr
import os

from jiwer import wer

# Initialize the recognizer
recognizer = sr.Recognizer()



def get_data():
    # تعریف مسیر پوشه حاوی تمام فایل‌ها
    folder_path = 'C:\\Users\\Kingpower\\Desktop\\audio_test\\p340'
    

    # لیستی برای ذخیره اطلاعات فایل‌ها
    data_list = []

    # گردش در تمام فایل‌ها
    for i in range(3, 424):
        # ساخت مسیر فایل تکست
        if i == 23:
            continue
        text_file_path = os.path.join(folder_path, f"p340_{i:03d}.txt")

        # ساخت مسیر فایل صوتی
        audio_file_path = os.path.join(folder_path, f"p340_{i:03d}_mic1.flac")

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
            print(f"فایل تکست p340_{i:03d}.txt یافت نشد.")
    return data_list

data_list = get_data()
# Iterate over each audio-text pair
total_wer = 0.0
total_items = 0
for item in data_list:
    audio_path = item["audio"]
    text = item["text"]

    # Load the audio file
    with sr.AudioFile(audio_path) as source:
        # Adjust for ambient noise
        # recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source)

        try:
            # Perform speech recognition
            transcript = recognizer.recognize_sphinx(audio)
            print("Transcript:", transcript)
            print("Reference Text:", text)
            
            # Calculate the Word Error Rate (WER)
            wer_score = wer(text.lower(), transcript.lower())
            print("WER:", wer_score)
            print("***************")

            total_wer += wer_score
            total_items += 1

        except sr.UnknownValueError:
            print(text)
            print("Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

# Compute the final average WER
average_wer = (total_wer / total_items) * 100
print("Average Word Error Rate:", average_wer)
