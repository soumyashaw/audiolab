import os
from pydub import AudioSegment

def convert_flac_to_wav(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.flac'):
            flac_file_path = os.path.join(input_folder, filename)
            wav_file_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.wav')

            flac_audio = AudioSegment.from_file(flac_file_path, format='flac')
            flac_audio.export(wav_file_path, format='wav')

if __name__ == "__main__":
    input_folder = r"C:\Users\Dell\Documents\GitHub\audiolab\Auxillary Code\LCNN\project\03-asvspoof-mega\LA\LA\ASVspoof2019_LA_dev\flac"
    output_folder = r"C:\Users\Dell\Documents\GitHub\audiolab\Auxillary Code\LCNN\project\03-asvspoof-mega\DATA\asvspoof2019_LA\train_dev"
    convert_flac_to_wav(input_folder, output_folder)
