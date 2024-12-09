import os
import numpy as np
from pydub import AudioSegment
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm
from colorama import Fore, Style, init
import matplotlib.pyplot as plt

# Initialize colorama
init()

def analyze_tuning(file_path):
    """
    Analyzes and extracts tuning information from an audio file.
    """
    try:
        # Load audio file
        audio = AudioSegment.from_file(file_path).set_channels(1)  # Mono
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        sample_rate = audio.frame_rate

        # Perform FFT to analyze frequency domain
        fft_vals = rfft(samples)
        fft_freqs = rfftfreq(len(samples), d=1 / sample_rate)

        # Focus on relevant range (400–500 Hz to capture harmonics)
        valid_indices = (fft_freqs >= 400) & (fft_freqs <= 500)
        filtered_fft_vals = np.abs(fft_vals[valid_indices])
        filtered_fft_freqs = fft_freqs[valid_indices]

        if len(filtered_fft_vals) > 0:
            # Dominant frequency (weighted average)
            dominant_freq = np.average(filtered_fft_freqs, weights=filtered_fft_vals)
            return {
                "dominant_freq": dominant_freq,
                "sample_rate": sample_rate,
                "duration": len(samples) / sample_rate,
                "harmonics": list(filtered_fft_freqs[np.argsort(-filtered_fft_vals)[:5]])  # Top 5 harmonics
            }
        return None
    except Exception as e:
        print(Fore.RED + f"Error analyzing {file_path}: {e}" + Style.RESET_ALL)
        return None


def convert_to_432hz(file_path, output_path, max_iterations=3, tolerance=0.5):
    """
    Converts an audio file to 432 Hz by adjusting playback speed iteratively.
    """
    try:
        # Detect initial tuning frequency
        tuning_info = analyze_tuning(file_path)
        if not tuning_info or "dominant_freq" not in tuning_info:
            print(Fore.RED + f"Could not analyze tuning for {file_path}. Skipping..." + Style.RESET_ALL)
            return False

        dominant_freq = tuning_info["dominant_freq"]
        print(Fore.YELLOW + f"Tuning info for {file_path}:" + Style.RESET_ALL)
        print(Fore.YELLOW + f"  Dominant Frequency: {dominant_freq:.2f} Hz" + Style.RESET_ALL)
        print(Fore.YELLOW + f"  Harmonics: {tuning_info['harmonics']}" + Style.RESET_ALL)

        # Skip upward tuning if already below 432 Hz
        if dominant_freq < 432:
            print(Fore.GREEN + f"{file_path} is already below 432 Hz. No upward tuning applied." + Style.RESET_ALL)
            return False

        # Iteratively adjust playback speed
        audio = AudioSegment.from_file(file_path)
        iteration = 0

        while iteration < max_iterations:
            speed_ratio = 432.0 / dominant_freq
            print(Fore.YELLOW + f"Iteration {iteration + 1}: Applying speed ratio {speed_ratio:.6f}" + Style.RESET_ALL)

            # Adjust playback speed
            audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * speed_ratio)
            }).set_frame_rate(audio.frame_rate)

            # Save to temporary path for verification
            temp_output_path = output_path.replace(".wav", f"_temp_{iteration}.wav")
            audio.export(temp_output_path, format="wav")

            # Re-analyze tuning
            tuning_info = analyze_tuning(temp_output_path)
            if not tuning_info or "dominant_freq" not in tuning_info:
                print(Fore.RED + f"Could not re-analyze tuning for iteration {iteration + 1}. Skipping..." + Style.RESET_ALL)
                break

            dominant_freq = tuning_info["dominant_freq"]
            print(Fore.YELLOW + f"Post-adjustment tuning: {dominant_freq:.2f} Hz" + Style.RESET_ALL)

            # Check if tuning is within tolerance
            if abs(dominant_freq - 432) <= tolerance:
                print(Fore.GREEN + f"Verification successful: {output_path} tuned to {dominant_freq:.2f} Hz." + Style.RESET_ALL)
                os.rename(temp_output_path, output_path)  # Rename temp file to final output
                return True

            iteration += 1

        # If iterations fail, save the last attempt
        print(Fore.RED + f"Verification failed after {max_iterations} iterations. Last tuning: {dominant_freq:.2f} Hz." + Style.RESET_ALL)
        audio.export(output_path, format="wav")
        return False
    except Exception as e:
        print(Fore.RED + f"Error converting {file_path}: {e}" + Style.RESET_ALL)
        return False



def batch_convert(input_folder, output_folder):
    """
    Converts all supported audio files in a folder to 432 Hz.
    """
    supported_formats = (".wav", ".mp3", ".flac", ".aac")
    audio_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_folder)
        for file in files if file.endswith(supported_formats)
    ]

    if not audio_files:
        print(Fore.RED + "No audio files found in the specified folder." + Style.RESET_ALL)
        return

    for file_path in tqdm(audio_files, desc="Processing files"):
        relative_path = os.path.relpath(file_path, input_folder)
        output_path = os.path.join(output_folder, relative_path)
        output_path = os.path.splitext(output_path)[0] + "_432Hz.wav"

        convert_to_432hz(file_path, output_path)


def main():
    """
    Main menu for the script.
    """
    while True:
        print(Fore.CYAN + "\n" + "-" * 40 + Style.RESET_ALL)
        print(Fore.BLUE + "Audio Tuning and Conversion Tool" + Style.RESET_ALL)
        print(Fore.CYAN + "-" * 40 + Style.RESET_ALL)
        print(Fore.YELLOW + "1. Analyze a single file for tuning" + Style.RESET_ALL)
        print(Fore.YELLOW + "2. Convert a single file to 432Hz" + Style.RESET_ALL)
        print(Fore.YELLOW + "3. Batch convert a folder to 432Hz" + Style.RESET_ALL)
        print(Fore.YELLOW + "4. Exit" + Style.RESET_ALL)
        print(Fore.CYAN + "-" * 40 + Style.RESET_ALL)
        choice = input(Fore.GREEN + "Enter your choice: " + Style.RESET_ALL)

        if choice == "1":
            file_path = input(Fore.GREEN + "Enter the path of the audio file: " + Style.RESET_ALL)
            if os.path.isfile(file_path):
                tuning_info = analyze_tuning(file_path)
                if tuning_info:
                    print(Fore.YELLOW + f"Dominant Frequency: {tuning_info['dominant_freq']:.2f} Hz" + Style.RESET_ALL)
                    print(Fore.YELLOW + f"Harmonics: {tuning_info['harmonics']}" + Style.RESET_ALL)
                else:
                    print(Fore.RED + "Could not analyze tuning." + Style.RESET_ALL)
            else:
                print(Fore.RED + "Invalid file path. Please try again." + Style.RESET_ALL)

        elif choice == "2":
            file_path = input(Fore.GREEN + "Enter the path of the audio file: " + Style.RESET_ALL)
            output_path = input(Fore.GREEN + "Enter the path to save the converted file: " + Style.RESET_ALL)
            if os.path.isfile(file_path):
                convert_to_432hz(file_path, output_path)
            else:
                print(Fore.RED + "Invalid file path. Please try again." + Style.RESET_ALL)

        elif choice == "3":
            input_folder = input(Fore.GREEN + "Enter the path of the folder containing audio files: " + Style.RESET_ALL)
            output_folder = input(Fore.GREEN + "Enter the path of the folder to save converted files: " + Style.RESET_ALL)
            if os.path.isdir(input_folder):
                batch_convert(input_folder, output_folder)
            else:
                print(Fore.RED + "Invalid folder path. Please try again." + Style.RESET_ALL)

        elif choice == "4":
            print(Fore.CYAN + "Exiting... Goodbye!" + Style.RESET_ALL)
            break

        else:
            print(Fore.RED + "Invalid choice. Please enter a valid option." + Style.RESET_ALL)


if __name__ == "__main__":
    main()
