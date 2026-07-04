from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.io.wavfile import write

from bigvgan import BigVGAN
from meldataset import MAX_WAV_VALUE, get_mel_spectrogram


def load_filelist(filelist_path):
    with open(filelist_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def resolve_input_wav(wavs_dir, entry):
    relative_path = entry.split("|")[0] + ".wav"
    return os.path.join(wavs_dir, relative_path)


def load_model(model_id, use_cuda_kernel, device):
    model = BigVGAN.from_pretrained(model_id, use_cuda_kernel=use_cuda_kernel)
    model.remove_weight_norm()
    return model.eval().to(device)


def synthesize_filelist(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)
    if device.type == "cuda":
        torch.cuda.manual_seed(1234)

    model = load_model(a.model_id, a.use_cuda_kernel, device)
    model_h = model.h

    filelist = load_filelist(a.filelist)
    os.makedirs(a.output_dir, exist_ok=True)
    os.makedirs(a.mel_dir, exist_ok=True)

    with torch.inference_mode():
        for index, entry in enumerate(filelist):
            wav_path = resolve_input_wav(a.input_wavs_dir, entry)
            if not os.path.isfile(wav_path):
                raise FileNotFoundError(f"Missing wav file: {wav_path}")

            wav, _ = librosa.load(wav_path, sr=model_h.sampling_rate, mono=True)
            wav_tensor = torch.FloatTensor(wav).unsqueeze(0).to(device)

            mel = get_mel_spectrogram(wav_tensor, model_h).to(device)
            y_hat = model(mel)

            mel_name = Path(entry.split("|")[0]).name + ".npy"
            mel_path = os.path.join(a.mel_dir, mel_name)
            np.save(mel_path, mel.squeeze(0).cpu().numpy())

            audio = y_hat.squeeze().mul(MAX_WAV_VALUE).cpu().numpy().astype("int16")
            output_name = Path(entry.split("|")[0]).name + "_generated.wav"
            output_path = os.path.join(a.output_dir, output_name)
            write(output_path, model_h.sampling_rate, audio)

            print(f"[{index + 1}/{len(filelist)}] {wav_path} -> {output_path} (mel: {mel_path})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", required=True, help="LibriTTS filelist such as dev-clean.txt or dev-other.txt")
    parser.add_argument("--input_wavs_dir", required=True, help="Root directory that contains the wav paths referenced by the filelist")
    parser.add_argument("--output_dir", default="generated_wavs", help="Directory to write synthesized wavs")
    parser.add_argument("--mel_dir", default="generated_mels", help="Directory to write corresponding mel .npy files")
    parser.add_argument(
        "--model_id",
        default="nvidia/bigvgan_v2_24khz_100band_256x",
        help="Pretrained BigVGAN model id or local checkpoint directory containing config.json and bigvgan_generator.pt",
    )
    parser.add_argument("--use_cuda_kernel", action="store_true", default=False, help="Use the fused CUDA inference kernel if available")

    args = parser.parse_args()
    synthesize_filelist(args)


if __name__ == "__main__":
    main()