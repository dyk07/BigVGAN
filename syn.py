import os
import torch
import bigvgan
import librosa
import soundfile as sf
from meldataset import get_mel_spectrogram   # 需要确保此模块在 Python 路径中

# ============ 配置参数 ============
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型选择（与你的 syn.py 一致）
model_name = 'nvidia/bigvgan_v2_24khz_100band_256x'

# 数据根目录（包含 dev-clean, dev-other 等文件夹）
data_root = 'C:\\Users\\Kelvin\\Documents\\GitHub\\vocoder\\LibriTTS'  # 修改为你的实际路径

# 输出根目录（生成的音频将保存在此处，保持子目录结构）
output_root = 'C:\\Users\\Kelvin\\Documents\\GitHub\\vocoder\\synthesized'  # 修改为你希望存放合成音频的目录

# 列表文件目录（包含 dev-clean.txt, dev-other.txt）
list_dir = 'filelists\\LibriTTS'   # 默认当前目录，可根据需要修改

# 要处理的子集列表
subsets = ['dev-clean', 'dev-other']
# ==================================

# 加载模型（只加载一次）
print("Loading model...")
model = bigvgan.BigVGAN.from_pretrained(
    'nvidia/bigvgan_v2_24khz_100band_256x',
    use_cuda_kernel=False
)
model.remove_weight_norm()
model = model.eval().to(device)
print("Model loaded.")

def process_subset(subset_name):
    """处理单个子集（如 dev-clean）"""
    list_file = os.path.join(list_dir, f"{subset_name}.txt")
    if not os.path.exists(list_file):
        print(f"Warning: {list_file} not found, skipping {subset_name}")
        return

    with open(list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total = len(lines)
    print(f"Processing {subset_name} with {total} utterances...")

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        # 格式: "path|text"，我们只取路径部分
        parts = line.split('|')
        rel_path = parts[0].strip()   # 例如 "dev-clean/1272/128104/1272_128104_000001_000000"
        
        # 构造完整的输入音频路径（原始 .wav 文件）
        input_wav_path = os.path.join(data_root, rel_path + '.wav')
        if not os.path.exists(input_wav_path):
            print(f"  Warning: {input_wav_path} not found, skipping.")
            continue

        # 构造输出路径（在 output_root 下保持相同子目录）
        output_wav_path = os.path.join(output_root, rel_path + '.wav')
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)

        # 如果输出文件已存在，可以跳过（避免重复处理）
        if os.path.exists(output_wav_path):
            # print(f"  Skipping {rel_path}, already exists.")
            continue

        try:
            # 1. 读取音频并重采样到模型所需的采样率
            wav, sr = librosa.load(input_wav_path, sr=model.h.sampling_rate, mono=True)
            wav_tensor = torch.FloatTensor(wav).unsqueeze(0).to(device)  # [1, T]

            # 2. 提取 Mel 谱图
            mel = get_mel_spectrogram(wav_tensor, model.h).to(device)

            # 3. 生成波形
            with torch.inference_mode():
                wav_gen = model(mel)  # [1, 1, T]

            # 4. 转换为 numpy 并保存
            wav_gen_float = wav_gen.squeeze(0).squeeze(0).cpu().numpy()  # [T]
            # 输出为 16-bit PCM（可选，根据需求）
            # wav_gen_int16 = (wav_gen_float * 32767).astype('int16')
            # sf.write(output_wav_path, wav_gen_int16, model.h.sampling_rate)
            # 或者保存为浮点数（范围 [-1,1]），soundfile 也支持
            sf.write(output_wav_path, wav_gen_float, model.h.sampling_rate)

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1}/{total} utterances.")

        except Exception as e:
            print(f"  Error processing {rel_path}: {e}")

    print(f"Finished {subset_name}.")


if __name__ == "__main__":
    for subset in subsets:
        process_subset(subset)
    print("All done.")