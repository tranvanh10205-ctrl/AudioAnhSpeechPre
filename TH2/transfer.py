import os
import subprocess

folder = r"D:\ATvsTN_TH2\data"

# Thư mục chứa các file WAV đầu ra
output_folder = r"D:\ATvsTN_TH2\data_wav"
# Tạo thư mục nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Đường dẫn trực tiếp tới ffmpeg.exe
ffmpeg = r"D:\Downnload\ffmpeg-9.0.2-essentials_build\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe"

for file in os.listdir(folder):

    if file.lower().endswith(".m4a"):

        input_file = os.path.join(folder, file)

        # Lưu WAV vào thư mục output_folder
        output_file = os.path.join(
            output_folder,
            os.path.splitext(file)[0] + ".wav"
        )

        print(f"Đang chuyển: {file}")

        subprocess.run([
            ffmpeg,
            "-i", input_file,
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            output_file,
            "-y"
        ])

        print(f"✓ Xong: {output_file}")

print("Hoàn thành!")