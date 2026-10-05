# CSE457 – Lab 1: Phân tích và xử lý tín hiệu âm thanh số

> **Môn học:** CSE457 – Xử lý âm thanh và tiếng nói  
> **Lab:** Lab 1 – Phân tích và xử lý tín hiệu âm thanh số  
> **Sinh viên:** `Trần Việt Anh`  
> **Môi trường:** Python 3.13 / VS Code / Jupyter Notebook  
> **Notebook:** `test.ipynb`

---

## 1. Giới thiệu

Lab 1 thực hiện một pipeline xử lý tín hiệu âm thanh số từ dữ liệu MP3 thực tế. Theo yêu cầu của tài liệu Lab, quá trình được triển khai theo chuỗi:

**Audio → biểu diễn số → miền thời gian → FFT → STFT/Spectrogram → Window → Filter → Quantization/Resampling → Coding/Evaluation**.

Trong bài này, file âm thanh được sử dụng là:

`tunetank-western-cowboy-duel-background-350113(1).mp3`

Notebook sử dụng `librosa` để đọc và phân tích âm thanh, `NumPy` cho tính toán số, `Matplotlib` cho trực quan hóa, `SciPy` cho thiết kế FIR filter và `SoundFile` để lưu các file WAV đã xử lý.

### Mục tiêu

- Kiểm tra metadata và biểu diễn tín hiệu âm thanh dưới dạng số.
- Phân tích tín hiệu trong miền thời gian bằng waveform, Peak, RMS và Energy.
- Phân tích miền tần số bằng FFT và hiểu frequency-bin spacing.
- Phân tích tín hiệu biến thiên theo thời gian bằng STFT/Spectrogram.
- So sánh Rectangular Window và Hamming Window.
- Thiết kế và áp dụng FIR Low-pass Filter 2 kHz.
- Thực nghiệm lượng tử hóa 4-bit, 8-bit và 16-bit, sau đó tính SNR.
- Resample âm thanh về 16 kHz và 8 kHz.
- Tính PCM bitrate, kích thước lý thuyết và compression ratio.

> **Lưu ý về dữ liệu:** Tài liệu PDF có một case study tham khảo tên `Western Cowboy Texas Music.mp3` với thời lượng khoảng 93.048 s. Notebook của bài này chạy trên một file `tunetank-western-cowboy-duel-background-350113(1).mp3` khác, có thời lượng khoảng 175.909 s.

---

## 2. Công cụ và thư viện

```text
Python 3.13
NumPy
Librosa
Matplotlib
SciPy
SoundFile
```

Cài đặt:

```bash
pip install numpy librosa matplotlib scipy soundfile
```

---

# A. Đọc và kiểm tra dữ liệu âm thanh

## A.1. Đọc file

File được đọc bằng:

```python
y, sr = librosa.load(audio_path, sr=None, mono=False)
```

`sr=None` giúp giữ nguyên sampling rate của file thay vì tự động resample về một sampling rate khác.

### Công thức liên quan

Sampling period:

$$
T = \frac{1}{F_s}
$$

Trong đó:

- $F_s$: sampling rate, đơn vị Hz.
- $T$: khoảng thời gian giữa hai mẫu liên tiếp.

Điều kiện Nyquist:

$$
F_s \ge 2F_{max}
$$

Do đó tần số Nyquist là:

$$
F_{Nyquist}=\frac{F_s}{2}
$$

Với $F_s=44,100$ Hz:

$$
F_{Nyquist}=22,050\ Hz
$$

Nghĩa là tín hiệu số có thể biểu diễn độc lập các thành phần tần số đến khoảng 22.05 kHz.

## A.2. Kết quả

| Thuộc tính | Kết quả từ `test.ipynb` |
|---|---:|
| Sampling rate | **44,100 Hz** |
| Số kênh | **2 (stereo)** |
| Thời lượng | **175.9086 s** |
| Kích thước file | **5,629,074 bytes ≈ 5.37 MiB** |
| Shape | **(2, 7,757,568)** |
| Dtype sau khi đọc bằng librosa | **float32** |
| Nyquist frequency | **22,050 Hz** |
| Mono shape | **(7,757,568,)** |

> `librosa.load()` giải mã tín hiệu thành mảng số thực `float32`, vì vậy `dtype` trong Notebook là `float32`. Đây không nên được hiểu trực tiếp là “file gốc có 32 bit/mẫu”. Bit depth/sample width của MP3 gốc cần đọc từ metadata codec nếu muốn báo cáo chính xác.

## A.3. Stereo → Mono

Tín hiệu stereo được chuyển thành mono bằng trung bình hai kênh:

$$
x_{mono}[n]=\frac{x_L[n]+x_R[n]}{2}
$$

Mục đích là tạo một tín hiệu một chiều thuận tiện cho các phép phân tích tiếp theo như FFT, STFT và filtering.

---

# B. Phân tích miền thời gian

Waveform biểu diễn biên độ của tín hiệu theo thời gian:

$$
 x[n] \leftrightarrow t=\frac{n}{F_s}
$$

## B.1. Waveform toàn bộ file

![Waveform toàn bộ file](readme_figures/01_waveform_full.png)

**Hình 1.** Waveform của toàn bộ tín hiệu âm thanh sau khi chuyển về mono.

### Nhận xét

Waveform cho thấy tín hiệu thay đổi đáng kể theo thời gian, biên độ không cố định. Một số vùng có biên độ lớn hơn, đặc biệt ở khoảng giữa và về cuối file, trong khi các vùng khác có năng lượng thấp hơn. Điều này cho thấy tín hiệu nhạc có đặc tính không dừng, vì vậy chỉ phân tích FFT trên toàn bộ file sẽ không cho biết chính xác thành phần tần số xuất hiện tại thời điểm nào.

## B.2. Peak

$$
Peak=\max_n|x[n]|
$$

Peak cho biết biên độ tuyệt đối lớn nhất của tín hiệu.

Kết quả:

```text
Peak = 1.003722
```

Giá trị này lớn hơn 1 một lượng nhỏ. Điều đó phù hợp với kết quả kiểm tra clipping bên dưới và cho thấy tín hiệu có một vài mẫu vượt mức full-scale chuẩn hóa.

## B.3. RMS

$$
RMS=\sqrt{\frac{1}{N}\sum_{n=0}^{N-1}x^2[n]}
$$

RMS biểu diễn mức năng lượng hiệu dụng của tín hiệu và thường phù hợp hơn Peak khi muốn mô tả mức tín hiệu trung bình.

Kết quả:

```text
RMS = 0.19058132
```

Nếu biểu diễn RMS theo dBFS:

$$
RMS_{dBFS}=20\log_{10}(RMS)
$$

với full-scale bằng 1.

## B.4. Energy

$$
E=\sum_{n=0}^{N-1}x^2[n]
$$

Kết quả:

```text
Energy = 281764.47
```

Energy phụ thuộc vào cả biên độ và số lượng mẫu. Vì vậy khi so sánh hai đoạn có cùng thời lượng, Energy có thể được dùng để nhận biết đoạn nào chứa nhiều năng lượng hơn.

## B.5. Kiểm tra clipping

Clipping được kiểm tra bằng điều kiện:

$$
|x[n]|\ge1
$$

Kết quả:

```text
Number of clipping samples = 3
```

Chỉ có 3 mẫu đạt/vượt ngưỡng 1 trong toàn bộ tín hiệu. Đây là số lượng rất nhỏ so với tổng số mẫu, nhưng vẫn cho thấy tín hiệu có một vài điểm chạm hoặc vượt full-scale.

## B.6. So sánh hai đoạn tín hiệu

Hai đoạn 1 giây được chọn tại **10–11 s** và **45–46 s**.

| Đoạn | Peak | RMS | Energy |
|---|---:|---:|---:|
| 10–11 s | 0.621057 | 0.163603 | 1180.3811 |
| 45–46 s | 0.660220 | 0.137166 | 829.72046 |

### Nhận xét

- Đoạn 45–46 s có **Peak lớn hơn** đoạn 10–11 s.
- Tuy nhiên, đoạn 10–11 s có **RMS và Energy lớn hơn**.
- Điều này cho thấy Peak chỉ phản ánh một giá trị cực đại, còn RMS/Energy phản ánh mức năng lượng của cả đoạn tín hiệu.
- Vì vậy không nên kết luận một đoạn “mạnh hơn” chỉ dựa trên Peak.

---

# C. Phân tích miền tần số bằng FFT

## C.1. Chọn đoạn phân tích

Theo yêu cầu của Lab, một đoạn ổn định khoảng 0.5–1 s được chọn. Notebook sử dụng đoạn:

```text
45–46 s
```

Đoạn này được nhân với cửa sổ Hamming trước khi thực hiện FFT.

## C.2. Cửa sổ Hamming

Công thức Hamming:

$$
 w[n]=0.54-0.46\cos\left(\frac{2\pi n}{L-1}\right)
$$

Cửa sổ làm giảm sự gián đoạn ở hai đầu frame, từ đó giảm spectral leakage khi tín hiệu được phân tích bằng FFT.

## C.3. DFT và FFT

DFT của frame $N$ mẫu:

$$
X[k]=\sum_{n=0}^{N-1}x[n]e^{-j2\pi kn/N}
$$

FFT là thuật toán tính DFT hiệu quả hơn, với độ phức tạp xấp xỉ:

$$
O(N\log N)
$$

Tần số của bin thứ $k$:

$$
f_k=\frac{kF_s}{NFFT}
$$

Khoảng cách giữa hai frequency bin:

$$
\Delta f=\frac{F_s}{NFFT}
$$

Với $F_s=44,100$ Hz và $NFFT=4096$:

$$
\Delta f=\frac{44100}{4096}\approx10.767\ Hz
$$

## C.4. So sánh NFFT

| NFFT | Frequency-bin spacing |
|---:|---:|
| 2048 | **21.533 Hz** |
| 4096 | **10.767 Hz** |
| 8192 | **5.383 Hz** |

### NFFT = 2048

![FFT 2048](readme_figures/02_fft_2048.png)

**Hình 2.** FFT spectrum với NFFT = 2048.

### NFFT = 4096

![FFT 4096](readme_figures/03_fft_4096.png)

**Hình 3.** FFT spectrum với NFFT = 4096.

### NFFT = 8192

![FFT 8192](readme_figures/04_fft_8192.png)

**Hình 4.** FFT spectrum với NFFT = 8192.

### Nhận xét

Khi tăng NFFT từ 2048 → 4096 → 8192, khoảng cách giữa các điểm trên trục tần số giảm lần lượt từ khoảng 21.53 Hz xuống 10.77 Hz và 5.38 Hz. Vì vậy đường phổ được lấy mẫu dày hơn và các đỉnh có thể được quan sát chi tiết hơn.

Tuy nhiên, cần phân biệt **frequency-bin spacing** với **true frequency resolution**. Nếu độ dài frame vật lý không thay đổi và chỉ tăng NFFT bằng zero-padding, ta không tạo thêm thông tin mới. Độ phân giải vật lý chủ yếu phụ thuộc vào thời lượng frame/cửa sổ.

## C.5. Các đỉnh phổ

Trong tài liệu Lab, yêu cầu chỉ ra ít nhất 3 đỉnh nổi bật. Với kết quả tham chiếu của case study trong PDF, các đỉnh được báo cáo khoảng 147.4, 221.4, 293.4, 441.4, 590.8 và 738.2 Hz. Tuy nhiên, **đây là số liệu của case study trong PDF, không phải kết quả đo trực tiếp từ file `tunetank...mp3` trong Notebook**. Vì vậy trong báo cáo thực nghiệm này không gán các tần số đó cho file hiện tại nếu chưa chạy peak detection riêng.

> Nếu giảng viên yêu cầu bắt buộc đánh dấu 3 peak trên chính hình FFT của file hiện tại, cần bổ sung một cell peak-detection vào `test.ipynb` rồi cập nhật Hình 2–4.

---

# D. STFT và Spectrogram

FFT cho biết thành phần tần số của một frame, nhưng âm thanh thực tế thay đổi theo thời gian. Vì vậy cần dùng STFT.

## D.1. Công thức STFT

$$
X[m,k]=\sum_nx[n]w[n-mH]e^{-j2\pi kn/NFFT}
$$

Trong đó:

- $m$: chỉ số frame theo thời gian.
- $k$: frequency bin.
- $w$: cửa sổ phân tích.
- $H$: hop size.
- `NFFT`: số điểm FFT.

Spectrogram thường được biểu diễn theo dB:

$$
S_{dB}[m,k]=20\log_{10}(|X[m,k]|+\epsilon)
$$

## D.2. Cấu hình chuẩn 25 ms / 10 ms

Với $F_s=44,100$ Hz:

$$
L=0.025F_s\approx1102\ samples
$$

$$
H=0.010F_s\approx441\ samples
$$

Kết quả Notebook:

```text
Frame length = 1102 samples
Hop length   = 441 samples
```

Overlap:

$$
Overlap=L-H=1102-441=661\ samples
$$

Tương đương khoảng 60% overlap.

## D.3. Spectrogram 25 ms / 10 ms

![Spectrogram 25ms 10ms](readme_figures/05_spectrogram_25ms_10ms.png)

**Hình 5.** Spectrogram với frame ≈ 25 ms và hop ≈ 10 ms.

### Nhận xét

Spectrogram cho thấy năng lượng tập trung mạnh ở vùng tần số thấp và thay đổi theo thời gian. Có những vùng năng lượng tương đối ổn định và các vùng có biến đổi nhanh hơn. Điều này minh họa lý do cần STFT thay vì chỉ dùng một FFT cho toàn bộ file.

## D.4. Trade-off frame length

Theo yêu cầu PDF, cần so sánh ít nhất ba frame length: **10 ms, 25 ms và 50 ms**.

| Frame | Đặc điểm kỳ vọng |
|---|---|
| 10 ms | Phân giải thời gian tốt, dễ quan sát transient; phân giải tần số kém hơn |
| 25 ms | Cân bằng giữa thời gian và tần số; cấu hình chuẩn của Lab |
| 50 ms | Phân giải tần số tốt hơn; các biến đổi nhanh theo thời gian dễ bị làm mờ |

> **Trạng thái Notebook:** `test.ipynb` hiện đã tạo spectrogram 25 ms / 10 ms. Chưa có ba hình riêng cho 10/25/50 ms trong output hiện tại. Khi hoàn thiện đúng yêu cầu PDF, nên bổ sung thêm hai hình cho 10 ms và 50 ms hoặc tạo một figure 3 panel.

---

# E. Thí nghiệm cửa sổ: Rectangular vs Hamming

Thí nghiệm giữ nguyên frame 45–46 s và `NFFT = 4096`, chỉ thay đổi cửa sổ.

### Rectangular Window

$$
 w[n]=1
$$

### Hamming Window

$$
 w[n]=0.54-0.46\cos\left(\frac{2\pi n}{L-1}\right)
$$

![Rectangular vs Hamming](readme_figures/06_rectangular_vs_hamming.png)

**Hình 6.** So sánh phổ của Rectangular Window và Hamming Window.

### Nhận xét

- Rectangular giữ nguyên toàn bộ mẫu trong frame nhưng có sidelobe cao hơn.
- Hamming làm giảm sidelobe nên giảm spectral leakage.
- Đổi lại, Hamming có main-lobe rộng hơn.
- Vì main-lobe rộng hơn, hai đỉnh tần số nằm rất gần nhau có thể khó phân biệt hơn.
- Thí nghiệm là controlled experiment vì cùng dữ liệu và cùng NFFT, chỉ thay đổi window.

Do đó, không có cửa sổ “tốt tuyệt đối”; lựa chọn phụ thuộc vào mục tiêu phân tích giữa giảm leakage và khả năng phân tách các thành phần tần số gần nhau.

---

# F. Lọc số bằng FIR

## F.1. Cơ sở lý thuyết

Với hệ LTI:

$$
y[n]=x[n]*h[n]=\sum_kx[k]h[n-k]
$$

Đối với FIR bậc $M$:

$$
y[n]=\sum_{r=0}^{M}b_rx[n-r]
$$

Đáp ứng tần số:

$$
H(e^{j\omega})=\sum_{r=0}^{M}b_re^{-j\omega r}
$$

## F.2. FIR Low-pass 2 kHz

Notebook thiết kế filter với:

```text
Type       : FIR Low-pass
Number taps: 201
Cutoff     : 2000 Hz
Window     : Hamming
Sampling   : 44100 Hz
```

Code thiết kế:

```python
b_lpf = signal.firwin(
    numtaps=201,
    cutoff=2000,
    fs=sr,
    window="hamming"
)
```

Notebook thu được **201 coefficients**. 10 hệ số đầu:

```text
[-5.56928888e-05,
  1.65011438e-05,
  8.97404129e-05,
  1.59235324e-04,
  2.20091630e-04,
  2.67507779e-04,
  2.97028077e-04,
  3.04853661e-04,
  2.88200031e-04,
  2.45675511e-04]
```

## F.3. Frequency response

![FIR frequency response](readme_figures/07_fir_lpf_response.png)

**Hình 7.** Đáp ứng tần số của FIR Low-pass 2 kHz.

Từ đồ thị có thể thấy vùng tần số thấp được giữ tương đối tốt, trong khi năng lượng ở vùng cao hơn cutoff bị suy giảm mạnh. Điều này phù hợp với chức năng của low-pass filter.

## F.4. Group delay

Với FIR đối xứng có $L=201$ taps:

$$
D=\frac{L-1}{2}=100\ samples
$$

Đổi sang thời gian:

$$
D_t=\frac{100}{44100}\approx2.27\ ms
$$

Đây là độ trễ nhóm xấp xỉ của FIR tuyến tính pha.

## F.5. So sánh phổ trước và sau lọc

![Spectrum before and after LPF](readme_figures/08_spectrum_before_after_lpf.png)

**Hình 8.** So sánh phổ đoạn 45–46 s trước và sau Low-pass 2 kHz.

### Nhận xét

Sau khi lọc, thành phần tần số cao giảm rõ rệt so với tín hiệu gốc. Sự thay đổi này phù hợp với frequency response của filter: các thành phần dưới vùng cutoff được giữ lại tốt hơn, trong khi thành phần trên cutoff bị suy giảm.

### File audio đầu ra

Notebook đã xuất:

```text
audio/filtered_lpf_2k.wav
```

> **Lưu ý về yêu cầu PDF:** PDF yêu cầu ít nhất **01 FIR low-pass và 01 high-pass/band-pass**. `test.ipynb` hiện mới có kết quả cho **low-pass 2 kHz**. Không nên ghi rằng đã hoàn thành HPF/BPF nếu chưa có code và audio output tương ứng.

---

# G. Lượng tử hóa, Resampling và Coding

## G.1. Lượng tử hóa

Với $B$ bit:

$$
L=2^B
$$

là số mức lượng tử.

Sai số lượng tử:

$$
 e[n]=\hat{x}[n]-x[n]
$$

SNR đo được:

$$
SNR=10\log_{10}\left(\frac{\sum x^2[n]}{\sum(\hat{x}[n]-x[n])^2}\right)
$$

Theo mô hình lượng tử đều, tài liệu Lab cho:

$$
SNR_Q(dB)=6B+4.77-20\log_{10}\left(\frac{X_{max}}{\sigma_x}\right)
$$

Trong điều kiện phù hợp, tăng 1 bit thường làm SNR tăng xấp xỉ 6 dB.

## G.2. Kết quả SNR

| Bit depth | SNR đo được |
|---:|---:|
| 4-bit | **13.47 dB** |
| 8-bit | **38.59 dB** |
| 16-bit | **86.50 dB** |

### Nhận xét

SNR tăng mạnh khi số bit tăng. Khi số mức lượng tử tăng, bước lượng tử nhỏ hơn nên sai số lượng tử giảm. Kết quả thực nghiệm cũng thể hiện xu hướng này: 4-bit có SNR thấp nhất, 8-bit cao hơn và 16-bit cao nhất.

### File đầu ra

Notebook đã tạo các file:

```text
audio/quantized_4bit.wav
audio/quantized_8bit.wav
audio/quantized_16bit.wav
```

> **Lưu ý kỹ thuật quan trọng:** Notebook lưu các tín hiệu lượng tử hóa bằng `subtype="PCM_16"`. Vì vậy, đây là **tín hiệu đã được mô phỏng lượng tử hóa 4/8/16-bit nhưng container WAV được lưu ở PCM 16-bit**. Không nên mô tả ba file này là file PCM vật lý lần lượt 4/8/16-bit nếu chưa thay đổi cách ghi file.

---

# G.3. Resampling

Notebook sử dụng:

```python
y_16k = librosa.resample(
    y_mono,
    orig_sr=sr,
    target_sr=16000
)

 y_8k = librosa.resample(
    y_mono,
    orig_sr=sr,
    target_sr=8000
)
```

và lưu:

```text
audio/resampled_16k.wav
audio/resampled_8k.wav
```

Khi giảm sampling rate, cần anti-aliasing trước khi downsample. `librosa.resample()` sử dụng quy trình resampling phù hợp thay vì chỉ lấy mỗi mẫu thứ $k$.

Theo Nyquist:

- 16 kHz → Nyquist = 8 kHz.
- 8 kHz → Nyquist = 4 kHz.

Vì vậy khi chuyển từ 44.1 kHz xuống 16 kHz hoặc 8 kHz, các thành phần vượt quá Nyquist mới phải được loại bỏ/giảm trước khi giảm sampling rate để tránh aliasing.

> **Trạng thái Notebook:** đã tạo và lưu audio 16 kHz và 8 kHz, nhưng output hiện tại chưa có figure phổ trước/sau resampling. Nếu cần đáp ứng đầy đủ phần “so sánh phổ và nghe thử”, nên bổ sung một figure so sánh FFT hoặc spectrogram của 44.1 kHz, 16 kHz và 8 kHz.

---

# G.4. PCM bitrate

Bitrate PCM:

$$
R_{PCM}=F_s\times B\times C
$$

Với:

- $F_s=44,100$ Hz
- $B=16$ bit/sample
- $C=2$ channels

ta có:

$$
R_{PCM}=44100\times16\times2
$$

$$
R_{PCM}=1,411,200\ bit/s=1411.2\ kbps
$$

Kết quả Notebook:

```text
PCM bitrate = 1,411,200 bit/s
PCM bitrate = 1411.2 kbps
```

## G.5. Kích thước PCM lý thuyết

$$
Size\approx\frac{R_{PCM}\times Duration}{8}
$$

Với thời lượng khoảng 175.909 s, Notebook tính:

```text
Theoretical PCM size = 29.59 MB
```

Đây là kích thước lý thuyết của phần dữ liệu PCM, chưa tính thêm phần header/metadata của WAV.

---

# G.6. Compression ratio và storage saving

Theo tài liệu Lab:

$$
Compression\ Ratio=\frac{R_{uncompressed}}{R_{compressed}}
$$

Notebook sử dụng bitrate MP3:

```text
256 kbps
```

Do đó:

$$
Compression\ Ratio=\frac{1411.2}{256}\approx5.51:1
$$

Kết quả:

```text
Compression ratio = 5.51:1
```

Storage saving được tính bằng:

$$
Saving(\%)=\left(1-\frac{Size_{compressed}}{Size_{uncompressed}}\right)\times100
$$

Kết quả Notebook:

```text
Storage saving = 81.86%
```

Điều này cho thấy MP3 sử dụng ít dung lượng hơn nhiều so với PCM 16-bit stereo ở cùng sampling rate.

> **Lưu ý:** Trong `test.ipynb`, `mp3_bitrate = 256` được khai báo thủ công. Vì vậy compression ratio 5.51:1 là kết quả dựa trên giả định MP3 = 256 kbps. Nếu muốn báo cáo “bitrate thực tế của file”, nên đọc metadata bitrate của chính file MP3 trước khi kết luận.

---

# 3. Tổng hợp kết quả thực nghiệm

| Phần | Nội dung | Kết quả chính |
|---|---|---|
| A | Metadata | 44.1 kHz, stereo, 175.909 s, 5.37 MiB |
| B | Time-domain | Peak = 1.003722; RMS = 0.190581; Energy = 281764.47; 3 clipping samples |
| C | FFT | Δf = 21.533 / 10.767 / 5.383 Hz với NFFT 2048 / 4096 / 8192 |
| D | STFT | Frame = 1102 samples ≈ 25 ms; Hop = 441 samples ≈ 10 ms |
| E | Window | Rectangular vs Hamming; Hamming giảm sidelobe/leakage |
| F | FIR | Low-pass 2 kHz, 201 taps, group delay ≈ 2.27 ms |
| G | Quantization | SNR 4/8/16-bit = 13.47 / 38.59 / 86.50 dB |
| G | Resampling | Xuất 16 kHz và 8 kHz |
| G | PCM | 1411.2 kbps; theoretical size ≈ 29.59 MB |
| G | Compression | 5.51:1 và saving ≈ 81.86% theo bitrate MP3 256 kbps |

---

# 4. Trả lời các câu hỏi báo cáo

## Câu 1. Vì sao $F_s=44.1$ kHz chỉ biểu diễn độc lập đến 22.05 kHz?

Theo điều kiện Nyquist:

$$
F_{max}\le\frac{F_s}{2}
$$

Với $F_s=44.1$ kHz:

$$
F_{max}=\frac{44100}{2}=22050\ Hz=22.05\ kHz
$$

Các thành phần cao hơn giới hạn này không thể được biểu diễn độc lập và có thể gây aliasing nếu không được lọc chống aliasing.

## Câu 2. Tăng NFFT từ 2048 lên 8192 nhưng frame vẫn dài 25 ms thì điều gì thay đổi?

Frequency-bin spacing thay đổi:

$$
\Delta f=\frac{F_s}{NFFT}
$$

nên từ khoảng 21.53 Hz giảm xuống khoảng 5.38 Hz. Tuy nhiên, nếu chỉ zero-padding mà không tăng độ dài frame vật lý, không tạo thêm thông tin mới. Độ phân giải tần số thực vẫn chủ yếu phụ thuộc vào độ dài frame/window.

## Câu 3. Vì sao Hamming giảm leakage nhưng có thể làm đỉnh gần nhau khó phân tách hơn?

Hamming giảm sidelobe của phổ, do đó năng lượng từ một thành phần mạnh lan sang các vùng xung quanh ít hơn. Tuy nhiên, nó làm main-lobe rộng hơn so với rectangular window. Main-lobe rộng khiến hai thành phần tần số rất gần nhau có thể bị chồng lấn và khó tách hơn.

## Câu 4. FIR 201 taps có độ trễ bao nhiêu?

Với FIR đối xứng:

$$
D=\frac{L-1}{2}=\frac{201-1}{2}=100\ samples
$$

Tại 44.1 kHz:

$$
D_t=\frac{100}{44100}\approx2.27\ ms
$$

Đây là độ trễ tương đối nhỏ đối với nhiều ứng dụng offline, nhưng trong hệ thống thời gian thực, độ trễ vẫn cần được xem xét cùng với các nguồn latency khác.

## Câu 5. Ảnh hưởng của số bit và mức tín hiệu đến SNR lượng tử?

Số bit tăng làm số mức lượng tử tăng:

$$
L=2^B
$$

Do đó bước lượng tử nhỏ hơn và sai số lượng tử giảm. Theo công thức trong tài liệu, mỗi bit bổ sung thường làm SNR tăng khoảng 6 dB trong điều kiện phù hợp.

Nếu mức tín hiệu đầu vào giảm nhưng mức full-scale của bộ lượng tử không đổi, tín hiệu sử dụng ít hơn các mức lượng tử. Sai số lượng tử không giảm tương ứng với tín hiệu, vì vậy tỷ lệ signal-to-quantization-noise có thể giảm.

## Câu 6. WAV 16-bit stereo 44.1 kHz dài 60 s có kích thước PCM lý thuyết bao nhiêu?

$$
R=44100\times16\times2=1,411,200\ bit/s
$$

Trong 60 s:

$$
Size=\frac{1,411,200\times60}{8}=10,584,000\ bytes
$$

Tương đương khoảng:

$$
10.09\ MiB
$$

Nếu so sánh với MP3 128 kbps, bitrate MP3 nhỏ hơn đáng kể so với 1411.2 kbps của PCM. Compression ratio lý thuyết theo bitrate là:

$$
\frac{1411.2}{128}\approx11.03:1
$$

## Câu 7. Vì sao “nghe tốt hơn” không đồng nghĩa với “SNR lớn hơn”?

SNR chỉ đo tỷ lệ năng lượng tín hiệu so với năng lượng sai số theo một cách định lượng cụ thể. Chất lượng nghe còn phụ thuộc vào:

1. Thành phần tần số của nhiễu.
2. Nhiễu có nằm trong vùng tai người nhạy hay không.
3. Masking và đặc tính cảm nhận của thính giác.
4. Các biến dạng hoặc artefact mà SNR đơn thuần không mô tả đầy đủ.

Do đó một tín hiệu có SNR cao hơn không nhất thiết luôn tạo cảm nhận chủ quan tốt hơn.

---

# 6. Cấu trúc thư mục đề xuất trên GitHub

```text
Lab01_MSSV_HoTen/
│
├── README.md
├── test.ipynb
│
├── audio/
│   ├── input.mp3
│   ├── filtered_lpf_2k.wav
│   ├── quantized_4bit.wav
│   ├── quantized_8bit.wav
│   ├── quantized_16bit.wav
│   ├── resampled_16k.wav
│   └── resampled_8k.wav
│
└── readme_figures/
    ├── 01_waveform_full.png
    ├── 02_fft_2048.png
    ├── 03_fft_4096.png
    ├── 04_fft_8192.png
    ├── 05_spectrogram_25ms_10ms.png
    ├── 06_rectangular_vs_hamming.png
    ├── 07_fir_lpf_response.png
    └── 08_spectrum_before_after_lpf.png
```

---

# 7. Các điểm cần hoàn thiện trước khi nộp

Dựa trực tiếp trên yêu cầu trong PDF và output hiện tại của `test.ipynb`, còn một số phần nên bổ sung để Notebook đáp ứng đầy đủ ma trận kết quả tối thiểu:

### 7.1. STFT

Bổ sung spectrogram cho:

- 10 ms frame.
- 25 ms frame.
- 50 ms frame.

Giữ hop size hợp lý và cùng color scale/dynamic range khi so sánh.

### 7.2. FFT peak detection

Bổ sung code tự động tìm ít nhất 3 peak nổi bật trên chính file đang phân tích và đánh dấu trực tiếp trên đồ thị FFT. Không nên lấy các peak trong case study của PDF thay cho kết quả của file hiện tại.

### 7.3. FIR

PDF yêu cầu:

- ít nhất 01 Low-pass; và
- ít nhất 01 High-pass hoặc Band-pass.

Notebook hiện mới có **Low-pass 2 kHz**. Nếu muốn hoàn thành đầy đủ yêu cầu, cần thêm HPF hoặc BPF, vẽ frequency response và xuất audio.

### 7.4. Resampling

Đã tạo file 16 kHz và 8 kHz nhưng nên bổ sung figure so sánh phổ/spectrogram và ghi nhận xét về giới hạn Nyquist 8 kHz và 4 kHz.

### 7.5. Bitrate MP3

`test.ipynb` đang đặt:

```python
mp3_bitrate = 256
```

Vì vậy compression ratio = 5.51:1 đang dựa trên **256 kbps được khai báo**, không phải metadata được đọc tự động từ file. Nếu muốn báo cáo chính xác bitrate của file MP3, nên đọc metadata của file rồi cập nhật phép tính.

### 7.6. Quantization output

Các file 4/8/16-bit đang được lưu bằng `PCM_16`. SNR vẫn được tính đúng trên tín hiệu đã lượng tử hóa, nhưng nếu giảng viên yêu cầu file vật lý đúng bit depth thì cần điều chỉnh cách lưu file.

---

# 8. Kết luận

Qua Lab 1, tín hiệu MP3 đã được chuyển thành biểu diễn số và phân tích lần lượt trong miền thời gian, miền tần số và miền thời gian–tần số.

Ở miền thời gian, Peak, RMS và Energy cho thấy mức biên độ và năng lượng của tín hiệu thay đổi theo từng đoạn. Kết quả Peak = 1.003722 và 3 clipping samples cho thấy tín hiệu có một số mẫu vượt hoặc chạm full-scale.

Trong miền tần số, FFT cho thấy khi tăng NFFT từ 2048 lên 8192 thì frequency-bin spacing giảm từ khoảng 21.53 Hz xuống 5.38 Hz. Tuy nhiên, điều này cần được phân biệt với true frequency resolution vì zero-padding không tự tạo thêm thông tin tần số.

STFT với frame khoảng 25 ms và hop khoảng 10 ms cho thấy năng lượng của bản nhạc phân bố thay đổi theo thời gian. Việc thay đổi frame length tạo ra trade-off giữa time resolution và frequency resolution.

Thí nghiệm Rectangular và Hamming cho thấy Hamming làm giảm sidelobe và spectral leakage, nhưng main-lobe rộng hơn. Đây là sự đánh đổi quan trọng khi lựa chọn cửa sổ trong phân tích phổ.

FIR Low-pass 2 kHz với 201 taps cho thấy các thành phần tần số cao bị suy giảm rõ rệt. Filter có group delay khoảng 100 mẫu, tương đương 2.27 ms tại 44.1 kHz.

Cuối cùng, lượng tử hóa cho thấy SNR tăng khi số bit tăng: 13.47 dB ở 4-bit, 38.59 dB ở 8-bit và 86.50 dB ở 16-bit. Resampling tạo được các phiên bản 16 kHz và 8 kHz. Với PCM 16-bit stereo 44.1 kHz, bitrate lý thuyết là 1411.2 kbps và kích thước PCM lý thuyết của file khoảng 29.59 MB.

Nhìn chung, Lab minh họa được pipeline cơ bản của xử lý tín hiệu âm thanh số: **sampling → time-domain analysis → FFT → STFT → windowing → filtering → quantization → resampling → coding/evaluation**.

---



