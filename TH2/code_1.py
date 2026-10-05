"""
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI
LAB 2: ĐẶC TRƯNG TIẾNG NÓI VÀ NHẬN DẠNG BẰNG DTW

Chạy trên VS Code:
    python lab2.py

Cấu trúc dataset:
dataset/
    khong/
        khong_01.wav ... khong_05.wav
    mot/
        mot_01.wav ... mot_05.wav
    hai/
        hai_01.wav ... hai_05.wav
    ba/
        ba_01.wav ... ba_05.wav
    bon/
        bon_01.wav ... bon_05.wav

Mặc định:
- 16 kHz, mono
- frame 25 ms = 400 samples
- hop 10 ms = 160 samples
- Hamming
- pre-emphasis alpha = 0.97
- NFFT = 512
- 24 Mel filters
- 13 MFCC
- Euclidean local distance
- DTW tự cài đặt
- 3 file đầu/lớp = template, phần còn lại = test
"""

from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

from scipy.signal import lfilter
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay


# ============================================================
# 0. CẤU HÌNH
# ============================================================

FS = 16000

FRAME_MS = 25
HOP_MS = 10

WIN = int(FS * FRAME_MS / 1000)      # 400
HOP = int(FS * HOP_MS / 1000)       # 160

ALPHA = 0.97
NFFT = 512
N_MELS = 24
N_MFCC = 13

# Endpoint detection:
TOP_DB = 35
MARGIN_MS = 50

# 3 file đầu/lớp làm template, các file còn lại làm test
N_TEMPLATES = 3

DATASET_DIR = Path("dataset")
FIG_DIR = Path("figures")
TRIM_DIR = Path("trimmed")

LABELS = ["khong", "mot", "hai", "ba", "bon"]

VN_LABEL = {
    "khong": "không",
    "mot": "một",
    "hai": "hai",
    "ba": "ba",
    "bon": "bốn",
}


# ============================================================
# 1. TIỆN ÍCH
# ============================================================

def ensure_dirs():
    FIG_DIR.mkdir(exist_ok=True)
    TRIM_DIR.mkdir(exist_ok=True)


def normalize_audio(y):
    """Chuẩn hóa biên độ về khoảng xấp xỉ [-1, 1]."""
    y = y.astype(np.float32)
    max_amp = np.max(np.abs(y)) if len(y) else 0.0
    if max_amp > 0:
        y = y / (max_amp + 1e-9)
    return y


def load_audio(path):
    """
    Đọc WAV, mono, resample về 16 kHz.
    librosa.load(sr=FS, mono=True) đảm bảo pipeline thống nhất.
    """
    y, sr = librosa.load(str(path), sr=FS, mono=True)
    y = normalize_audio(y)
    return y


def frame_signal(y):
    """
    Chia tín hiệu thành các frame chồng lấn.
    Trả về:
        frames: shape (T, WIN)
        times: thời gian bắt đầu mỗi frame.
    """
    if len(y) < WIN:
        y = np.pad(y, (0, WIN - len(y)))

    n_frames = 1 + int(np.floor((len(y) - WIN) / HOP))

    frames = np.zeros((n_frames, WIN), dtype=np.float32)

    for r in range(n_frames):
        start = r * HOP
        frames[r] = y[start:start + WIN]

    times = np.arange(n_frames) * HOP / FS
    return frames, times


def hamming_window():
    return np.hamming(WIN).astype(np.float32)


# ============================================================
# A. KIỂM TRA DỮ LIỆU + WAVEFORM
# ============================================================

def plot_waveform(path, save_name=None):
    y = load_audio(path)
    t = np.arange(len(y)) / FS

    plt.figure(figsize=(12, 4))
    plt.plot(t, y)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform: {path.name}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_name:
        plt.savefig(FIG_DIR / save_name, dpi=150)
    plt.show()


def inspect_dataset():
    print("\n========== A. KIỂM TRA DATASET ==========")

    total = 0

    for label in LABELS:
        files = sorted((DATASET_DIR / label).glob("*.wav"))

        print(f"\n{label} ({VN_LABEL.get(label, label)}): {len(files)} file")

        if len(files) == 0:
            print("  [WARNING] Không có file WAV.")
            continue

        for f in files:
            y = load_audio(f)

            peak = np.max(np.abs(y))
            duration = len(y) / FS
            clipping = np.sum(np.abs(y) >= 0.999)

            print(
                f"  {f.name:20s} "
                f"duration={duration:6.3f}s | "
                f"peak={peak:.3f} | "
                f"clipping_samples={clipping}"
            )

            total += 1

    print(f"\nTổng số file: {total}")

    # Vẽ ít nhất 3 waveform
    plotted = 0

    for label in LABELS:
        files = sorted((DATASET_DIR / label).glob("*.wav"))

        for f in files[:1]:
            plot_waveform(
                f,
                save_name=f"01_waveform_{label}.png"
            )
            plotted += 1

            if plotted >= 3:
                return


# ============================================================
# B. SHORT-TIME ENERGY / RMS / ZCR
# ============================================================

def compute_energy_rms_zcr(y):
    """
    Tính:
        Energy_r = sum(x_r[n]^2)
        RMS_r    = sqrt(mean(x_r[n]^2))
        ZCR_r    = số lần đổi dấu / số mẫu

    Dùng Hamming window cho energy/RMS.
    """
    frames, times = frame_signal(y)
    w = hamming_window()

    windowed = frames * w

    energy = np.sum(windowed ** 2, axis=1)

    rms = np.sqrt(
        np.mean(windowed ** 2, axis=1) + 1e-12
    )

    zcr = np.zeros(len(frames))

    for i, frame in enumerate(frames):
        signs = np.where(frame >= 0, 1, -1)

        crossings = np.sum(
            np.abs(signs[1:] - signs[:-1])
        ) / 2.0

        zcr[i] = crossings / len(frame)

    log_energy_db = 10 * np.log10(energy + 1e-12)

    return times, energy, rms, zcr, log_energy_db


def plot_energy_zcr(path, save_name=None):
    y = load_audio(path)

    times, energy, rms, zcr, log_energy = \
        compute_energy_rms_zcr(y)

    t = np.arange(len(y)) / FS

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    axes[0].plot(t, y)
    axes[0].set_title(f"Waveform - {path.name}")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(times, log_energy)
    axes[1].set_title("Short-time log-energy")
    axes[1].set_ylabel("Energy (dB)")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(times, zcr)
    axes[2].set_title("Zero-Crossing Rate")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("ZCR")
    axes[2].grid(True, alpha=0.25)

    plt.tight_layout()

    if save_name:
        plt.savefig(FIG_DIR / save_name, dpi=150)

    plt.show()


def run_part_B():
    print("\n========== B. ENERGY / RMS / ZCR ==========")

    count = 0

    for label in LABELS:
        files = sorted((DATASET_DIR / label).glob("*.wav"))

        for f in files[:1]:
            plot_energy_zcr(
                f,
                save_name=f"02_energy_zcr_{label}.png"
            )

            count += 1

            if count >= 3:
                return


# ============================================================
# C. ENDPOINT DETECTION
# ============================================================

def trim_energy_zcr(
    y,
    top_db=TOP_DB,
    margin_ms=MARGIN_MS
):
    """
    Endpoint detection thủ công dựa trên log-energy.

    Ý tưởng:
    1. Tính log-energy từng frame.
    2. Lấy mức tham chiếu từ các frame đầu/cuối.
    3. Frame nào vượt ngưỡng tương đối -> speech.
    4. Thêm margin để tránh cắt phụ âm đầu/cuối.
    5. ZCR được dùng như tín hiệu hỗ trợ để quan sát/tinh chỉnh.

    Đây là heuristic, không phải ngưỡng tuyệt đối.
    """

    frames, times = frame_signal(y)

    w = hamming_window()
    windowed = frames * w

    energy = np.sum(windowed ** 2, axis=1)
    log_energy = 10 * np.log10(energy + 1e-12)

    n_ref = max(1, min(5, len(log_energy) // 10))

    # Ước lượng noise floor từ đầu và cuối
    edge_values = np.concatenate([
        log_energy[:n_ref],
        log_energy[-n_ref:]
    ])

    noise_floor = np.median(edge_values)

    threshold = noise_floor + top_db

    speech_idx = np.where(log_energy > threshold)[0]

    if len(speech_idx) == 0:
        return y, (0, len(y)), {
            "noise_floor_db": noise_floor,
            "threshold_db": threshold,
            "start_frame": 0,
            "end_frame": len(frames) - 1
        }

    start_frame = speech_idx[0]
    end_frame = speech_idx[-1]

    # Margin để tránh cắt mất phụ âm
    margin_samples = int(FS * margin_ms / 1000)

    start_sample = max(
        0,
        start_frame * HOP - margin_samples
    )

    end_sample = min(
        len(y),
        end_frame * HOP + WIN + margin_samples
    )

    y_trim = y[start_sample:end_sample]

    return y_trim, (start_sample, end_sample), {
        "noise_floor_db": noise_floor,
        "threshold_db": threshold,
        "start_frame": start_frame,
        "end_frame": end_frame
    }


def save_trimmed_audio(path):
    y = load_audio(path)

    y_trim, (s, e), info = trim_energy_zcr(y)

    out_dir = TRIM_DIR / path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / path.name

    sf.write(out_path, y_trim, FS)

    print(
        f"{path.name}: "
        f"{len(y)/FS:.3f}s -> {len(y_trim)/FS:.3f}s | "
        f"threshold={info['threshold_db']:.2f} dB"
    )

    return y_trim, out_path


def run_part_C():
    print("\n========== C. ENDPOINT DETECTION ==========")

    count = 0

    for label in LABELS:
        files = sorted((DATASET_DIR / label).glob("*.wav"))

        for f in files[:1]:
            save_trimmed_audio(f)
            count += 1

            if count >= 3:
                return


# ============================================================
# D. MFCC
# ============================================================

def pre_emphasis(y, alpha=ALPHA):
    """
    y[n] = x[n] - alpha*x[n-1]
    """
    return lfilter(
        [1.0, -alpha],
        [1.0],
        y
    )


def mfcc_feature(
    y,
    use_cmn=True,
    use_delta=False
):
    """
    Pipeline:
        pre-emphasis
        -> framing/window
        -> FFT/power spectrum
        -> Mel filterbank
        -> log
        -> DCT
        -> CMN tùy chọn
        -> delta tùy chọn

    Kết quả cuối:
        shape = (T, D)
    trong đó:
        T = số frame
        D = 13 hoặc 26 nếu thêm delta
    """

    y = pre_emphasis(y)

    M = librosa.feature.mfcc(
        y=y,
        sr=FS,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
        n_fft=NFFT,
        win_length=WIN,
        hop_length=HOP,
        window="hamming",
        center=False
    )

    # M có shape (13, T)
    if use_cmn:
        M = M - np.mean(
            M,
            axis=1,
            keepdims=True
        )

    if use_delta:
        delta = librosa.feature.delta(M)

        # 13 MFCC + 13 delta = 26 chiều
        M = np.vstack([M, delta])

    # DTW cần mỗi hàng là một frame
    return M.T


def plot_mfcc(path, use_delta=False, save_name=None):
    y = load_audio(path)
    y, _, _ = trim_energy_zcr(y)

    X = mfcc_feature(
        y,
        use_cmn=True,
        use_delta=use_delta
    )

    plt.figure(figsize=(12, 5))

    plt.imshow(
        X.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest"
    )

    plt.colorbar(label="MFCC value")
    plt.xlabel("Frame")
    plt.ylabel("MFCC coefficient")
    plt.title(
        f"MFCC: {path.name} | shape={X.shape}"
    )

    plt.tight_layout()

    if save_name:
        plt.savefig(FIG_DIR / save_name, dpi=150)

    plt.show()


def run_part_D():
    print("\n========== D. MFCC ==========")

    count = 0

    for label in LABELS:
        files = sorted((DATASET_DIR / label).glob("*.wav"))

        for f in files[:1]:
            y = load_audio(f)
            y_trim, _, _ = trim_energy_zcr(y)

            X = mfcc_feature(y_trim)

            print(
                f"{f.name}: "
                f"MFCC shape = {X.shape}"
            )

            plot_mfcc(
                f,
                save_name=f"03_mfcc_{label}.png"
            )

            count += 1

            if count >= 2:
                return


# ============================================================
# E. EUCLIDEAN + DTW TỰ CÀI ĐẶT
# ============================================================

def euclidean_distance(x, y):
    """
    d(x,y) = sqrt(sum((x_q-y_q)^2))
    """
    return np.linalg.norm(x - y)


def local_distance_matrix(X, Y):
    """
    C[i,j] = Euclidean distance giữa frame i của X
    và frame j của Y.
    """
    N = len(X)
    M = len(Y)

    C = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            C[i, j] = euclidean_distance(
                X[i],
                Y[j]
            )

    return C


def dtw_distance(X, Y):
    """
    DTW tự cài đặt bằng dynamic programming.

    X: (N, D)
    Y: (M, D)

    D[i,j] =
        C[i-1,j-1]
        + min(
            D[i-1,j],
            D[i,j-1],
            D[i-1,j-1]
          )

    Trả về:
        dtw_norm
        optimal path
        local distance matrix C
        accumulated matrix D
    """

    N = len(X)
    M = len(Y)

    C = local_distance_matrix(X, Y)

    # D có thêm hàng/cột biên
    D = np.full(
        (N + 1, M + 1),
        np.inf
    )

    D[0, 0] = 0.0

    back = np.zeros(
        (N + 1, M + 1, 2),
        dtype=int
    )

    for i in range(1, N + 1):
        for j in range(1, M + 1):

            choices = [
                D[i - 1, j],       # bước dọc
                D[i, j - 1],       # bước ngang
                D[i - 1, j - 1]    # bước chéo
            ]

            best_index = int(
                np.argmin(choices)
            )

            best = choices[best_index]

            if best_index == 0:
                pi, pj = i - 1, j
            elif best_index == 1:
                pi, pj = i, j - 1
            else:
                pi, pj = i - 1, j - 1

            D[i, j] = C[i - 1, j - 1] + best

            back[i, j] = [pi, pj]

    # Backtracking
    path = []

    i, j = N, M

    while i > 0 or j > 0:
        path.append(
            (i - 1, j - 1)
        )

        i, j = back[i, j]

    path.reverse()

    # Chuẩn hóa theo độ dài path
    dtw_norm = D[N, M] / max(len(path), 1)

    return dtw_norm, path, C, D


def plot_dtw(X, Y, title, save_name=None):
    distance, path, C, D = dtw_distance(X, Y)

    plt.figure(figsize=(8, 7))

    plt.imshow(
        C,
        origin="lower",
        aspect="auto",
        interpolation="nearest"
    )

    path_arr = np.array(path)

    plt.plot(
        path_arr[:, 1],
        path_arr[:, 0],
        linewidth=2
    )

    plt.xlabel("Frame Y")
    plt.ylabel("Frame X")

    plt.title(
        f"{title}\nDTW_norm = {distance:.4f}"
    )

    plt.colorbar(
        label="Euclidean local distance"
    )

    plt.tight_layout()

    if save_name:
        plt.savefig(FIG_DIR / save_name, dpi=150)

    plt.show()

    return distance


def run_part_E():
    print("\n========== E. DTW ==========")

    files = {}

    for label in LABELS:
        fs = sorted(
            (DATASET_DIR / label).glob("*.wav")
        )

        if len(fs) >= 2:
            files[label] = fs

    if not files:
        print("Không đủ dữ liệu.")
        return

    label = next(iter(files))

    f1 = files[label][0]
    f2 = files[label][1]

    y1 = load_audio(f1)
    y2 = load_audio(f2)

    y1, _, _ = trim_energy_zcr(y1)
    y2, _, _ = trim_energy_zcr(y2)

    X = mfcc_feature(y1)
    Y = mfcc_feature(y2)

    print(f"Cùng từ: {label}")

    d_same = plot_dtw(
        X,
        Y,
        f"DTW cùng từ: {label}",
        "04_dtw_same_word.png"
    )

    # So sánh với một từ khác
    other_label = None

    for lab in files:
        if lab != label:
            other_label = lab
            break

    if other_label:
        f3 = files[other_label][0]

        y3 = load_audio(f3)
        y3, _, _ = trim_energy_zcr(y3)

        Z = mfcc_feature(y3)

        d_diff = plot_dtw(
            X,
            Z,
            f"DTW khác từ: {label} vs {other_label}",
            "05_dtw_different_word.png"
        )

        print(f"DTW_norm cùng từ : {d_same:.4f}")
        print(f"DTW_norm khác từ  : {d_diff:.4f}")


# ============================================================
# F. NEAREST-TEMPLATE RECOGNIZER
# ============================================================

def extract_feature_from_file(
    path,
    use_trim=True,
    use_delta=False
):
    y = load_audio(path)

    if use_trim:
        y, _, _ = trim_energy_zcr(y)

    return mfcc_feature(
        y,
        use_cmn=True,
        use_delta=use_delta
    )


def build_templates(
    use_trim=True,
    use_delta=False
):
    """
    Chọn N_TEMPLATES file đầu mỗi lớp.

    Quan trọng:
    Không dùng các file test làm template.
    """

    templates = {
        label: []
        for label in LABELS
    }

    for label in LABELS:

        files = sorted(
            (DATASET_DIR / label).glob("*.wav")
        )

        if len(files) < N_TEMPLATES:
            print(
                f"[WARNING] {label} có "
                f"{len(files)} file < {N_TEMPLATES}"
            )

        train_files = files[:N_TEMPLATES]

        for f in train_files:

            feature = extract_feature_from_file(
                f,
                use_trim=use_trim,
                use_delta=use_delta
            )

            templates[label].append(
                feature
            )

    return templates


def recognize(
    path,
    templates,
    use_trim=True,
    use_delta=False
):
    X = extract_feature_from_file(
        path,
        use_trim=use_trim,
        use_delta=use_delta
    )

    scores = {}

    for label, refs in templates.items():

        if len(refs) == 0:
            scores[label] = np.inf
            continue

        template_scores = []

        for R in refs:

            d, _, _, _ = dtw_distance(
                X,
                R
            )

            template_scores.append(d)

        # D_w(X) = min_r DTW_norm(X, T_w,r)
        scores[label] = min(
            template_scores
        )

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1]
    )

    prediction = sorted_scores[0][0]

    return prediction, sorted_scores


def run_recognizer(
    use_trim=True,
    use_delta=False
):
    templates = build_templates(
        use_trim=use_trim,
        use_delta=use_delta
    )

    rows = []

    print("\nTên file | True | Pred | Top-3")

    for label in LABELS:

        files = sorted(
            (DATASET_DIR / label).glob("*.wav")
        )

        # Test = các file sau N_TEMPLATES
        test_files = files[N_TEMPLATES:]

        for f in test_files:

            pred, scores = recognize(
                f,
                templates,
                use_trim=use_trim,
                use_delta=use_delta
            )

            top3 = scores[:3]

            print(
                f"{f.name:20s} | "
                f"{label:6s} | "
                f"{pred:6s} | "
                f"{top3}"
            )

            rows.append({
                "file": f.name,
                "true": label,
                "pred": pred,
                "top1_score": scores[0][1],
                "top2_score": scores[1][1]
                    if len(scores) > 1 else np.nan
            })

    return rows


# ============================================================
# G. ACCURACY + CONFUSION MATRIX
# ============================================================

def evaluate_rows(
    rows,
    title,
    save_name
):
    y_true = [
        r["true"]
        for r in rows
    ]

    y_pred = [
        r["pred"]
        for r in rows
    ]

    if not y_true:
        print("Không có test sample.")
        return 0.0

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS
    )

    print(
        f"\n{title}"
    )

    print(
        f"Accuracy = {accuracy * 100:.2f}%"
    )

    print("\nConfusion matrix:")
    print(cm)

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            VN_LABEL.get(x, x)
            for x in LABELS
        ]
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(
        FIG_DIR / save_name,
        dpi=150
    )
    plt.show()

    return accuracy


def save_results_csv(rows, filename):
    """
    Lưu results.csv mà không cần pandas.
    """
    import csv

    path = Path(filename)

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "true",
                "pred",
                "top1_score",
                "top2_score"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Đã lưu: {path}")


# ============================================================
# THÍ NGHIỆM E1: NO TRIM vs TRIM
# ============================================================

def experiment_E1():
    print(
        "\n========== E1: NO TRIM vs TRIM =========="
    )

    # Không trim
    rows_no_trim = run_recognizer(
        use_trim=False,
        use_delta=False
    )

    acc_no_trim = evaluate_rows(
        rows_no_trim,
        "E1 - Không endpoint detection",
        "06_confusion_no_trim.png"
    )

    # Có trim
    rows_trim = run_recognizer(
        use_trim=True,
        use_delta=False
    )

    acc_trim = evaluate_rows(
        rows_trim,
        "E1 - Có endpoint detection",
        "07_confusion_trim.png"
    )

    print(
        "\nSo sánh E1:"
    )

    print(
        f"Không trim: {acc_no_trim * 100:.2f}%"
    )

    print(
        f"Có trim   : {acc_trim * 100:.2f}%"
    )


# ============================================================
# THÍ NGHIỆM E2: MFCC 13 vs MFCC + DELTA
# ============================================================

def experiment_E2():
    print(
        "\n========== E2: MFCC 13 vs MFCC + DELTA =========="
    )

    rows_mfcc = run_recognizer(
        use_trim=True,
        use_delta=False
    )

    acc_mfcc = evaluate_rows(
        rows_mfcc,
        "E2 - MFCC 13",
        "08_confusion_mfcc13.png"
    )

    rows_delta = run_recognizer(
        use_trim=True,
        use_delta=True
    )

    acc_delta = evaluate_rows(
        rows_delta,
        "E2 - MFCC + Delta",
        "09_confusion_mfcc_delta.png"
    )

    print(
        "\nSo sánh E2:"
    )

    print(
        f"MFCC 13       : {acc_mfcc * 100:.2f}%"
    )

    print(
        f"MFCC + Delta   : {acc_delta * 100:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_dirs()

    print("=" * 70)
    print("CSE457 - LAB 2: MFCC + DTW")
    print("=" * 70)

    if not DATASET_DIR.exists():
        print(
            f"\n[ERROR] Không tìm thấy thư mục: "
            f"{DATASET_DIR.resolve()}"
        )

        print(
            "\nHãy tạo dataset theo dạng:"
        )

        print(
            """
dataset/
    khong/
    mot/
    hai/
    ba/
    bon/
            """
        )

        return

    # A
    inspect_dataset()

    # B
    run_part_B()

    # C
    run_part_C()

    # D
    run_part_D()

    # E
    run_part_E()

    # F + G baseline
    print(
        "\n========== F + G: BASELINE =========="
    )

    rows = run_recognizer(
        use_trim=True,
        use_delta=False
    )

    evaluate_rows(
        rows,
        "Baseline - MFCC 13 + DTW",
        "10_confusion_baseline.png"
    )

    save_results_csv(
        rows,
        "results.csv"
    )

    # E1
    experiment_E1()

    # E2
    experiment_E2()

    print("\n" + "=" * 70)
    print("HOÀN THÀNH LAB 2")
    print("=" * 70)

    print(
        "\nCác thư mục/file kết quả:"
    )
    print("  figures/   -> các biểu đồ")
    print("  trimmed/   -> audio sau endpoint detection")
    print("  results.csv -> kết quả test")


if __name__ == "__main__":
    main()

