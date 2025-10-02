import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import correlate
import csv
import os

class ExtractFeature:
    def __init__(self, input_path: str, output_path: str, sample_rate, duration):
        """
        Khởi tạo đối tượng trích xuất đặc trưng từ file tín hiệu nhị phân.

        Args:
            input_path (str): Đường dẫn đến file tín hiệu nhị phân (I/Q).
            output_path (str): Đường dẫn file CSV đầu ra để lưu đặc trưng.
            sample_rate (int): Tốc độ lấy mẫu tín hiệu (Hz).
            duration (float): Thời lượng mỗi frame (s).
        """
        self.input_path = input_path
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.duration = duration

        # ánh xạ file nhị phân -> int16
        self.mm = np.memmap(self.input_path, dtype=np.int16, mode='r')

        # tổng số mẫu int16
        total_samples = self.mm.shape[0]
        # tổng số mẫu phức (I+Q)
        self.total_complex_samples = total_samples // 2
        # số mẫu mỗi frame
        self.frame_samples = int(self.duration * self.sample_rate)
        # tổng số frame
        self.total_frames = self.total_complex_samples // self.frame_samples

    def read_frame(self, frame_idx):
        """
        Đọc một frame dữ liệu I/Q từ file nhị phân tại chỉ số frame cho trước.

        Args:
            frame_idx (int): Chỉ số frame cần đọc (bắt đầu từ 0).

        Returns:
            np.ndarray: Mảng 1-D gồm các mẫu phức (I + jQ) của frame.
        """
        start = frame_idx * self.frame_samples * 2  # mỗi mẫu phức = 2 int16
        end = start + self.frame_samples * 2
        raw_data = self.mm[start:end]

        I = raw_data[0::2].astype(np.float32)
        Q = raw_data[1::2].astype(np.float32)
        return I + 1j * Q

    def extract_features_to_csv(self, samples, mode='a'):
        """
        Tính toán các đặc trưng miền thời gian, miền tần số, năng lượng, 
        SNR, pha... từ một frame và ghi kết quả vào file CSV.

        Args:
            samples (np.ndarray): Mảng phức 1-D chứa dữ liệu tín hiệu I/Q.
            mode (str, optional): Chế độ ghi file CSV, 'a' (append) hoặc 'w' (overwrite).
                Mặc định là 'a'.
        """
        features = {}

        I = samples.real
        Q = samples.imag
        amplitude = np.abs(samples)
        power = amplitude**2
        eps = np.finfo(float).eps

        # ===== Miền thời gian =====
        features['mean_amplitude'] = np.mean(amplitude)
        features['variance_amplitude'] = np.var(amplitude)
        features['skewness_amplitude'] = skew(amplitude)
        features['kurtosis_amplitude'] = kurtosis(amplitude)
        features['std_amplitude'] = np.std(amplitude)
        features['rms_amplitude'] = np.sqrt(np.mean(amplitude**2))
        features['crest_factor'] = np.max(amplitude) / (features['rms_amplitude'] + eps)
        features['papr_time'] = np.max(power) / (np.mean(power) + eps)
        prob = power / (np.sum(power) + eps)
        features['entropy_time'] = -np.sum(prob * np.log2(prob + eps))

        # Zero Crossing Rate
        features['zcr_time'] = np.sum(np.abs(np.diff(np.sign(I)))) / len(I)

        # ===== Miền tần số =====
        fft_signal = np.fft.fft(samples)
        fft_magnitude = np.abs(fft_signal)
        fft_magnitude = np.nan_to_num(fft_magnitude)

        features['mean_fft'] = np.mean(fft_magnitude)
        features['variance_fft'] = np.var(fft_magnitude)
        features['skewness_fft'] = skew(fft_magnitude)
        features['kurtosis_fft'] = kurtosis(fft_magnitude)
        prob_fft = fft_magnitude / (np.sum(fft_magnitude) + eps)
        features['entropy_fft'] = -np.sum(prob_fft * np.log2(prob_fft + eps))
        features['peak_to_mean_fft'] = np.max(fft_magnitude) / (np.mean(fft_magnitude) + eps)

        # ===== Tự tương quan và năng lượng =====
        signal_energy = np.sum(power)
        features['signal_energy'] = signal_energy

        ac = correlate(amplitude, amplitude, mode='full')
        mid = len(ac) // 2
        features['autocorr_lag1'] = ac[mid + 1] / (ac[mid] + eps)
        features['autocorr_lag2'] = ac[mid + 2] / (ac[mid] + eps)

        # ===== SNR ước lượng =====
        signal_power = signal_energy / len(amplitude)
        noise_power = np.var(amplitude - np.mean(amplitude))
        features['snr_estimate'] = 10 * np.log10((signal_power + eps) / (noise_power + eps))

        # ===== Pha =====
        phi = np.angle(samples)
        features['mean_phase'] = np.mean(phi)
        features['variance_phase'] = np.var(phi)

        unwrapped_phi = np.unwrap(phi)
        phase_diff = np.diff(unwrapped_phi)
        features['phase_noise'] = np.std(phase_diff)

        # ===== Ghi CSV =====
        file_exists = os.path.exists(self.output_path)
        with open(self.output_path, mode, newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=features.keys())
            if not file_exists or mode == 'w':
                writer.writeheader()
            writer.writerow(features)

    def extract_all(self):
        """
        Lặp qua toàn bộ các frame, trích xuất đặc trưng và ghi tuần tự vào CSV.

        Returns:
            None
        """
        print("Đang xử lý...")
        mode = 'w'
        for frame_idx in range(self.total_frames):
            samples = self.read_frame(frame_idx)
            self.extract_features_to_csv(samples, mode=mode)
            mode = 'a'
        print("Hoàn thành!")

    def info(self):
        """
        Hiển thị thông tin về file tín hiệu.

        Bao gồm:
            - Kích thước file (bytes)
            - Tổng số mẫu
            - Số frame
            - Thời lượng tín hiệu (s)

        Returns:
            None
        """
        file_size = os.path.getsize(self.input_path)
        total_samples = self.mm.shape[0]
        total_complex_samples = self.total_complex_samples
        total_frames = self.total_frames
        total_duration = total_complex_samples / self.sample_rate

        print(f"Kích thước file: {file_size:.2f} byte ({file_size / (1024 ** 3):.2f} GB)")
        print(f"Tổng số mẫu (int16): {total_samples:,} giá trị")
        print(f"Tổng số mẫu phức (I + jQ): {total_complex_samples:,} mẫu")
        print(f"Tổng số khung (frame {self.duration:.3f}s): {total_frames:,} khung")
        print(f"Tổng thời lượng tín hiệu: {total_duration:.2f} giây")
