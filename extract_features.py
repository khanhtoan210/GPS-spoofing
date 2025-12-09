import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import signal, fft
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed
import os


class ExtractFeature:
    def __init__(self, input_path: str, output_path: str, sample_rate: int, duration: float = 0.02):
        self.input_path = input_path
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.duration = duration

        # Memory map binary file -> int16
        self.mm = np.memmap(self.input_path, dtype=np.int16, mode='r')

        # Total int16 samples
        total_samples = self.mm.shape[0]
        # Total complex samples (I+Q)
        self.total_complex_samples = total_samples // 2
        # Samples per frame
        self.frame_samples = int(self.duration * self.sample_rate)
        # Total frames
        self.total_frames = self.total_complex_samples // self.frame_samples

        # 2MHz bandwidth lowpass filter for L1 GPS band
        self.sos = signal.butter(10, 1e6, 'lowpass', fs=self.sample_rate, output='sos')

        # Precompute constants
        self.eps = np.finfo(np.float32).eps  # Use float32 epsilon for consistency
        self.log2 = np.log(2)
        self.inv_duration = 1.0 / self.duration
        self.inv_frame_samples = 1.0 / self.frame_samples

        # C/N0 computation constants
        self.M = 20  # Number of blocks for C/N0
        self.K = self.frame_samples // self.M

        # Precompute filter initial conditions for better stability
        self.zi = signal.sosfilt_zi(self.sos)

    def read_frame(self, frame_idx: int) -> np.ndarray:
        """Read a single frame efficiently"""
        start = frame_idx * self.frame_samples * 2
        end = start + self.frame_samples * 2
        raw_data = self.mm[start:end]

        # Direct view casting for better performance
        I = raw_data[0::2].astype(np.float32, copy=False)
        Q = raw_data[1::2].astype(np.float32, copy=False)
        return I + 1j * Q

    def read_frame_batch(self, frame_indices: list) -> list:
        """Read multiple frames at once for better memory access"""
        return [self.read_frame(frame_idx) for frame_idx in frame_indices]

    def filter(self, sig: np.ndarray) -> np.ndarray:
        """Apply 2MHz lowpass filter"""
        return signal.sosfilt(self.sos, sig)

    def compute_cn0_nwpr_optimized(self, I: np.ndarray, Q: np.ndarray) -> float:
        """
        Optimized C/N0 computation using Narrow-band Wide-band Power Ratio
        """
        if self.K == 0:
            return 0.0

        # Reshape into blocks (vectorized)
        trim = self.K * self.M
        I_blocks = I[:trim].reshape(self.K, self.M)
        Q_blocks = Q[:trim].reshape(self.K, self.M)

        # Narrow-band power (coherent integration) - vectorized
        NBPI = I_blocks.sum(axis=1)
        NBPQ = Q_blocks.sum(axis=1)
        NBP = NBPI**2 + NBPQ**2

        # Wide-band power (non-coherent integration) - vectorized
        WBP = np.sum(I_blocks**2 + Q_blocks**2, axis=1)

        # Avoid division by zero
        WBP = np.maximum(WBP, 1e-10)
        NP = NBP / WBP

        mu_NP = np.mean(NP)

        # Avoid invalid log arguments
        if mu_NP <= 1 or self.M <= mu_NP:
            return 0.0

        C_N0 = 10 * np.log10(self.inv_duration * (mu_NP - 1) / (self.M - mu_NP))
        return C_N0

    def extract_features_optimized(self, samples: np.ndarray) -> dict:
        """Optimized feature extraction with vectorized operations"""
        # Apply 2MHz lowpass filter
        filtered_samples = self.filter(samples)
        I = filtered_samples.real
        Q = filtered_samples.imag

        # Precompute common values
        amplitude = np.abs(filtered_samples)
        amplitude_sq = amplitude**2

        # Time domain features (all vectorized)
        mean_amp = np.mean(amplitude)
        mean_amp_sq = np.mean(amplitude_sq)
        
        features = {
            'mean_amplitude': mean_amp,
            'variance_amplitude': np.var(amplitude),
            'std_amplitude': np.std(amplitude),
            'rms_amplitude': np.sqrt(mean_amp_sq)
        }

        # Received Power (dBW)
        features['power'] = 10 * np.log10(mean_amp_sq)

        # C/N0 using optimized NWPR method
        features['C_N0'] = self.compute_cn0_nwpr_optimized(I, Q)

        # Zero crossing rate (optimized with boolean indexing)
        features['zcr_time'] = np.count_nonzero(np.diff(np.signbit(I))) * self.inv_frame_samples

        # Frequency domain features
        fft_signal = fft.fft(samples)
        fft_magnitude = np.abs(fft_signal)

        # Replace NaN with zeros in one operation
        fft_magnitude = np.nan_to_num(fft_magnitude, nan=0.0, copy=False)

        mean_fft = np.mean(fft_magnitude)
        features['mean_fft'] = mean_fft
        features['variance_fft'] = np.var(fft_magnitude)
        features['skewness_fft'] = skew(fft_magnitude)
        features['kurtosis_fft'] = kurtosis(fft_magnitude)

        # Entropy computation (optimized)
        fft_sum = np.sum(fft_magnitude)
        if fft_sum > self.eps:
            prob_fft = fft_magnitude / fft_sum
            # Use np.where to avoid log of zero
            log_prob = np.where(prob_fft > self.eps, np.log2(prob_fft), 0)
            features['entropy_fft'] = -np.sum(prob_fft * log_prob)
        else:
            features['entropy_fft'] = 0.0

        features['peak_to_mean_fft'] = np.max(fft_magnitude) / (mean_fft + self.eps)

        # Phase features (optimized)
        phi = np.angle(samples)
        unwrapped_phi = np.unwrap(phi)
        phase_diff = np.diff(unwrapped_phi)
        features['phase_noise'] = np.std(phase_diff)

        return features

    def process_batch(self, frame_indices: list) -> list:
        """Process a batch of frames"""
        return [
            self.extract_features_optimized(self.read_frame(frame_idx))
            for frame_idx in frame_indices
        ]

    def extract_all(self, batch_size: int = 1000, n_jobs: int = -1, 
                    compression: str = 'snappy', row_group_size: int = 10000):
        """
        Extract features from all frames with parallel processing and optimized Parquet writing
        
        Args:
            batch_size: Number of frames to process in each batch
            n_jobs: Number of parallel jobs (-1 uses all cores)
            compression: Parquet compression algorithm ('snappy', 'gzip', 'brotli', 'lz4', 'zstd', or None)
            row_group_size: Number of rows per row group (larger = better compression, slower random access)
        """
        print("Starting feature extraction...")
        print(f"Total frames to process: {self.total_frames:,}")
        print(f"Compression: {compression}, Row group size: {row_group_size:,}")

        # Prepare batches
        frame_batches = [
            list(range(start, min(start + batch_size, self.total_frames)))
            for start in range(0, self.total_frames, batch_size)
        ]
        total_batches = len(frame_batches)

        writer = None
        output_file = self.output_path
        
        # Accumulate results for larger row groups
        accumulated_results = []
        accumulated_count = 0

        try:
            for batch_idx, batch in enumerate(frame_batches, 1):
                # Read frames first (serial for memory mapping)
                samples_batch = self.read_frame_batch(batch)
                
                # Process in parallel
                batch_results = Parallel(n_jobs=n_jobs, backend='loky')(
                    delayed(self.extract_features_optimized)(samples)
                    for samples in samples_batch
                )

                # Accumulate results
                accumulated_results.extend(batch_results)
                accumulated_count += len(batch_results)

                # Write when we have enough data for a row group or it's the last batch
                if accumulated_count >= row_group_size or batch_idx == total_batches:
                    # Convert to PyArrow Table directly (skip pandas for better performance)
                    table = pa.Table.from_pylist(accumulated_results)

                    if writer is None:
                        # Optimized Parquet writer settings
                        writer = pq.ParquetWriter(
                            output_file, 
                            table.schema,
                            compression=compression,
                            use_dictionary=True,  # Enable dictionary encoding for better compression
                            write_statistics=True,  # Write column statistics for query optimization
                            data_page_size=1024*1024,  # 1MB data pages
                            version='2.6'  # Use latest Parquet version for better features
                        )
                    
                    writer.write_table(table)
                    
                    # Clear accumulated results
                    accumulated_results = []
                    accumulated_count = 0

                # Progress update
                frames_processed = min(batch[-1] + 1, self.total_frames)
                progress = (frames_processed / self.total_frames) * 100
                print(f"Batch {batch_idx}/{total_batches} | "
                      f"Frames: {frames_processed:,}/{self.total_frames:,} | "
                      f"Progress: {progress:.1f}%")

        finally:
            if writer:
                writer.close()

        # Print file statistics
        output_size = os.path.getsize(self.output_path)
        print("\nFeature extraction completed!")
        print(f"Output saved to: {self.output_path}")
        print(f"Output size: {output_size:,} bytes ({output_size / (1024 ** 2):.2f} MB)")
        
        # Calculate compression ratio if applicable
        input_size = os.path.getsize(self.input_path)
        features_per_frame = 14  # Number of features extracted
        uncompressed_estimate = self.total_frames * features_per_frame * 8  # 8 bytes per float64
        compression_ratio = uncompressed_estimate / output_size if output_size > 0 else 0
        print(f"Estimated compression ratio: {compression_ratio:.2f}x")

    def info(self):
        """Display file and processing information"""
        file_size = os.path.getsize(self.input_path)
        total_samples = self.mm.shape[0]
        total_complex_samples = self.total_complex_samples
        total_frames = self.total_frames
        total_duration = total_complex_samples / self.sample_rate

        print("=" * 60)
        print("FILE INFORMATION")
        print("=" * 60)
        print(f"File path:             {self.input_path}")
        print(f"File size:             {file_size:,} bytes ({file_size / (1024 ** 3):.2f} GB)")
        print(f"Sample rate:           {self.sample_rate:,} Hz")
        print(f"Frame duration:        {self.duration:.3f} seconds")
        print()
        print("SIGNAL INFORMATION")
        print("=" * 60)
        print(f"Total samples (int16): {total_samples:,} values")
        print(f"Complex samples (I+Q): {total_complex_samples:,} samples")
        print(f"Total frames:          {total_frames:,} frames")
        print(f"Total duration:        {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        print(f"Samples per frame:     {self.frame_samples:,}")
        print()
        print("C/N0 COMPUTATION")
        print("=" * 60)
        print(f"Number of blocks (M):  {self.M}")
        print(f"Block size (K):        {self.K}")
        print("=" * 60)