import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np

class DataLoader:
    
    def __init__(self):
        pass

    def save_scaler(self, scaler = 0, path="scaler.pkl"):
        """
        Lưu scaler vào file
        """
        if self.scaler:
            scaler = self.scaler
        joblib.dump(scaler, path)
        print(f"Scaler đã được lưu tại: {path}")

    def load_scaler(self, path="scaler.pkl"):
        """
        Tải scaler từ file
        """
        self.scaler = joblib.load(path)
        print(f"Scaler đã được tải từ: {path}")
        return self.scaler

    def load_data(self, train_path, val_path, test_path):
        train_df = pd.read_parquet(train_path, engine="pyarrow")  # giả định cột label tên 'label'
        assert 'label' in train_df.columns, "CSV phải có cột 'label'"
        val_df = pd.read_parquet(val_path, engine="pyarrow")  # giả định cột label tên 'label'
        assert 'label' in val_df.columns, "CSV phải có cột 'label'"
        test_df = pd.read_parquet(test_path, engine="pyarrow")  # giả định cột label tên 'label'
        assert 'label' in test_df.columns, "CSV phải có cột 'label'"

        # Tách features và labels cho từng tập
        feat_cols = [c for c in train_df.columns if c != 'label']

        X_train = train_df[feat_cols].values
        y_train = train_df['label'].values.astype(int)

        X_val = val_df[feat_cols].values
        y_val = val_df['label'].values.astype(int)

        X_test = test_df[feat_cols].values
        y_test = test_df['label'].values.astype(int)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def preprocess_features(self, X, is_train=False, log_transform=True, robust=True):
        """
        Tiền xử lý đặc trưng: log-transform và scale.

        Args:
            X (ndarray): Dữ liệu đặc trưng.
            is_train (bool): True nếu đang train (fit scaler).
            log_transform (bool): Có log-transform không.
            robust (bool): Dùng RobustScaler thay vì StandardScaler.

        Returns:
            ndarray: Dữ liệu đã chuẩn hóa.
        """
        
        # Log-transform nếu phân phối lệch
        if log_transform:
            X = np.log1p(np.abs(X)) #log(1 + x)

        if is_train:
            # Scaling
            if robust:
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X) # fit rồi transform luôn
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled
