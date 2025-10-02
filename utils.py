import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd

def save_scaler(scaler, path="scaler.pkl"):
    """
    Lưu scaler vào file
    """
    joblib.dump(scaler, path)
    print(f"Scaler đã được lưu tại: {path}")

def load_scaler(path="scaler.pkl"):
    """
    Tải scaler từ file
    """
    scaler = joblib.load(path)
    print(f"Scaler đã được tải từ: {path}")
    return scaler

def load_data(train_path, val_path, test_path):

    train_df = pd.read_csv(train_path)  # giả định cột label tên 'label'
    assert 'label' in train_df.columns, "CSV phải có cột 'label'"
    val_df = pd.read_csv(val_path)  # giả định cột label tên 'label'
    assert 'label' in val_df.columns, "CSV phải có cột 'label'"
    test_df = pd.read_csv(test_path)  # giả định cột label tên 'label'
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
