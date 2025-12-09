import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense, PReLU, Input, Activation, BatchNormalization
from tensorflow.keras import regularizers
import numpy as np
import os

class CNN_LSTM:
    def __init__(self, batch_size: int, n_steps: int, epochs: int, lr = 1e-3):
        """
        Khởi tạo mô hình CNN-LSTM.

        Args:
            batch_size (int): Kích thước batch huấn luyện.
            n_steps (int): Số bước trong mỗi sequence.
            epochs (int): Số epoch huấn luyện.
            lr (float): Learning rate.
        """
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.epochs = epochs
        self.lr = lr
        
    def create_cnn_lstm_model(self, input_shape):
        """
        Xây dựng kiến trúc CNN-LSTM.

        Args:
            input_shape (tuple): Dạng đầu vào (n_steps, features).

        Returns:
            tf.keras.Model: Mô hình CNN-LSTM.
        """
        model = Sequential()

        model.add(Input(shape=input_shape))

        model.add(Conv1D(filters=64, kernel_size=5, padding="same", kernel_regularizer=regularizers.l2(0.001)))
        # model.add(BatchNormalization())
        model.add(PReLU())
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(filters=64, kernel_size=3, padding="same", kernel_regularizer=regularizers.l2(0.001)))
        # model.add(BatchNormalization())
        model.add(PReLU())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        
        # Stacked LSTM layers
        model.add(LSTM(units=128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.3))
        model.add(LSTM(units=64, kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.3))

        model.add(Dense(units=32, kernel_regularizer=regularizers.l2(0.001)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        
        # Output layer (choose one approach)
        model.add(Dense(units=1, activation='sigmoid'))  # Binary
        
        return model

    def make_sequences(self, df_features: np.ndarray, labels: np.ndarray | None, seq_len: int, label_mode: str = "last"):
        """
        Tạo sequence (cửa sổ trượt) từ dữ liệu.

        Args:
            df_features (ndarray): Ma trận đặc trưng (N, F).
            labels (ndarray | None): Nhãn cho từng hàng, None nếu không có.
            seq_len (int): Độ dài mỗi sequence.
            label_mode (str): Cách chọn nhãn: 'first', 'last', 'middle', 'vote'.

        Returns:
            X (ndarray): Tập sequence (N_seq, seq_len, F).
            y (ndarray | None): Nhãn (N_seq,) nếu có.
        """
        N, F = df_features.shape
        X, y = [], []

        for i in range(N - seq_len + 1):
            seq = df_features[i:i + seq_len]

            if labels is not None:
                if label_mode == "first":
                    label = labels[i]
                elif label_mode == "last":
                    label = labels[i + seq_len - 1]
                elif label_mode == "middle":
                    label = labels[i + seq_len // 2]
                elif label_mode == "vote":
                    # Lấy nhãn xuất hiện nhiều nhất trong sequence
                    values, counts = np.unique(labels[i:i + seq_len], return_counts=True)
                    label = values[np.argmax(counts)]
                else:
                    raise ValueError("label_mode phải là: 'first', 'last', 'middle', hoặc 'vote'")
                y.append(label)

            X.append(seq)

        X = np.array(X, dtype=np.float32)
        if labels is None:
            return X, None
        else:
            return X, np.array(y, dtype=np.float32)

    
    
    def fit(self, X_train, y_train, X_val, y_val, model_path = ""):
        """
        Huấn luyện mô hình.

        Args:
            X_train, y_train: Tập train.
            X_val, y_val: Tập validation.
            model_path (str): Đường dẫn lưu mô hình.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.features = self.X_train.shape[1]
        #Tạo model
        input_shape = (self.n_steps, self.features)
        self.model = self.create_cnn_lstm_model(input_shape)
        self.model.summary()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss='binary_crossentropy',  # Match with sigmoid output
            metrics=['accuracy', 'precision', 'recall']
        )
        # Tạo đường dẫn để lưu model (.keras)
        if model_path == "":
            cwd = os.getcwd()
            model_file = os.path.join(cwd + '/model'+ ".keras")
        else:
            model_file = os.path.join(model_path + ".keras")
        if os.path.exists(model_file):
            print(f"! File '{model_file}' đã tồn tại.")

        # Callbacks
        ckpt = tf.keras.callbacks.ModelCheckpoint(model_file,
                                        monitor='val_loss', save_best_only=True, verbose=1)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(model_path, "logs"))

        X_train, y_train = self.make_sequences(df_features=X_train, labels=y_train, seq_len=self.n_steps, label_mode="vote")
        X_val, y_val = self.make_sequences(df_features=X_val, labels=y_val, seq_len=self.n_steps, label_mode="vote")

        # Training with class weights
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[ckpt, es, tensorboard_cb],
            verbose=1
        )
        return history

    def evaluate(self, X_test, y_test):
        """
        Đánh giá mô hình trên tập test.

        Args:
            X_test, y_test: Tập kiểm tra.

        Returns:
            list: Loss và các metrics.
        """

        X_test, y_test = self.make_sequences(df_features=X_test, labels=y_test, seq_len=self.n_steps, label_mode="vote")
        return self.model.evaluate(X_test, y_test, verbose=1)
    
    def predict(self, X):
        """
        Dự đoán nhãn cho dữ liệu mới.

        Args:
            X (ndarray): Dữ liệu đặc trưng.

        Returns:
            ndarray: Xác suất dự đoán.
        """
        X = self.make_sequences(df_features=X, seq_len=self.n_steps, label_mode="vote")
        y_pred = self.model.predict(X)
        return y_pred
    
    def load_model(self, model_path: str):
        """
        Tải mô hình đã huấn luyện.

        Args:
            model_path (str): Đường dẫn file mô hình (.keras).
        """
        self.model = tf.keras.models.load_model(model_path)
        self.model.summary()
        return self.model
