import tensorflow as tf
import keras
from keras import ops
from keras import layers
from keras import Model

INPUT_FEATURES_SIZE = 14
CONTEXT_SIZE = 2        # One-hot encoded (Static/Dynamic) 
LATENT_DIM = 2           # Kích thước không gian tiềm ẩn z [cite: 325, 326]
CLIP_VALUE = 0.01        # Kẹp trọng số cho WGAN (không cần thiết với WGAN-GP)
LAMBDA_GP = 10.0         # Hằng số Gradient Penalty [cite: 202]
KAPPA = 5                # Tỷ lệ huấn luyện (Critic/Generator/VAE) [cite: 332]
LEARNING_RATE = 1e-5     # Tốc độ học ban đầu [cite: 331]


@keras.saving.register_keras_serializable(package="VAE_GAN")
class Sampling(keras.layers.Layer):
    """Sử dụng (mu, log_var) để lấy mẫu z theo phân phối Gaussian."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape = (batch, dim),dtype=tf.dtypes.float32)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# --- 1. Mạng Encoder (E_phi) ---
@keras.saving.register_keras_serializable(package="VAE_GAN")
class Encoder(Model):
    def __init__(self, latent_dim=LATENT_DIM, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.latent_dim = latent_dim

        self.concat = layers.Concatenate(axis=-1)
        self.dense1 = layers.Dense(20)
        self.prelu1 = layers.PReLU()
        self.dense2 = layers.Dense(10)
        self.prelu2 = layers.PReLU()
        self.dense3 = layers.Dense(5)
        self.prelu3 = layers.PReLU()
        
        # 4 nodes output: mu_z (2 nodes) và log_var_z (2 nodes) [cite: 325]
        self.mu_layer = layers.Dense(self.latent_dim, name="z_mean")
        self.log_var_layer = layers.Dense(self.latent_dim, name="z_log_var")
        self.sampling = Sampling()
    

    def call(self, x, c):
        x_in = self.concat([x, c])
        h = self.prelu1(self.dense1(x_in))
        h = self.prelu2(self.dense2(h))
        h = self.prelu3(self.dense3(h))
        
        z_mean = self.mu_layer(h)
        z_log_var = self.log_var_layer(h)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z
    
    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config

# --- 2. Mạng Decoder (D_theta) ---
@keras.saving.register_keras_serializable(package="VAE_GAN")
class Decoder(Model):
    def __init__(self, output_dim=INPUT_FEATURES_SIZE, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.output_dim = output_dim

        self.concat = layers.Concatenate(axis=-1)
        # Bắt đầu từ lớp ẩn đầu tiên (5 nodes)
        self.dense1 = layers.Dense(5)
        self.prelu1 = layers.PReLU()
        self.dense2 = layers.Dense(10)
        self.prelu2 = layers.PReLU()
        self.dense3 = layers.Dense(20)
        self.prelu3 = layers.PReLU()
        
        # Output: reconstructed features (7 nodes) [cite: 327]
        self.output_layer = layers.Dense(self.output_dim, name="x_hat")

    def call(self, z, c):
        z_in = self.concat([z, c])
        h = self.prelu1(self.dense1(z_in))
        h = self.prelu2(self.dense2(h))
        h = self.prelu3(self.dense3(h))
        x_hat = self.output_layer(h) # No activation at the final layer [cite: 327]
        return x_hat
    
    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim})
        return config
    
# --- 3. Mạng Critic (C_phi) ---
@keras.saving.register_keras_serializable(package="VAE_GAN")
class Critic(Model):
    def __init__(self, **kwargs):
        super(Critic, self).__init__(**kwargs)

        self.dense1 = layers.Dense(20)
        self.prelu1 = layers.PReLU()
        self.dense2 = layers.Dense(10)
        self.prelu2 = layers.PReLU()
        self.dense3 = layers.Dense(5)
        self.prelu3 = layers.PReLU()
        
        # Output: single score (no activation) [cite: 329]
        self.output_layer = layers.Dense(1, name="critic_score")
        

    def call(self, x_in):
        h = self.prelu1(self.dense1(x_in))
        h = self.prelu2(self.dense2(h))
        h = self.prelu3(self.dense3(h))
        score = self.output_layer(h)
        return score
    
    def get_config(self):
        return super().get_config()
    
# --- 4. Mô Hình Tổ Hợp VAE-WGAN-GP ---
@keras.saving.register_keras_serializable(package="VAE_GAN")
class VAE_WGAN_GP(Model):
    def __init__(self, encoder, decoder, critic, kappa=KAPPA, lambda_gp=LAMBDA_GP, latent_dim=LATENT_DIM, **kwargs):
        super(VAE_WGAN_GP, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.critic = critic
        self.kappa = kappa
        self.lambda_gp = lambda_gp
        self.latent_dim = LATENT_DIM
        self.critic_optimizer = None
        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.epoch_tracker = tf.Variable(0, dtype=tf.int64, trainable=False)
        
        # Loss Metrics (để theo dõi trong quá trình huấn luyện)
        self.critic_loss_metric = keras.metrics.Mean(name="critic_loss")
        self.reconstruction_loss_metric = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_metric = keras.metrics.Mean(name="kl_loss")
        self.decoder_loss_metric = keras.metrics.Mean(name="decoder_loss")
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'encoder': keras.saving.serialize_keras_object(self.encoder),
            'decoder': keras.saving.serialize_keras_object(self.decoder),
            'critic': keras.saving.serialize_keras_object(self.critic),
            'kappa': self.kappa,
            'lambda_gp': self.lambda_gp,
            'latent_dim': self.latent_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        
        encoder_config = config.pop('encoder')
        decoder_config = config.pop('decoder')
        critic_config = config.pop('critic')

        encoder = keras.saving.deserialize_keras_object(encoder_config)
        decoder = keras.saving.deserialize_keras_object(decoder_config)
        critic = keras.saving.deserialize_keras_object(critic_config)
        
        return cls(encoder=encoder, decoder=decoder, critic=critic, **config)  

    # Gán Optimizer cho các mô hình
    def compile(self, critic_optimizer, encoder_optimizer, decoder_optimizer, **kwargs):
        super(VAE_WGAN_GP, self).compile(**kwargs)
        self.critic_optimizer = critic_optimizer
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer

    @property
    def metrics(self):
        return [
            self.critic_loss_metric,
            self.reconstruction_loss_metric,
            self.kl_loss_metric,
            self.decoder_loss_metric,
        ]

    # Tính Gradient Penalty (phần của WGAN-GP)
    def gradient_penalty(self, real_samples, fake_samples):

        real_samples = tf.cast(real_samples, tf.float32)
        fake_samples = tf.cast(fake_samples, tf.float32)
        # Lấy mẫu ngẫu nhiên từ đường nối giữa real và fake
        alpha = tf.random.uniform(shape=[tf.shape(real_samples)[0], 1], minval=0., maxval=1., dtype=tf.float32)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # Tính Critic score trên mẫu ngẫu nhiên
            critic_score = self.critic(interpolated)

        # Tính đạo hàm của Critic score theo mẫu ngẫu nhiên
        grads = gp_tape.gradient(critic_score, interpolated)
        
        # Tính chuẩn l2 của gradient
        norm = tf.norm(grads, axis=1)
        
        # Gradient Penalty [cite: 202]
        gp = tf.reduce_mean((norm - 1.0)**2)
        return gp

    # Tính VAE Loss
    def vae_loss(self, x_real, x_recon, z_mean, z_log_var):
        # 1. Reconstruction Loss: MSE (Phù hợp với đầu ra x là các features thực)
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_real - x_recon), axis=1))
        
        # 2. KL Divergence Loss: DKL(q(z|x) || p(z)) [cite: 224]
        # (p(z) là phân phối chuẩn tắc N(0, I))thê
        
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        
        # VAE loss (sử dụng factor beta=1)
        total_vae_loss = recon_loss + kl_loss
        return total_vae_loss, recon_loss, kl_loss

    # Vòng lặp huấn luyện tùy chỉnh (Algorithm 1)
    def train_step(self, data):
        # Giả sử `data` là tuple (x_real, c_context)
        # x_real: features (7)
        # c_context: context one-hot (2)
        x_real, c_context = data
        batch_size = tf.shape(x_real)[0]
        
        # ----------------------------------------------------
        # 1. Huấn luyện Critic (C_phi) - Chạy KAPPA lần 
        # ----------------------------------------------------
        for i in range(self.kappa):
            # Tạo z ngẫu nhiên (sử dụng p_z(z) ~ N(0,I))
            random_z = tf.random.normal(shape=(batch_size, LATENT_DIM))
            
            with tf.GradientTape() as critic_tape:
                # 1. Tạo mẫu giả (fake sample) [cite: 213]
                x_fake = self.decoder(random_z, c_context)
                
                # 2. Tính Critic score cho Real và Fake
                real_scores = self.critic(x_real)
                fake_scores = self.critic(x_fake)
                
                # 3. Tính Gradient Penalty
                gp = self.gradient_penalty(x_real, x_fake)
                
                # 4. Critic Loss (Wasserstein Distance + GP) [cite: 202]
                # L_critic = E[C(x_real)] - E[C(x_fake)] - lambda * GP [cite: 213]
                critic_loss = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores) + self.lambda_gp * gp
            
            # Cập nhật tham số Critic
            critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        self.critic_loss_metric.update_state(critic_loss)

        # ----------------------------------------------------
        # 2. Huấn luyện Encoder (E_phi) và Decoder (D_theta) - 1 lần
        # ----------------------------------------------------
        # Chỉ huấn luyện VAE/Generator nếu đạt tỷ lệ kappa (mod(t, kappa) == 0) [cite: 216]
        with tf.GradientTape(persistent=True) as gen_tape:
            # --- a) VAE Part ---
            z_mean, z_log_var, z_latent = self.encoder(x_real, c_context)
            x_recon = self.decoder(z_latent, c_context)
            vae_loss, recon_loss, kl_loss = self.vae_loss(x_real, x_recon, z_mean, z_log_var)
            
            # --- b) GAN Part (Decoder as Generator) ---
            # Tạo mẫu giả từ z ngẫu nhiên (epsilon)
            random_z = tf.random.normal(shape=(batch_size, LATENT_DIM))
            x_fake_gan = self.decoder(random_z, c_context)
            fake_scores_gen = self.critic(x_fake_gan)
            
            # Decoder/Generator Loss: max E[C(x_fake)] (hoặc min -E[C(x_fake)]) [cite: 207]
            decoder_loss = -tf.reduce_mean(fake_scores_gen)
            
            # --- c) Total Generator/VAE Loss ---
            # Kết hợp VAE Loss (cho Encoder + Decoder) và GAN Loss (cho Decoder)
            # Bài báo không nêu rõ cách kết hợp, sử dụng tổng đơn giản là một cách tiếp cận hợp lý.
            total_gen_loss = vae_loss + decoder_loss 

        # Cập nhật Encoder (chỉ chịu ảnh hưởng bởi VAE Loss)
        encoder_grads = gen_tape.gradient(total_gen_loss, self.encoder.trainable_variables)
        self.encoder_optimizer.apply_gradients(zip(encoder_grads, self.encoder.trainable_variables))

        # Cập nhật Decoder (chịu ảnh hưởng bởi cả VAE Loss và GAN Loss)
        decoder_grads = gen_tape.gradient(total_gen_loss, self.decoder.trainable_variables)
        self.decoder_optimizer.apply_gradients(zip(decoder_grads, self.decoder.trainable_variables))

        del gen_tape
        
        # Cập nhật metrics
        self.reconstruction_loss_metric.update_state(recon_loss)
        self.kl_loss_metric.update_state(kl_loss)
        self.decoder_loss_metric.update_state(decoder_loss)
        
        return {
            "c_loss": self.critic_loss_metric.result(),
            "recon_loss": self.reconstruction_loss_metric.result(),
            "kl_loss": self.kl_loss_metric.result(),
            "g_loss": self.decoder_loss_metric.result(),
        }
    
    # --- Phương thức test_step (Kiểm tra hợp lệ) ---
    def test_step(self, data):
        # Giả sử `data` là tuple (x_real, c_context)
        x_real, c_context = data
        batch_size = tf.shape(x_real)[0]
        
        # 1. Tính VAE Loss (Reconstruction + KL)
        z_mean, z_log_var, z_latent = self.encoder(x_real, c_context, training=False)
        x_recon = self.decoder(z_latent, c_context, training=False)
        vae_loss_val, recon_loss_val, kl_loss_val = self.vae_loss(x_real, x_recon, z_mean, z_log_var)
        
        # 2. Tính Critic Loss (Chỉ tính score, không tính GP hoặc áp dụng W-loss)
        # Tạo z ngẫu nhiên
        random_z = tf.random.normal(shape=(batch_size, LATENT_DIM))
        x_fake_val = self.decoder(random_z, c_context, training=False)
        
        # Tính Critic score cho Real và Fake
        real_scores = self.critic(x_real, training=False)
        fake_scores = self.critic(x_fake_val, training=False)
        
        # Critic Loss (Wasserstein Distance - không cần GP khi chỉ đánh giá)
        critic_loss_val = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)

        # 3. Tính Generator Loss (từ góc độ GAN)
        # Decoder/Generator Loss: min -E[C(x_fake)]
        decoder_loss_val = -tf.reduce_mean(fake_scores)
        
        # Cập nhật Metrics (thêm "val_" cho metrics khi đánh giá)
        self.critic_loss_metric.update_state(critic_loss_val)
        self.reconstruction_loss_metric.update_state(recon_loss_val)
        self.kl_loss_metric.update_state(kl_loss_val)
        self.decoder_loss_metric.update_state(decoder_loss_val)
        
        return {
            "c_loss": self.critic_loss_metric.result(),
            "recon_loss": self.reconstruction_loss_metric.result(),
            "kl_loss": self.kl_loss_metric.result(),
            "g_loss": self.decoder_loss_metric.result(),
        }
    
    # --- Phương thức call (Inference) ---
    def call(self, inputs, training=False):
        # inputs có thể là: (x_in, c_context) HOẶC c_context HOẶC (z_latent, c_context)
        
        # Tái tạo (Reconstruction) - VAE mode
        if isinstance(inputs, tuple) and len(inputs) == 2:
            x_in, c_context = inputs
            # Bỏ qua z_mean, z_log_var trong chế độ inference
            _, _, z_latent = self.encoder(x_in, c_context, training=training) 
            x_recon = self.decoder(z_latent, c_context, training=training)
            return x_recon # x_hat
        
        # Tạo mẫu (Generation) - GAN mode (Giả sử inputs là c_context)
        elif ops.is_tensor(inputs) and inputs.shape[-1] == CONTEXT_SIZE:
            c_context = inputs
            batch_size = tf.shape(c_context)[0]
            # Lấy mẫu từ phân phối chuẩn tắc p(z) ~ N(0, I)
            random_z = tf.random.normal(shape=(batch_size, LATENT_DIM))
            x_gen = self.decoder(random_z, c_context, training=training)
            return x_gen # x_fake
            
        else:
            raise ValueError("Input không hợp lệ. Cần (x_in, c_context) để tái tạo, hoặc c_context để tạo mẫu.")


# TESTING
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix

class SpoofingEvaluator:
    def __init__(self, model, train_gen_data, train_gen_context):
        """
        Khởi tạo bộ đánh giá.
        :param model: Mô hình VAE_WGAN_GP đã huấn luyện
        :param train_gen_data: Dữ liệu thật (Genuine) dùng để training (để tính ngưỡng)
        :param train_gen_context: Nhãn ngữ cảnh tương ứng (Static/Dynamic)
        """
        self.model = model
        # Tính toán tham số tham chiếu từ dữ liệu huấn luyện (Genuine)
        print("Đang tính toán thống kê tham chiếu từ tập huấn luyện...")
        self.ref_mu_z = self._compute_reference_latent_mean(train_gen_data, train_gen_context)
        
        # Tính ngưỡng (Thresholds) dựa trên FPR mong muốn (ví dụ 5%)
        self.thresholds = self._calibrate_thresholds(train_gen_data, train_gen_context, target_fpr=0.05)

    def _compute_reference_latent_mean(self, x, c):
        """Tính E[mu_zg]: Trung bình của các vector latent mean từ dữ liệu thật [cite: 552]"""
        # Truy cập trực tiếp vào Encoder
        z_mean, _, _ = self.model.encoder(x, c, training=False)
        return np.mean(z_mean.numpy(), axis=0)

    def _get_scores(self, x, c):
        """Tính toán 3 loại điểm số cho một batch dữ liệu"""
        # 1. Đi qua Encoder
        z_mean, _, z_sample = self.model.encoder(x, c, training=False)
        
        # 2. Đi qua Decoder (Tái tạo)
        x_recon = self.model.decoder(z_sample, c, training=False)
        
        # 3. Đi qua Critic (Đánh giá)
        critic_score = self.model.critic(x, training=False)
        
        # Chuyển sang numpy để tính toán
        x_np = x if isinstance(x, np.ndarray) else x.numpy()
        x_recon_np = x_recon.numpy()
        z_mean_np = z_mean.numpy()
        critic_score_np = critic_score.numpy().flatten() # Flatten thành vector 1 chiều
        
        # --- Tính toán các chỉ số thống kê (Test Statistics) ---
        
        # A. D_l2: Khoảng cách Euclidean đến tâm latent của dữ liệu thật 
        # zeta_l2 = || mu_z - E[mu_zg] ||_2
        diff_latent = z_mean_np - self.ref_mu_z
        score_l2 = np.linalg.norm(diff_latent, axis=1)
        
        # B. D_rec: Lỗi tái tạo theo chuẩn L1 (Manhattan) 
        # zeta_rec = || x - x_hat ||_1
        score_rec = np.sum(np.abs(x_np - x_recon_np), axis=1)
        
        # C. D_cr: Điểm Critic (Càng cao càng Thật, càng thấp càng Giả) 
        score_cr = critic_score_np 
        
        return score_l2, score_rec, score_cr

    def _calibrate_thresholds(self, x_gen, c_gen, target_fpr=0.05):
        """Tính ngưỡng dựa trên tập Training Genuine với FPR cố định [cite: 581-584]"""
        l2_scores, rec_scores, cr_scores = self._get_scores(x_gen, c_gen)
        
        # Với L2 và Reconstruction Error: Giá trị LỚN là Giả, NHỎ là Thật
        # Ngưỡng là giá trị tại percentile thứ (1 - FPR)
        thresh_l2 = np.percentile(l2_scores, (1 - target_fpr) * 100)
        thresh_rec = np.percentile(rec_scores, (1 - target_fpr) * 100)
        
        # Với Critic Score: Giá trị LỚN là Thật, NHỎ là Giả (trong WGAN)
        # Ngưỡng là giá trị tại percentile thứ (FPR)
        thresh_cr = np.percentile(cr_scores, target_fpr * 100)
        
        print(f"Ngưỡng đã tính (FPR={target_fpr}): L2={thresh_l2:.4f}, Rec={thresh_rec:.4f}, Critic={thresh_cr:.4f}")
        return {"l2": thresh_l2, "rec": thresh_rec, "cr": thresh_cr}

    def evaluate_test_set(self, x_test, c_test, y_test):
        """
        Đánh giá trên tập test (bao gồm cả Genuine và Spoof)
        y_test: 0 là Genuine, 1 là Spoofed
        """
        l2_scores, rec_scores, cr_scores = self._get_scores(x_test, c_test)
        
        results = {}
        
        # Đánh giá từng bộ phát hiện theo công thức (22) trong bài báo [cite: 584]
        
        # 1. D_l2 evaluation
        # Spoofed nếu score > threshold
        pred_l2 = (l2_scores >= self.thresholds['l2']).astype(int)
        results['D_l2'] = self._calculate_metrics(y_test, pred_l2)
        
        # 2. D_rec evaluation
        # Spoofed nếu score > threshold
        pred_rec = (rec_scores >= self.thresholds['rec']).astype(int)
        results['D_rec'] = self._calculate_metrics(y_test, pred_rec)
        
        # 3. D_cr evaluation
        # Spoofed nếu score <= threshold (Lưu ý dấu <= cho Critic)
        pred_cr = (cr_scores <= self.thresholds['cr']).astype(int)
        results['D_cr'] = self._calculate_metrics(y_test, pred_cr)
        
        return results

    def _calculate_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0 # Detection Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 # False Alarm Rate
        return {"Accuracy": acc, "TPR": tpr, "FPR": fpr}

# --- CÁCH SỬ DỤNG ---
# Giả sử bạn đã có model 'vae_wgan' và dữ liệu:
# train_x, train_c: Dữ liệu huấn luyện (chỉ chứa Genuine)
# test_x, test_c: Dữ liệu kiểm thử (trộn lẫn Genuine và Spoof)
# test_y: Nhãn kiểm thử (0: Genuine, 1: Spoof)

# 1. Khởi tạo Evaluator (Tính toán ngưỡng tự động)
# evaluator = SpoofingEvaluator(vae_wgan, train_x, train_c)

# 2. Chạy đánh giá
# metrics = evaluator.evaluate_test_set(test_x, test_c, test_y)

# 3. In kết quả
# for detector, scores in metrics.items():
#     print(f"--- {detector} ---")
#     print(f"Accuracy: {scores['Accuracy']:.4f}")
#     print(f"TPR (Detection Rate): {scores['TPR']:.4f}")
#     print(f"FPR (False Alarm): {scores['FPR']:.4f}")
