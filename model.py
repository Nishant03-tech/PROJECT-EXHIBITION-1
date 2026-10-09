"""Advanced Model Architecture with Superior Feature Extraction & Classification.

Fully compatible with Keras 3 and TensorFlow 2.15+.

Incorporates:
1. DenseNet121 Feature Extractor (dense gradient flow & feature reuse)
2. CBAM Attention Layer (Channel Attention + Spatial Attention):
   - Channel Attention: Prioritizes disease-specific channels (opacities/cavities)
   - Spatial Attention: Prioritizes thoracic lung regions over borders & collarbones
3. Hybrid Dual-Pooling (GAP + GMP):
   - Global Average Pooling: Captures diffuse, widespread infiltrates
   - Global Max Pooling: Captures localized, focal cavitary nodules
4. Binary Focal Loss:
   - Suppresses gradients from easy negatives (clear lung tissue)
   - Focuses training heavily on hard, subtle lesions
5. Regularized Classification Head (BatchNorm + Progressive Dropout)
"""

from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)


# ==============================================================================
# 1. ATTENTION MECHANISM: CBAM (Channel + Spatial Attention)
# ==============================================================================

@tf.keras.utils.register_keras_serializable(package="Custom")
class CBAMBlock(layers.Layer):
    """Convolutional Block Attention Module (CBAM) for Keras 3 / TensorFlow.

    Sequentially applies:
    1. Channel Attention: 'What' clinical features are meaningful.
    2. Spatial Attention: 'Where' in the thorax (lung fields) to focus.
    """

    def __init__(self, ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.channel_dense1 = layers.Dense(
            max(1, channels // self.ratio),
            activation="relu",
            name="cbam_ch_dense1",
        )
        self.channel_dense2 = layers.Dense(channels, name="cbam_ch_dense2")
        self.spatial_conv = layers.Conv2D(
            filters=1,
            kernel_size=7,
            padding="same",
            activation="sigmoid",
            name="cbam_spatial_conv",
        )
        super().build(input_shape)

    def call(self, inputs):
        # --- 1. CHANNEL ATTENTION ---
        # Squeeze via average and max pooling across spatial dimensions
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)

        avg_out = self.channel_dense2(self.channel_dense1(avg_pool))
        max_out = self.channel_dense2(self.channel_dense1(max_pool))

        channel_attention = tf.nn.sigmoid(avg_out + max_out)
        refined_channel = inputs * channel_attention

        # --- 2. SPATIAL ATTENTION ---
        # Inter-channel pooling
        avg_spatial = tf.reduce_mean(refined_channel, axis=-1, keepdims=True)
        max_spatial = tf.reduce_max(refined_channel, axis=-1, keepdims=True)
        spatial_concat = tf.concat([avg_spatial, max_spatial], axis=-1)

        spatial_attention = self.spatial_conv(spatial_concat)
        return refined_channel * spatial_attention

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


# ==============================================================================
# 2. SUPERIOR LOSS FUNCTION: BINARY FOCAL LOSS
# ==============================================================================

@tf.keras.utils.register_keras_serializable(package="Custom")
class BinaryFocalLoss(tf.keras.losses.Loss):
    """Focal Loss for Binary Classification.

    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Down-weights easy, well-classified examples (e.g. obvious normal tissue),
    forcing the network to focus on hard, borderline tuberculosis cases.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Compute p_t
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_factor = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)

        focal_loss = -alpha_factor * modulating_factor * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


# ==============================================================================
# 3. ENHANCED CLASSIFIER ARCHITECTURE
# ==============================================================================

def build_enhanced_tb_model(
    input_shape: tuple[int, int, int] = (*IMG_SIZE, 3),
    dropout_rate: float = 0.35,
    use_cbam: bool = True,
    use_focal_loss: bool = True,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Builds DenseNet121 with CBAM attention, hybrid pooling, and focal loss."""
    inputs = layers.Input(shape=input_shape, name="input_xray")

    # 1. DenseNet121 Feature Extractor Backbone
    backbone = tf.keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    backbone.trainable = False

    raw_features = backbone(inputs)  # Shape: (None, 7, 7, 1024)

    # 2. CBAM Dual Attention Layer
    if use_cbam:
        attended_features = CBAMBlock(ratio=16, name="cbam_attention")(raw_features)
    else:
        attended_features = raw_features

    # 3. Hybrid Spatial Pooling (GAP for diffuse, GMP for focal lesions)
    gap = layers.GlobalAveragePooling2D(name="global_avg_pool")(attended_features)
    gmp = layers.GlobalMaxPooling2D(name="global_max_pool")(attended_features)
    combined = layers.Concatenate(name="hybrid_pooling")([gap, gmp])  # Shape: (None, 2048)

    # 4. Deep Regularized Classification Head
    x = layers.Dense(256, activation="relu", name="head_dense_1")(combined)
    x = layers.BatchNormalization(name="head_bn_1")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout_1")(x)

    x = layers.Dense(128, activation="relu", name="head_dense_2")(x)
    x = layers.BatchNormalization(name="head_bn_2")(x)
    x = layers.Dropout(dropout_rate * 0.7, name="head_dropout_2")(x)

    # Output Sigmoid Probability
    outputs = layers.Dense(1, activation="sigmoid", name="tb_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="TB_DenseNet121_CBAM")

    loss_fn = BinaryFocalLoss(gamma=2.0, alpha=0.25) if use_focal_loss else "binary_crossentropy"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model, backbone
