# pyrefly: ignore [missing-import]
import tensorflow as tf
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications import EfficientNetB0
# pyrefly: ignore [missing-import]
from tensorflow.keras.models import Sequential
# pyrefly: ignore [missing-import]
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
# pyrefly: ignore [missing-import]
from tensorflow.keras.optimizers import Adam
# pyrefly: ignore [missing-import]
from tensorflow.keras.regularizers import l2

IMG_SIZE = 224

def build_model():
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.0005)),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=l2(0.0005)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model
