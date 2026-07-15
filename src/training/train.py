# pyrefly: ignore [missing-import]
import tensorflow as tf
# pyrefly: ignore [missing-import]
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# pyrefly: ignore [missing-import]
from tensorflow.keras.optimizers import Adam

def train(model, train_generator, val_generator, class_weight):
    early_stop = EarlyStopping(
        monitor='val_auc',
        patience=10  # generous patience for stable convergence  # generous patience for stable convergence  # generous patience for stable convergence  # generous patience for stable convergence  # generous patience for stable convergence  # generous patience for stable convergence  # generous patience for stable convergence  # generous patience for stable convergence  # generous patience for stable convergence,
        restore_best_weights=True,
        mode='max'
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        mode='max'
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=25,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr]
    )
    return history


# Strategy: unfreeze last 50 layers with very low LR to adapt features to X-ray domain
def fine_tune(model, train_generator, val_generator, class_weight):
    base_model = model.layers[0]
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=5e-6),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    early_stop_fine = EarlyStopping(
        monitor='val_auc',
        patience=8,
        restore_best_weights=True,
        mode='max'
    )
    reduce_lr_fine = ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        mode='max'
    )

    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15,
        class_weight=class_weight,
        callbacks=[early_stop_fine, reduce_lr_fine]
    )
    return history_fine
