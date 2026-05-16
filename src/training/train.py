# pyrefly: ignore [missing-import]
import tensorflow as tf
# pyrefly: ignore [missing-import]
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# pyrefly: ignore [missing-import]
from tensorflow.keras.optimizers import Adam

def train(model, train_generator, val_generator, class_weight):
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr]
    )
    return history, early_stop, reduce_lr


def fine_tune(model, train_generator, val_generator, class_weight, early_stop, reduce_lr):
    base_model = model.layers[0]
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr]
    )
    return history_fine
