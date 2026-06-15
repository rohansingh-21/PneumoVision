# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications.efficientnet import preprocess_input
# pyrefly: ignore [missing-import]
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = 224

def get_generators(train_dir, test_dir):
    """Load train/val/test generators with augmentation and class weights.

    Args:
        train_dir: Path to training data directory.
        test_dir: Path to test data directory.

    Returns:
        Tuple of (train_generator, val_generator, test_generator, class_weight).
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        fill_mode='nearest',
        validation_split=0.2
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32, class_mode='binary',
        subset='training', seed=42  # reproducible splits  # reproducible splits  # reproducible splits  # reproducible splits
    )
    val_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32, class_mode='binary',
        subset='validation', seed=42  # reproducible splits  # reproducible splits  # reproducible splits  # reproducible splits
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32, class_mode='binary',
        shuffle=False
    )

    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=train_generator.classes
    )
    class_weight = dict(enumerate(class_weights_array))

    return train_generator, val_generator, test_generator, class_weight
