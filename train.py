"""Train the custom CNN (from scratch - no pretrained weights).

Usage:  python train.py [--epochs 150] [--batch-size 32] [--resume]
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from utils.helpers import (IMG_SHAPE, IMG_SIZE, MODEL_PATH, OUTPUTS_DIR, TRAIN_DIR, VAL_DIR,
                           CLASS_NAMES_PATH, discover_class_names, ensure_dirs, read_json, write_json)

SEED = 42


def load_split(directory, class_names):
    """Unbatched dataset of (image 0-255 float32 128x128x3, one-hot label)."""
    return keras.utils.image_dataset_from_directory(
        directory, labels="inferred", label_mode="categorical", class_names=class_names,
        color_mode="rgb", image_size=(IMG_SIZE, IMG_SIZE), batch_size=None, shuffle=False)


def build_augmentation():
    """Moderate augmentation, applied to TRAINING images only."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05, fill_mode="reflect"),            # about +/-18 degrees
        layers.RandomZoom(0.15, fill_mode="reflect"),
        layers.RandomTranslation(0.10, 0.10, fill_mode="reflect"),   # shifting
        layers.RandomContrast(0.20),
        layers.RandomBrightness(0.15, value_range=(0, 255)),
    ], name="augmentation")


def mixup(x, y, alpha):
    """Mixup: blend random pairs of training images and their labels (a strong regulariser
    for small datasets). Applied to training batches only."""
    b = tf.shape(x)[0]
    g1 = tf.random.gamma([b], alpha)
    g2 = tf.random.gamma([b], alpha)
    lam = g1 / (g1 + g2 + 1e-8)
    lx, ly = tf.reshape(lam, [-1, 1, 1, 1]), tf.reshape(lam, [-1, 1])
    idx = tf.random.shuffle(tf.range(b))
    return x * lx + tf.gather(x, idx) * (1.0 - lx), y * ly + tf.gather(y, idx) * (1.0 - ly)


def conv_bn_relu(x, filters, kernel=3, strides=1):
    x = layers.Conv2D(filters, kernel, strides=strides, padding="same", use_bias=False,
                      kernel_initializer="he_normal")(x)
    # momentum 0.9 (not 0.99): with few steps per epoch the moving statistics must
    # adapt quickly or validation results are unstable.
    x = layers.BatchNormalization(momentum=0.9)(x)
    return layers.Activation("relu")(x)


def block(x, filters, drop):
    x = conv_bn_relu(x, filters)
    x = conv_bn_relu(x, filters)
    x = layers.MaxPooling2D(2)(x)
    return layers.SpatialDropout2D(drop)(x)


def build_model(num_classes):
    inputs = keras.Input(shape=IMG_SHAPE, name="image")
    x = layers.Rescaling(1.0 / 255, name="normalize")(inputs)
    x = conv_bn_relu(x, 32, kernel=5, strides=2)       # stem: 128 -> 64
    x = block(x, 48, 0.05)                              # 64 -> 32
    x = block(x, 96, 0.10)                              # 32 -> 16
    x = block(x, 160, 0.10)                             # 16 -> 8
    x = block(x, 256, 0.15)                             # 8 -> 4
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="vehicle_cnn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--mixup", type=float, default=0.3, help="mixup alpha (0 disables)")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true", help="continue from the saved best model")
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    ensure_dirs()
    class_names = discover_class_names(TRAIN_DIR)
    assert class_names == discover_class_names(VAL_DIR), "train/validation class folders differ"
    write_json(CLASS_NAMES_PATH, {"class_names": class_names})
    print("Classes:", class_names)

    train_raw = load_split(TRAIN_DIR, class_names).cache()
    val_ds = load_split(VAL_DIR, class_names).cache().batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    n_train = int(train_raw.cardinality())
    augment = build_augmentation()

    # Class-balanced sampling: every batch contains all classes equally often, so the
    # smaller class (bicycle) is not drowned out. Only the TRAINING set is resampled.
    def is_class(k):
        return lambda x, y: tf.equal(tf.argmax(y, axis=-1), k)
    per_class = [train_raw.filter(is_class(k)).repeat().shuffle(256, seed=SEED)
                 for k in range(len(class_names))]
    balanced = tf.data.Dataset.sample_from_datasets(
        per_class, weights=[1.0 / len(class_names)] * len(class_names), seed=SEED)
    train_ds = (balanced.batch(args.batch_size)
                .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE))
    if args.mixup > 0:
        train_ds = train_ds.map(lambda x, y: mixup(x, y, args.mixup), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    steps_per_epoch = int(np.ceil(n_train / args.batch_size))

    counts = [len(list((TRAIN_DIR / c).glob("*"))) for c in class_names]
    y = np.concatenate([[i] * n for i, n in enumerate(counts)])
    cw = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=y)
    class_weight = {i: float(w) for i, w in enumerate(cw)}
    print("Training images per class:", dict(zip(class_names, counts)))

    history_path = OUTPUTS_DIR / "history.csv"
    best_val = None
    initial_epoch = 0
    if args.resume and MODEL_PATH.exists() and history_path.exists():
        import pandas as pd
        model = keras.models.load_model(MODEL_PATH)
        prev = pd.read_csv(history_path)
        initial_epoch, best_val = len(prev), float(prev["val_loss"].min())
        print(f"Resuming from epoch {initial_epoch} (best val_loss {best_val:.4f})")
    else:
        model = build_model(len(class_names))
    model.summary()
    model.compile(optimizer=keras.optimizers.AdamW(args.lr, weight_decay=1e-4),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
                  metrics=["accuracy"])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True,
                                        initial_value_threshold=best_val, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6,
                                          min_lr=1e-5, verbose=1),
        keras.callbacks.CSVLogger(str(history_path), append=bool(initial_epoch)),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
              callbacks=callbacks, verbose=2)
    print(f"Best model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
