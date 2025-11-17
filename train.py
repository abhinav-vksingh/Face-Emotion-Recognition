
import os
import argparse
import json
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data", help="Folder with train/val/test subfolders")
    p.add_argument("--model_dir", type=str, default="./models", help="Where to save models/artifacts")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--fine_tune_epochs", type=int, default=10)
    p.add_argument("--unfreeze_last_n", type=int, default=50, help="Number of base layers to unfreeze for fine-tune")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--fine_tune_lr", type=float, default=1e-5)
    p.add_argument("--use_augmentation", action="store_true", help="Enable extra augmentations")
    p.add_argument("--resume", action="store_true", help="If set, will try to resume from best checkpoint if present")
    return p.parse_args()

def build_generators(data_dir, img_size=(224,224), batch_size=32, use_augmentation=False):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    if use_augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=18,
            width_shift_range=0.12,
            height_shift_range=0.12,
            shear_range=0.12,
            zoom_range=0.12,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode="categorical", shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size,
        class_mode="categorical", shuffle=False
    )
    test_gen = val_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=1,
        class_mode="categorical", shuffle=False
    )

    return train_gen, val_gen, test_gen

def build_model(img_size=(224,224), num_classes=7, base_trainable=False):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    base.trainable = base_trainable

    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def compute_class_weights(generator):
    classes = generator.classes
    classes_unique = np.unique(classes)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes_unique, y=classes)
    return {int(i): float(w) for i,w in zip(classes_unique, class_weights)}

def plot_and_save_history(history, outdir):
    p = os.path.join(outdir, "training_history.png")
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.legend(); plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.legend(); plt.title("Accuracy")
    plt.tight_layout(); plt.savefig(p); plt.close()
    print("Saved training plot to:", p)

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    IMG_SIZE = (args.img_size, args.img_size)
    print("Using data dir:", args.data_dir)

    train_gen, val_gen, test_gen = build_generators(args.data_dir, img_size=IMG_SIZE, batch_size=args.batch_size, use_augmentation=args.use_augmentation)
    num_classes = len(train_gen.class_indices)
    print("Detected classes:", train_gen.class_indices)

    # Save class indices (class_name -> index) for inference mapping
    class_indices_path = os.path.join(args.model_dir, "class_indices.json")
    with open(class_indices_path, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print("Saved class indices to:", class_indices_path)

    class_weights = compute_class_weights(train_gen)
    print("Computed class weights:", class_weights)

    # Build or load model
    model = build_model(img_size=IMG_SIZE, num_classes=num_classes, base_trainable=False)
    model.summary()

    # Callbacks
    best_path = os.path.join(args.model_dir, "fer_mobilenetv2_best.h5")
    checkpoint = ModelCheckpoint(best_path, monitor='val_accuracy', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # Optionally resume: load best weights if present
    if args.resume and os.path.exists(best_path):
        print("Resuming from:", best_path)
        model.load_weights(best_path)

    # Train
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=[checkpoint, reduce_lr, early]
    )
    plot_and_save_history(history, args.model_dir)

    # Fine-tune: unfreeze last N layers and run a shorter training
    try:
        base = None
        # find MobileNet base layer inside model
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2"):
                base = layer
                break
        if base is None:
            # fallback: usually the base is second layer in our build_model
            base = model.layers[1]
    except Exception:
        base = model.layers[1]

    print("Starting fine-tuning: unfreezing last", args.unfreeze_last_n, "layers of the base model.")
    try:
        base.trainable = True
        # freeze early layers, unfreeze last N
        total_layers = len(base.layers)
        n = args.unfreeze_last_n
        for layer in base.layers[:-n]:
            layer.trainable = False
        # recompile with smaller lr
        model.compile(optimizer=optimizers.Adam(learning_rate=args.fine_tune_lr), loss='categorical_crossentropy', metrics=['accuracy'])
        ft_callbacks = [
            ModelCheckpoint(os.path.join(args.model_dir, "fer_mobilenetv2_ft_best.h5"), monitor='val_accuracy', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8, verbose=1),
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
        ]
        history_ft = model.fit(train_gen, epochs=args.fine_tune_epochs, validation_data=val_gen, class_weight=class_weights, callbacks=ft_callbacks)
        plot_and_save_history(history_ft, args.model_dir)
    except Exception as e:
        print("Fine-tune skipped due to error:", e)

    # Save final model
    final_path = os.path.join(args.model_dir, "fer_mobilenetv2_final.h5")
    model.save(final_path)
    print("Saved final model to:", final_path)

    # Evaluate on test set using best fine-tuned model (if present)
    best_ft = os.path.join(args.model_dir, "fer_mobilenetv2_ft_best.h5")
    best_to_load = best_ft if os.path.exists(best_ft) else best_path
    print("Loading best model from:", best_to_load)
    model = tf.keras.models.load_model(best_to_load)

    steps = test_gen.samples
    preds = model.predict(test_gen, steps=steps, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    idx_to_class = {v:k for k,v in train_gen.class_indices.items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    # Save confusion matrix plot
    try:
        import seaborn as sns
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
        plt.title("Confusion matrix")
        plt.savefig(os.path.join(args.model_dir, "confusion_matrix.png"))
        plt.close()
        print("Saved confusion matrix to", os.path.join(args.model_dir, "confusion_matrix.png"))
    except Exception:
        # fallback simple plot
        plt.figure(figsize=(8,6))
        plt.imshow(cm, interpolation='nearest')
        plt.title("Confusion matrix")
        plt.colorbar()
        plt.xticks(range(len(target_names)), target_names, rotation=45)
        plt.yticks(range(len(target_names)), target_names)
        plt.tight_layout()
        plt.savefig(os.path.join(args.model_dir, "confusion_matrix.png"))
        plt.close()
        print("Saved confusion matrix to", os.path.join(args.model_dir, "confusion_matrix.png"))

    print("All done. Check", args.model_dir, "for artifacts.")

if __name__ == "__main__":
    main()