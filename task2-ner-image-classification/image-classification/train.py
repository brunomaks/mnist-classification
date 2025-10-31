import argparse
import tensorflow as tf
from tensorflow import keras
from keras import layers, applications, callbacks
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageClassifierTrainer:
    def __init__(self, args):
        self.args = args
        self.setup_directories()
        self.model = None
        self.history = None
        
    def setup_directories(self):
        self.checkpoint_dir = Path(self.args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir = Path(self.args.tensorboard_dir)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
    def apply_augmentation(self, image):
        # Apply augmentation operations manually
        # Random flip
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
        
        # Random rotation 0, 90, 180, 270)
        random_rotation = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=random_rotation)
        
        # Random zoom
        zoom_factor = tf.random.uniform([], 1.0 - self.args.zoom_range, 1.0 + self.args.zoom_range)
        new_size = tf.cast(tf.cast(tf.shape(image)[:2], tf.float32) * zoom_factor, tf.int32)
        image = tf.image.resize(image, new_size)
        image = tf.image.resize_with_crop_or_pad(image, self.args.image_size, self.args.image_size)
        
        # Random contrast
        image = tf.image.random_contrast(image, 1.0 - self.args.contrast_range, 1.0 + self.args.contrast_range)
        
        return image
    
    # Workaround to first augment and then preprocess
    # As a way of not doing it in the model layer
    def augment_and_preprocess(self, image, preprocess_fn, isTraining):
        if not tf.is_tensor(image):
            image = tf.convert_to_tensor(image)

        # Check for empty images and skip augmentation if empty (return empty image)
        if tf.reduce_all(tf.shape(image) == 0) or tf.reduce_min(tf.shape(image)) == 0:
            return tf.zeros((self.args.image_size, self.args.image_size, 3), dtype=tf.float32)

        if isTraining and self.args.use_augmentation:
            image = self.apply_augmentation(image)

        return preprocess_fn(image)

        
    def create_data_generators(self):
        # Create image data generators
        if self.args.base_model == 'EfficientNetB0':
            preprocess_fn = applications.efficientnet.preprocess_input
        elif self.args.base_model == 'ResNet50':
            preprocess_fn = applications.resnet50.preprocess_input
        elif self.args.base_model == 'MobileNetV2':
            preprocess_fn = applications.mobilenet_v2.preprocess_input
        else:
            raise ValueError(f"Unsupported base model: {self.args.base_model}")

        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            validation_split=self.args.validation_split,
            preprocessing_function= lambda x: self.augment_and_preprocess(x, preprocess_fn, True)
        )

        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            validation_split=self.args.validation_split,
            preprocessing_function= lambda x: self.augment_and_preprocess(x, preprocess_fn, False)
        )
        
        # Training generator with augmentation
        train_generator = train_datagen.flow_from_directory(
            self.args.data_dir,
            target_size=(self.args.image_size, self.args.image_size),
            batch_size=self.args.batch_size,
            subset='training',
            class_mode='categorical',
            shuffle=True,
            seed=self.args.seed
        )
        
        # Validation generator without augmentation
        validation_generator = val_datagen.flow_from_directory(
            self.args.data_dir,
            target_size=(self.args.image_size, self.args.image_size),
            batch_size=self.args.batch_size,
            subset='validation',
            class_mode='categorical',
            shuffle=False,
            seed=self.args.seed
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        logger.info(f"Found {train_generator.samples} training images")
        logger.info(f"Found {validation_generator.samples} validation images")
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Class names: {self.class_names}")
        
        return train_generator, validation_generator
    
    def compute_class_weights(self, train_generator):
        class_names = list(train_generator.class_indices.keys())
        train_labels = train_generator.classes  # numeric labels

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(len(class_names)),
            y=train_labels
        )
        class_weights = dict(enumerate(class_weights))

        class_mapping = {i: name for i, name in enumerate(class_names)}
        print("Class weights with actual names:")
        for class_idx, weight in class_weights.items():
            print(f"  {class_idx} ({class_mapping[class_idx]}): {weight:.4f}")

        return class_weights
    
    def create_model(self):
        # Base model (feature extractor)
        if self.args.base_model == 'EfficientNetB0':
            base_model = applications.EfficientNetB0(
                include_top=False,
                weights='imagenet' if self.args.use_pretrained else None,
                input_shape=(self.args.image_size, self.args.image_size, 3)
            )
        elif self.args.base_model == 'ResNet50':
            base_model = applications.ResNet50(
                include_top=False,
                weights='imagenet' if self.args.use_pretrained else None,
                input_shape=(self.args.image_size, self.args.image_size, 3)
            )
        elif self.args.base_model == 'MobileNetV2':
            base_model = applications.MobileNetV2(
                include_top=False,
                weights='imagenet' if self.args.use_pretrained else None,
                input_shape=(self.args.image_size, self.args.image_size, 3)
            )
        else:
            raise ValueError(f"Unsupported base model: {self.args.base_model}")
        
        # Freeze base model if specified
        if self.args.freeze_base:
            base_model.trainable = False
            logger.info("Base model frozen")
        else:
            base_model.trainable = True
            logger.info("Base model trainable")

        # Build sequential
        model = keras.Sequential()

        # Base model
        model.add(base_model)
        
        # Global average pooling
        model.add(layers.GlobalAveragePooling2D())
        
        # Additional dense layers
        for units in self.args.dense_layers:
            model.add(layers.Dense(units, activation='relu'))
        
        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Compiling the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model created and compiled successfully")
        return model
    
    def setup_callbacks(self):
        callbacks_list = []
        
        # Model checkpoint
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=str(self.checkpoint_dir / 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        callbacks_list.append(checkpoint_callback)
        
        # Early stopping
        if self.args.early_stopping_patience > 0:
            early_stopping = callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.args.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks_list.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.args.lr_patience,
            min_lr=1e-7,
            verbose=1
        )
        callbacks_list.append(reduce_lr)
        
        # TensorBoard
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=str(self.tensorboard_dir),
            histogram_freq=1,
            write_graph=True
        )
        callbacks_list.append(tensorboard_callback)
        
        return callbacks_list
    
    # Main training loop
    def train(self):
        logger.info("Starting training process...")

        # Create data generators
        train_generator, validation_generator = self.create_data_generators()

        # Compute class weights
        class_weights = self.compute_class_weights(train_generator)
        
        # Create model
        self.model = self.create_model()

        # Display model summary
        self.model.summary()
        
        # Setup callbacks
        training_callbacks = self.setup_callbacks()
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            epochs=self.args.epochs,
            class_weight=class_weights,
            validation_data=validation_generator,
            callbacks=training_callbacks,
            verbose=1
        )
        
        # Save final model
        self.model.save(str(self.checkpoint_dir / 'final_model.keras'))
        logger.info("Final model saved")
        
        # Save class names
        with open(self.checkpoint_dir / 'class_names.json', 'w') as f:
            json.dump(self.class_names, f)
        
        # Save training history
        self.save_training_history()
        
        return self.history
    
    def save_training_history(self):
        # Save history as JSON
        history_dict = {key: [float(val) for val in values] 
                       for key, values in self.history.history.items()}
        
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        # Create and save plots
        plt.figure(figsize=(10, 4))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train Image Classification Model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Model parameters
    parser.add_argument('--base_model', type=str, default='MobileNetV2',
                       choices=['EfficientNetB0', 'ResNet50', 'MobileNetV2'],
                       help='Base model architecture')
    parser.add_argument('--use_pretrained', action='store_true', default=True,
                       help='Use pretrained ImageNet weights')
    parser.add_argument('--freeze_base', action='store_true', default=True,
                       help='Freeze base model weights')
    parser.add_argument('--dense_layers', type=int, nargs='+', default=[256],
                       help='Dense layer sizes')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    
    # Data augmentation parameters
    parser.add_argument('--use_augmentation', action='store_true', default=True,
                       help='Use data augmentation')
    parser.add_argument('--rotation_range', type=float, default=0.2,
                       help='Rotation range for augmentation')
    parser.add_argument('--zoom_range', type=float, default=0.2,
                       help='Zoom range for augmentation')
    parser.add_argument('--contrast_range', type=float, default=0.2,
                       help='Contrast range for augmentation')
    parser.add_argument('--height_shift_range', type=float, default=0.1,
                       help='Height shift range for augmentation')
    parser.add_argument('--width_shift_range', type=float, default=0.1,
                       help='Width shift range for augmentation')
    
    # Callback parameters
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--lr_patience', type=int, default=5,
                       help='Learning rate reduction patience')
    
    # System parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--tensorboard_dir', type=str, default='logs',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create trainer and start training
    trainer = ImageClassifierTrainer(args)
    history = trainer.train()
    
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    main()