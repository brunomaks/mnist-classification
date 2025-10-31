import argparse
import os
import tensorflow as tf
from tensorflow import keras
from keras import applications 
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageClassifierInference:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.class_names = None
        self.load_model()
        
    def load_model(self):
        logger.info(f"Loading model from {self.args.model_path}")
        
        # Load model
        self.model = keras.models.load_model(self.args.model_path)
        
        # Load class names
        model_dir = Path(self.args.model_path).parent
        class_names_path = model_dir / 'class_names.json'
        
        if class_names_path.exists():
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
            logger.info(f"Loaded {len(self.class_names)} classes")
        else:
            logger.warning("Class names file not found. Using numeric labels.")
            self.class_names = None
            
        logger.info("Model loaded successfully")
        
    def preprocess_image(self, image):
        # Resize image
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [self.args.image_size, self.args.image_size])
        
        # Preprocess based on model type
        if self.args.base_model == 'EfficientNetB0':
            preprocess_fn = applications.efficientnet.preprocess_input
        elif self.args.base_model == 'ResNet50':
            preprocess_fn = applications.resnet50.preprocess_input
        elif self.args.base_model == 'MobileNetV2':
            preprocess_fn = applications.mobilenet_v2.preprocess_input
        else:
            raise ValueError(f"Unsupported base model: {self.args.base_model}")

        image = preprocess_fn(image)

        return image
    
    def predict_single_image(self, image_path):
        # Load image
        if isinstance(image_path, str):
            image = tf.keras.preprocessing.image.load_img(image_path)
            image = tf.keras.preprocessing.image.img_to_array(image)
        else:
            image = image_path
            
        # Preprocess
        processed_image = self.preprocess_image(image)
        processed_image = tf.expand_dims(processed_image, axis=0)
        
        # Predict
        start_time = time.time()
        predictions = self.model.predict(processed_image, verbose=0)
        inference_time = time.time() - start_time
        
        # Get top predictions
        top_k = min(self.args.top_k, len(predictions[0]))
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        top_probabilities = predictions[0][top_indices]
        
        # Create results
        results = []
        for i, (idx, prob) in enumerate(zip(top_indices, top_probabilities)):
            class_name = self.class_names[idx] if self.class_names else f"Class_{idx}"
            results.append({
                'rank': i + 1,
                'class_id': int(idx),
                'class_name': class_name,
                'confidence': float(prob),
                'percentage': float(prob * 100)
            })
            
        return {
            'predictions': results,
            'inference_time': inference_time,
            'top_class': results[0]['class_name'],
            'top_confidence': results[0]['confidence'],
        }
    
    def generate_summary(self, results):
        total_images = results['total_images']
        inference_times = [p['inference_time'] for p in results['predictions']]
        avg_inference_time = np.mean(inference_times)
        
        # Count predictions per class
        class_counts = {}
        for prediction in results['predictions']:
            top_class = prediction['top_prediction']['class']
            class_counts[top_class] = class_counts.get(top_class, 0) + 1
            
        print("\n" + "="*50)
        print("PREDICTION SUMMARY")
        print("="*50)
        print(f"Total images processed: {total_images}")
        print(f"Average inference time: {avg_inference_time:.4f} seconds")
        print(f"Total processing time: {sum(inference_times):.2f} seconds")
        print(f"Images per second: {total_images / sum(inference_times):.2f}")
        print("\nClass distribution:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_images) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        print("="*50)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification Inference')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size for inference')

    # SHOULD MATCH THE BASE MODEL USED DURING TRAINING
    parser.add_argument('--base_model', type=str, default='MobileNetV2',
                       choices=['EfficientNetB0', 'ResNet50', 'MobileNetV2'],
                       help='Base model architecture')

    # Input parameters
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to single image for prediction')
    # Output parameters
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to show')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize inference engine
    classifier = ImageClassifierInference(args)
    
    # Run inference
    if args.image_path:
        # Single image prediction
        if os.path.isfile(args.image_path):
            result = classifier.predict_single_image(args.image_path)
            
            print(f"\nPrediction for: {args.image_path}")
            print(f"Inference time: {result['inference_time']:.4f}s")
            print(f"Top prediction: {result['top_class']} ({result['top_confidence']*100:.2f}%)")
            print("\nTop predictions:")
            for pred in result['predictions']:
                print(f"  {pred['rank']}. {pred['class_name']}: {pred['percentage']:.2f}%")
                
        else:
            logger.error(f"Image file not found: {args.image_path}")
    else:
        logger.error("Please provide --image_path")

if __name__ == '__main__':
    main()