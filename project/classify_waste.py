import numpy as np
import cv2
from PIL import Image
import json
import sys
import os

class WasteClassifier:
    """
    CleanTech Waste Classifier using Transfer Learning
    
    This class simulates a transfer learning-based waste classifier
    that would typically use a pre-trained CNN model like ResNet, VGG, or EfficientNet
    """
    
    def __init__(self):
        self.waste_types = ['organic', 'plastic', 'paper', 'metal', 'glass', 'electronic']
        self.model_loaded = False
        
        # In a real implementation, you would load your trained model here
        # self.model = tf.keras.models.load_model('path/to/your/model.h5')
        
    def preprocess_image(self, image_path):
        """
        Preprocess the input image for classification
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size (typically 224x224 for transfer learning models)
            image = cv2.resize(image, (224, 224))
            
            # Normalize pixel values to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None
    
    def extract_features(self, preprocessed_image):
        """
        Extract features from the preprocessed image
        This would typically use the feature extraction layers of a pre-trained CNN
        
        Args:
            preprocessed_image (np.ndarray): Preprocessed image array
            
        Returns:
            np.ndarray: Extracted features
        """
        # Simulate feature extraction
        # In reality, this would be done by the pre-trained CNN layers
        features = np.random.rand(1, 2048)  # Typical feature vector size
        return features
    
    def classify_features(self, features):
        """
        Classify the extracted features into waste types
        
        Args:
            features (np.ndarray): Extracted features
            
        Returns:
            dict: Classification results with probabilities
        """
        # Simulate classification
        # In reality, this would use the trained classifier head
        
        # Generate random probabilities for each waste type
        probabilities = np.random.rand(len(self.waste_types))
        
        # Make one class more likely (simulate realistic classification)
        dominant_class = np.random.randint(0, len(self.waste_types))
        probabilities[dominant_class] += np.random.rand() * 2
        
        # Normalize probabilities
        probabilities = probabilities / np.sum(probabilities)
        
        # Create results dictionary
        results = {}
        for i, waste_type in enumerate(self.waste_types):
            results[waste_type] = float(probabilities[i])
        
        # Get predicted class
        predicted_class = self.waste_types[np.argmax(probabilities)]
        confidence = float(np.max(probabilities))
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': results
        }
    
    def get_waste_info(self, waste_type):
        """
        Get detailed information about the classified waste type
        
        Args:
            waste_type (str): The classified waste type
            
        Returns:
            dict: Description, recycling instructions, and environmental impact
        """
        waste_info = {
            'organic': {
                'description': 'Organic waste includes food scraps, yard trimmings, and other biodegradable materials. This type of waste can be composted to create nutrient-rich soil amendments.',
                'recycling_instructions': [
                    'Separate from other waste types',
                    'Remove any non-organic materials (plastic stickers, etc.)',
                    'Compost in backyard bin or municipal composting program',
                    'Avoid composting meat, dairy, or oily foods in home systems',
                    'Turn compost regularly to ensure proper decomposition'
                ],
                'environmental_impact': 'Proper composting of organic waste reduces methane emissions from landfills and creates valuable soil amendments, supporting sustainable agriculture and reducing the need for chemical fertilizers.'
            },
            'plastic': {
                'description': 'Plastic waste encompasses various polymer-based materials. Different types of plastics require different recycling processes, and proper sorting is crucial for effective recycling.',
                'recycling_instructions': [
                    'Check the recycling number on the bottom',
                    'Rinse containers to remove food residue',
                    'Remove caps and lids if required by local facility',
                    'Place in designated plastic recycling bin',
                    'Avoid mixing different plastic types'
                ],
                'environmental_impact': 'Recycling plastic reduces the need for virgin plastic production, saving petroleum resources and reducing greenhouse gas emissions. Each recycled plastic bottle saves enough energy to power a light bulb for 3 hours.'
            },
            'paper': {
                'description': 'Paper waste includes newspapers, cardboard, office paper, and packaging materials. Most paper products can be recycled multiple times before the fibers become too short.',
                'recycling_instructions': [
                    'Remove any plastic components or tape',
                    'Keep paper dry and clean',
                    'Separate different paper types if required',
                    'Place in paper recycling bin',
                    'Avoid recycling paper with heavy ink or coatings'
                ],
                'environmental_impact': 'Recycling paper saves trees, reduces water usage by 50%, and decreases energy consumption by 40% compared to making paper from virgin materials. One ton of recycled paper saves 17 trees.'
            },
            'metal': {
                'description': 'Metal waste includes aluminum cans, steel containers, and other metallic objects. Metals can be recycled indefinitely without losing their properties.',
                'recycling_instructions': [
                    'Rinse containers to remove food residue',
                    'Remove labels if possible',
                    'Separate aluminum from steel if required',
                    'Place in metal recycling bin',
                    'Check if lids need to be removed'
                ],
                'environmental_impact': 'Metal recycling saves 95% of the energy required to produce new metal from ore. Recycling aluminum cans saves enough energy to power a TV for 3 hours per can.'
            },
            'glass': {
                'description': 'Glass waste includes bottles, jars, and other glass containers. Glass is 100% recyclable and can be recycled endlessly without loss of quality or purity.',
                'recycling_instructions': [
                    'Rinse containers thoroughly',
                    'Remove metal lids and plastic labels',
                    'Separate by color if required by local facility',
                    'Place in glass recycling bin',
                    'Avoid mixing with other glass types (windows, mirrors)'
                ],
                'environmental_impact': 'Glass recycling reduces raw material extraction and saves energy. Using recycled glass in manufacturing reduces air pollution by 20% and water pollution by 50%.'
            },
            'electronic': {
                'description': 'Electronic waste (e-waste) includes computers, phones, and other electronic devices. These items contain valuable materials but also hazardous substances requiring special handling.',
                'recycling_instructions': [
                    'Take to certified e-waste recycling facility',
                    'Remove personal data from devices',
                    'Keep components together when possible',
                    'Check for manufacturer take-back programs',
                    'Never dispose of in regular trash'
                ],
                'environmental_impact': 'Proper e-waste recycling recovers valuable materials like gold, silver, and rare earth elements while preventing toxic substances from contaminating soil and water sources.'
            }
        }
        
        return waste_info.get(waste_type, {
            'description': 'Unknown waste type',
            'recycling_instructions': [],
            'environmental_impact': 'Unknown environmental impact'
        })
    
    def classify_image(self, image_path):
        """
        Complete classification pipeline for a waste image
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            dict: Complete classification results
        """
        try:
            # Preprocess image
            preprocessed_image = self.preprocess_image(image_path)
            if preprocessed_image is None:
                return {'error': 'Failed to preprocess image'}
            
            # Extract features
            features = self.extract_features(preprocessed_image)
            
            # Classify
            classification_results = self.classify_features(features)
            
            # Get detailed information
            waste_info = self.get_waste_info(classification_results['predicted_class'])
            
            # Combine results
            final_results = {
                **classification_results,
                **waste_info,
                'image_processed': True,
                'model_version': '1.0.0',
                'processing_method': 'Transfer Learning with CNN'
            }
            
            return final_results
            
        except Exception as e:
            return {'error': f'Classification failed: {str(e)}'}

def main():
    """
    Main function to run waste classification
    """
    if len(sys.argv) != 2:
        print("Usage: python classify_waste.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        sys.exit(1)
    
    # Initialize classifier
    classifier = WasteClassifier()
    
    # Classify image
    results = classifier.classify_image(image_path)
    
    # Output results as JSON
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
