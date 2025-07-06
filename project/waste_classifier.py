import os
import sys
import json
import random
import time
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import numpy as np

class CleanTechWasteClassifier:
    """
    CleanTech Waste Classifier
    Simulates advanced transfer learning-based waste classification
    """
    
    def __init__(self):
        self.waste_types = ['organic', 'plastic', 'paper', 'metal', 'glass', 'electronic']
        self.model_version = "1.0.0"
        self.confidence_threshold = 0.75
        
        # Waste type information database
        self.waste_info = {
            'organic': {
                'description': 'Organic waste includes food scraps, yard trimmings, and other biodegradable materials. This type of waste can be composted to create nutrient-rich soil amendments and reduce methane emissions from landfills.',
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
                'description': 'Plastic waste encompasses various polymer-based materials including bottles, containers, and packaging. Different types of plastics require different recycling processes, and proper sorting is crucial for effective recycling.',
                'recycling_instructions': [
                    'Check the recycling number on the bottom (1-7)',
                    'Rinse containers to remove food residue',
                    'Remove caps and lids if required by local facility',
                    'Place in designated plastic recycling bin',
                    'Avoid mixing different plastic types'
                ],
                'environmental_impact': 'Recycling plastic reduces the need for virgin plastic production, saving petroleum resources and reducing greenhouse gas emissions. Each recycled plastic bottle saves enough energy to power a light bulb for 3 hours.'
            },
            'paper': {
                'description': 'Paper waste includes newspapers, cardboard, office paper, and packaging materials. Most paper products can be recycled multiple times before the fibers become too short for further processing.',
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
                'description': 'Metal waste includes aluminum cans, steel containers, and other metallic objects. Metals can be recycled indefinitely without losing their properties, making them highly valuable recyclable materials.',
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
                'description': 'Glass waste includes bottles, jars, and other glass containers. Glass is 100% recyclable and can be recycled endlessly without loss of quality or purity, making it one of the most sustainable packaging materials.',
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
                'description': 'Electronic waste (e-waste) includes computers, phones, and other electronic devices. These items contain valuable materials like gold and silver but also hazardous substances requiring special handling.',
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
    
    def preprocess_image(self, image_path):
        """
        Preprocess the uploaded image for classification
        """
        try:
            # Open and validate image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to standard size (224x224 for most CNN models)
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                
                # Convert to numpy array and normalize
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                return img_array
                
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    def extract_features(self, image_array):
        """
        Simulate feature extraction using transfer learning
        In a real implementation, this would use a pre-trained CNN
        """
        # Simulate feature extraction process
        time.sleep(1)  # Simulate processing time
        
        # Generate realistic feature vector
        features = np.random.rand(2048)  # Typical CNN feature vector size
        
        # Add some image-based variations
        mean_intensity = np.mean(image_array)
        std_intensity = np.std(image_array)
        
        # Modify features based on image characteristics
        features[0] = mean_intensity
        features[1] = std_intensity
        
        return features
    
    def classify_waste(self, features):
        """
        Classify the waste based on extracted features
        """
        # Simulate neural network classification
        # Generate base probabilities
        probabilities = {}
        
        # Create realistic probability distribution
        for waste_type in self.waste_types:
            probabilities[waste_type] = random.uniform(0.05, 0.25)
        
        # Select a dominant class with higher probability
        dominant_class = random.choice(self.waste_types)
        probabilities[dominant_class] = random.uniform(0.6, 0.9)
        
        # Normalize probabilities to sum to 1
        total = sum(probabilities.values())
        for waste_type in probabilities:
            probabilities[waste_type] /= total
        
        # Get predicted class and confidence
        predicted_class = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_class]
        
        return predicted_class, confidence, probabilities
    
    def generate_report(self, predicted_class, confidence, probabilities, processing_time):
        """
        Generate comprehensive classification report
        """
        waste_data = self.waste_info[predicted_class]
        
        report = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {k: float(v) for k, v in probabilities.items()},
            'description': waste_data['description'],
            'recycling_instructions': waste_data['recycling_instructions'],
            'environmental_impact': waste_data['environmental_impact'],
            'metadata': {
                'model_version': self.model_version,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'method': 'Transfer Learning CNN',
                'confidence_threshold': self.confidence_threshold
            }
        }
        
        return report
    
    def classify_image(self, image_path):
        """
        Complete classification pipeline
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not os.path.exists(image_path):
                raise Exception("Image file not found")
            
            # Preprocess image
            print("Preprocessing waste image...")
            image_array = self.preprocess_image(image_path)
            
            # Extract features
            print("Extracting features using transfer learning...")
            features = self.extract_features(image_array)
            
            # Classify
            print("Classifying waste type...")
            predicted_class, confidence, probabilities = self.classify_waste(features)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Generate report
            report = self.generate_report(predicted_class, confidence, probabilities, processing_time)
            
            print(f"Classification complete: {predicted_class} ({confidence:.1%} confidence)")
            
            return report
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def main():
    """
    Main function for command-line usage
    """
    if len(sys.argv) != 2:
        print("Usage: python waste_classifier.py <image_path>")
        print("Example: python waste_classifier.py waste_item.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("=" * 50)
    print("CleanTech Waste Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = CleanTechWasteClassifier()
    
    # Classify image
    result = classifier.classify_image(image_path)
    
    # Output results
    if 'error' in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    else:
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
