from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging
import json

# Add the parent directory to Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our ML implementations
try:
    from algorithms.naive_bayes_implementation.naive_bayes import NaiveBayes
    from algorithms.knn_classifier_implementation.knn_classifier import KNNClassifier
    from algorithms.decision_tree_implementation.decision_tree import DecisionTree
    from algorithms.adaboost_implementation.adaboost_classifier import AdaBoostClassifier
    from algorithms.KMeans_implementation.KMeans import KMeans
    from algorithms.PCA_implementation.preprocess import pca
    logger.info("Successfully imported all ML implementations")
except ImportError as e:
    logger.error(f"Error importing ML implementations: {str(e)}")
    raise

app = Flask(__name__)

# Only load the Iris dataset
data_path = parent_dir / 'data' / 'Iris_train.csv'
data = pd.read_csv(data_path)
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']

# Initialize and train models as before
# ...

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_features')
def get_features():
    """Return feature information for the Iris dataset"""
    return jsonify({
        'features': ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
        'description': 'Classify iris flowers into three species based on sepal and petal measurements.'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_name = data['model']
        
        # Initialize models for this dataset if not already done
        models = {
            'naive_bayes': NaiveBayes(),
            'knn': KNNClassifier(),
            'decision_tree': DecisionTree(),
            'adaboost': AdaBoostClassifier(base_estimator=DecisionTree(max_depth=2, criterion='entropy')),
            'kmeans': KMeans(n_clusters=3),
            'pca': None
        }
        
        # Train models
        for name, model in models.items():
            if name == 'pca':
                models['pca'] = pca(X.to_numpy(), n_components=2)
            elif name == 'kmeans':
                model.fit(X)
            else:
                model.fit(X, y)
        
        # Get feature values
        features = np.array([[
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]])
        
        model = models[model_name]
        
        if model_name == 'pca':
            # PCA returns transformed data
            principal_components, _ = model
            result = features @ principal_components
            return jsonify({
                'success': True,
                'result': result.tolist(),
                'message': 'PCA transformation completed'
            })
        elif model_name == 'kmeans':
            # KMeans returns cluster labels
            prediction = model.predict(features)
            return jsonify({
                'success': True,
                'prediction': f'Cluster {int(prediction[0])}',
                'message': 'Clustering completed'
            })
        else:
            # Convert numpy array to DataFrame for models that expect it
            if model_name in ['naive_bayes', 'knn', 'decision_tree', 'adaboost']:
                features_df = pd.DataFrame(features, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
                
                # Special handling for Decision Tree and AdaBoost
                if model_name in ['decision_tree', 'adaboost']:
                    # Get the first row as a Series for prediction
                    features_series = features_df.iloc[0]
                    prediction = model.predict(pd.DataFrame([features_series]))
                else:
                    prediction = model.predict(features_df)
                
                response = {
                    'success': True,
                    'prediction': str(prediction[0]),
                    'message': 'Prediction completed'
                }
                
                # Get probabilities
                try:
                    if hasattr(model, 'predict_proba'):
                        if model_name in ['decision_tree', 'adaboost']:
                            probabilities = model.predict_proba(pd.DataFrame([features_series]))
                        else:
                            probabilities = model.predict_proba(features_df)
                        # Convert probabilities to a dictionary with class labels
                        prob_dict = {
                            str(label): float(prob) 
                            for label, prob in zip(model.classes_, probabilities.iloc[0])
                        }
                        response['probabilities'] = prob_dict
                except Exception as e:
                    logger.warning(f"Could not get probabilities for {model_name}: {str(e)}")
                
                return jsonify(response)
            else:
                # Other classification models
                prediction = model.predict(features)
                pred_value = str(prediction[0]) if isinstance(prediction[0], (np.integer, np.floating)) else prediction[0]
                
                response = {
                    'success': True,
                    'prediction': pred_value,
                    'message': 'Prediction completed'
                }
                
                # Add probabilities if available
                try:
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(features)
                        prob_dict = {
                            str(label): float(prob) 
                            for label, prob in zip(model.classes_, probabilities[0])
                        }
                        response['probabilities'] = prob_dict
                except Exception as e:
                    logger.warning(f"Could not get probabilities for {model_name}: {str(e)}")
                
                return jsonify(response)
                
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)