# model_trainer.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
import joblib
import logging
from typing import Dict, Optional, List
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

class ModelTrainer:
    """Complete fantasy cricket model trainer with error handling"""
    
    def __init__(self):
        """Initialize with feature set and target variable"""
        self.features = [
            'boundary_ratio',
            'bowling_ipl_econ',
            'batting_ipl_avg',
            'is_hot_batter',
            'is_cold_batter',
            'is_hot_bowler',
            'is_cold_bowler',
            'partnership_avg',
            'recent_form'
        ]
        self.target = 'fantasy_points'
        self.model = self._create_pipeline()
        self.feature_importances_ = None

    def _create_pipeline(self) -> Pipeline:
        """Create robust preprocessing and modeling pipeline"""
        try:
            # Numeric features that need scaling
            numeric_features = ['bowling_ipl_econ', 'batting_ipl_avg']
            
            # Binary features to pass through
            binary_features = [
                'is_hot_batter',
                'is_cold_batter',
                'is_hot_bowler',
                'is_cold_bowler'
            ]
            
            # Create preprocessing steps
            preprocessor = ColumnTransformer([
                ('imputer', SimpleImputer(strategy='median'), numeric_features),
                ('scaler', StandardScaler(), numeric_features),
                ('binary_passthrough', 'passthrough', binary_features)
            ], remainder='drop')
            
            # Create complete pipeline
            return Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    enable_categorical=False
                ))
            ])
        except Exception as e:
            logging.error(f"Pipeline creation failed: {str(e)}")
            raise

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data structure"""
        try:
            if data.empty:
                raise ValueError("Empty DataFrame provided")
                
            missing_features = [f for f in self.features if f not in data.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
                
            if self.target not in data.columns:
                raise ValueError(f"Missing target column: {self.target}")
                
            if len(data) < 20:
                logging.warning(f"Low sample size: {len(data)} records")
                
            return True
        except Exception as e:
            logging.error(f"Data validation failed: {str(e)}")
            return False

    def train(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train model with comprehensive validation"""
        try:
            if not self._validate_data(data):
                raise ValueError("Invalid training data")
                
            X = data[self.features]
            y = data[self.target]
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            y = y.replace([np.inf, -np.inf], np.nan)
            
            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                self.model.fit(X_train, y_train)
                score = self.model.score(X_test, y_test)
                cv_scores.append(score)
                logging.info(f"Fold {fold+1} R²: {score:.3f}")
            
            # Final training on all data
            self.model.fit(X, y)
            
            # Save model artifacts
            joblib.dump(self.model, 'fantasy_model.pkl')
            self._save_feature_importances(X)
            
            return {
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'best_score': max(cv_scores)
            }
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            return {'error': str(e)}

    def _save_feature_importances(self, X: pd.DataFrame) -> None:
        """Calculate and save feature importance metrics"""
        try:
            # Get feature names after preprocessing
            feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
            
            # Get importance scores from XGBoost
            importances = self.model.named_steps['regressor'].feature_importances_
            
            # Create and save importance dataframe
            self.feature_importances_ = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            self.feature_importances_.to_csv('feature_importances.csv', index=False)
            
            # Plot importance
            plt.figure(figsize=(10, 6))
            self.feature_importances_.plot.bar(x='feature', y='importance')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()
            
        except Exception as e:
            logging.warning(f"Could not save feature importances: {str(e)}")

    def predict(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate predictions with validation"""
        try:
            if not self._validate_data(data):
                raise ValueError("Invalid prediction data")
                
            return self.model.predict(data[self.features])
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return None

if __name__ == "__main__":
    """Example usage"""
    try:
        from data_processor import DataProcessor
        
        # Initialize components
        trainer = ModelTrainer()
        processor = DataProcessor()
        
        # Load and validate data
        match_num = int(input("Enter match number for training: "))
        data = processor.load_match_data(match_num)
        
        if data is None:
            raise ValueError("Data loading failed")
            
        # Train model
        results = trainer.train(data)
        
        if 'error' in results:
            print(f"❌ Training failed: {results['error']}")
        else:
            print(f"✅ Training successful - Average R²: {results['cv_mean']:.3f}")
            
    except Exception as e:
        print(f"Fatal error: {str(e)}")