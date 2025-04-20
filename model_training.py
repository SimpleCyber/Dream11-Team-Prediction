import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
import joblib
from pathlib import Path

def train_fantasy_model(merged_file_path):
    df = pd.read_csv(merged_file_path)
    
    # Calculate target (Dream11 points)
    df['fantasy_points'] = (
        df['batting_ipl_runs'] * 1 +
        df['batting_ipl_4s'] * 0.5 +
        df['batting_ipl_6s'] * 1 +
        df['bowling_ipl_wkts'] * 25
    )
    
    # Simplified feature set
    features = [
        'boundary_ratio',        # (4s + 6s)/balls faced
        'bowling_ipl_econ',      # Economy rate
        'batting_ipl_avg',       # Batting average
        'is_allrounder'          # Boolean flag
    ]
    
    # Prepare data
    X = df[features].fillna(0)
    y = df['fantasy_points']
    
    # Reduced complexity model
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=50,        # Reduced from 100
        max_depth=2,            # Reduced from 3
        learning_rate=0.05,     # Smaller steps
        reg_alpha=0.1,          # L1 regularization
        reg_lambda=0.1,         # L2 regularization
        random_state=42
    )
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-validation R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    
    # Final training
    model.fit(X, y)
    joblib.dump(model, 'fantasy_model.pkl')
    print("✅ Model saved with reduced complexity")

if __name__ == "__main__":
    match_num = int(input("Enter match number for training: "))
    train_fantasy_model(f"Data/merged/merged_match_{match_num}.csv")