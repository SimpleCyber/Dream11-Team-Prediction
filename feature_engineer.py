import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
DATA_DIR = Path("Data")
FEATURES_DIR = DATA_DIR / "features"
FEATURES_DIR.mkdir(exist_ok=True)

def load_sports_data():
    """Load and merge all data sources efficiently"""
    # Load core player data
    players = pd.read_csv(DATA_DIR / "clean_data/cleaned_player_performance.csv")
    
    # Load matchup data
    def process_matchup(file):
        df = pd.read_csv(file)
        player_type = "batter" if "batter" in file.name else "bowler"
        status = "hot" if "hot" in file.name else "cold"
        return df.assign(
            player_type=player_type,
            status=status,
            form_strength=df['SR' if player_type == 'batter' else 'ER'] / 100
        )[['Name', 'Team', 'status', 'form_strength']]

    matchup_files = list((DATA_DIR / "matchup").glob("*.csv"))
    matchups = pd.concat(process_matchup(f) for f in matchup_files)
    
    # Load batter vs bowler data
    bvb = pd.read_csv(DATA_DIR / "t20_ipl_data/batter_vs_bowler_summary.csv")
    bvb_features = bvb.groupby('batter').agg(
        dominance_ratio=('total_runs', 'mean'),
        boundary_threat=('sixes', lambda x: x.sum()/len(x))
    ).reset_index()
    
    # Merge all data
    df = players.merge(matchups, left_on='player_name', right_on='Name', how='left')
    df = df.merge(bvb_features, left_on='player_name', right_on='batter', how='left')
    
    return df.drop(columns=['Name', 'batter'])

def create_features(df):
    """Create robust features with error handling"""
    try:
        # Base features
        df['boundary_ratio'] = (df['batting_ipl_4s'] + df['batting_ipl_6s']) / (df['batting_ipl_bf'] + 1e-6)
        df['economy_impact'] = df['bowling_ipl_wkts'] / (df['bowling_ipl_econ'] + 1e-6)
        
        # Advanced features
        df['form_impact'] = df['form_strength'].fillna(0) * np.log1p(df['batting_ipl_runs'])
        df['pressure_performance'] = np.where(
            df['status'] == 'hot',
            df['batting_ipl_sr'] * 1.1,
            df['batting_ipl_avg']
        )
        
        # Team features
        df['team_strength'] = df.groupby('team')['boundary_ratio'].transform('mean')
        
        # Final value metric
        df['value_score'] = (
            0.4 * df['boundary_ratio'] +
            0.3 * df['economy_impact'] +
            0.2 * df['form_impact'] +
            0.1 * df['team_strength']
        )
        
    except KeyError as e:
        print(f"Missing column: {e}")
        return None
    
    return df

def main():
    print("🚀 Starting feature engineering...")
    
    # Load and merge data
    df = load_sports_data()
    if df is None:
        print("❌ Failed to load data")
        return
    
    # Create features
    featured_df = create_features(df)
    if featured_df is None:
        print("❌ Feature creation failed")
        return
    
    # Save results
    save_path = FEATURES_DIR / "final_features.csv"
    featured_df.to_csv(save_path, index=False)
    print(f"✅ Success! Features saved to {save_path}")
    print(f"📊 Final dataset shape: {featured_df.shape}")

if __name__ == "__main__":
    main()