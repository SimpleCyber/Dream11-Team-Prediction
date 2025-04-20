import pandas as pd
from pathlib import Path

# Directory setup
DATA_DIR = Path("Data")
FEATURES_DIR = DATA_DIR / "features"
SQUAD_DIR = DATA_DIR / "squad_data"
MERGED_DIR = DATA_DIR / "merged"
MERGED_DIR.mkdir(exist_ok=True)

# Full Name → Short Code Mapping
REVERSE_TEAM_MAPPING = {
    'PUNJAB KINGS': 'PBKS',
    'PUNJAB KINGS': 'PNS',
    'SUNRISERS HYDERABAD': 'SRH',
    'ROYAL CHALLENGERS BANGALORE': 'RCB',
    'CHENNAI SUPER KINGS': 'CHE',
    'MUMBAI INDIANS': 'MI',
    'KOLKATA KNIGHT RIDERS': 'KKR',
    'DELHI CAPITALS': 'DC',
    'DELHI CAPITALS': 'DG',
    'RAJASTHAN ROYALS': 'RR',
    'LUCKNOW SUPER GIANTS': 'LSG',
    'GUJARAT TITANS': 'GT'
}

def load_and_standardize_lineup(match_number):
    """Load and standardize lineup data with team short codes"""
    lineup_path = SQUAD_DIR / "lineups" / f"Match_{match_number}_Lineup.csv"
    lineup = pd.read_csv(lineup_path)
    
    # Standardize player and team names
    lineup['player'] = lineup['Player Name'].str.strip().str.lower()
    full_names = lineup['Team'].str.strip().str.upper()
    lineup['team'] = full_names.map(REVERSE_TEAM_MAPPING).fillna(full_names)
    
    return lineup

def merge_datasets(match_number):
    """Merge player features with lineup info for a given match"""
    # Load player features
    players = pd.read_csv(FEATURES_DIR / "player_performance_with_features.csv")
    players['player'] = players['player_name'].str.strip().str.lower()
    players['team'] = players['team'].str.strip().str.upper()
    
    # Load standardized lineup
    lineup = load_and_standardize_lineup(match_number)
    
    # Merge datasets
    merged = pd.merge(
        players,
        lineup[['player', 'team', 'IsPlaying', 'Credits', 'Player Type']],
        on=['player', 'team'],
        how='inner'
    )
    
    # Filter only playing players
    merged = merged[merged['IsPlaying'] == 'PLAYING']
    
    if merged.empty:
        print("\n❌ No matching players found. Debug info:")
        print("Features teams:", players['team'].unique())
        print("Lineup teams:", lineup['team'].unique())
    else:
        merged_path = MERGED_DIR / f"merged_match_{match_number}.csv"
        merged.to_csv(merged_path, index=False)
        print(f"\n✅ Successfully merged {len(merged)} playing players.")
        print(f"Saved to: {merged_path}")
    
    return merged

if __name__ == "__main__":
    match_num = int(input("Enter match number to process: "))
    merge_datasets(match_num)
