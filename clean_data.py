# main.py
import pandas as pd
import numpy as np
import os
from pathlib import Path

class DataMerger:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "Data"
        
        # Load core datasets
        self.batting_stats = self._load_csv("ipl_batting_stats (3).csv", "Batting Stats")
        self.bowling_stats = self._load_csv("ipl_bowling_stats (3).csv", "Bowling Stats")
        self.matchups = self._load_csv("batter_vs_bowler_summary.csv", "Matchups")
        self.venue_data = self._load_csv("clean_data/cleaned_historical_pitch_data.csv", "Venue Data")
        self.squad_data = self._load_csv("squad_data/SquadPlayerNames_IndianT20League.csv", "Squad Data")
        
        # Create venue mapping
        self.venue_mapping = self._create_venue_mapping()

    def _load_csv(self, rel_path, dataset_name):
        """Safe CSV loading with error handling"""
        full_path = self.data_dir / rel_path
        try:
            df = pd.read_csv(full_path)
            print(f"✅ Loaded {dataset_name} from {full_path}")
            return df
        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {str(e)}")
            return pd.DataFrame()

    def _create_venue_mapping(self):
        """Generate venue mapping if missing"""
        map_path = self.data_dir / "venue_mapping.csv"
        if not map_path.exists():
            print("⚠️ Creating default venue mapping...")
            venues = pd.DataFrame({
                'match_id': range(100),
                'venue': ['Unknown'] * 100
            })
            venues.to_csv(map_path, index=False)
            return venues
        return pd.read_csv(map_path)

    def merge_match_data(self, lineup_path):
        """Process a single match lineup"""
        try:
            lineup = pd.read_csv(lineup_path)
            match_id = int(lineup_path.stem.split('_')[1])
            
            # Add core match info
            lineup['match_id'] = match_id
            lineup['venue'] = self.venue_mapping.loc[
                self.venue_mapping['match_id'] == match_id,
                'venue'
            ].values[0]

            # Merge batting stats
            merged = pd.merge(
                lineup, 
                self.batting_stats,
                left_on='Player Name',
                right_on='player',
                how='left',
                suffixes=('', '_bat')
            )

            # Merge bowling stats
            merged = pd.merge(
                merged,
                self.bowling_stats,
                left_on='Player Name',
                right_on='bowler',
                how='left',
                suffixes=('', '_bowl')
            )

            # Add venue features
            venue_features = self.venue_data[
                self.venue_data['venue'] == merged['venue'].iloc[0]
            ].iloc[0].to_dict()
            
            for key, value in venue_features.items():
                merged[f'venue_{key}'] = value

            return merged
        
        except Exception as e:
            print(f"🚨 Error processing {lineup_path.name}: {str(e)}")
            return pd.DataFrame()

    def process_all_matches(self):
        """Process all lineup files"""
        lineup_dir = self.data_dir / "squad_data/lineups"
        all_matches = []
        
        for f in lineup_dir.glob("Match_*.csv"):
            print(f"🔨 Processing {f.name}...")
            match_data = self.merge_match_data(f)
            if not match_data.empty:
                all_matches.append(match_data)
        
        return pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()

class DataCleaner:
    def __init__(self, raw_df):
        self.df = raw_df
        self._validate_input()

    def _validate_input(self):
        """Ensure required columns exist"""
        required = ['Player Name', 'Team', 'Credits']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def calculate_points(self):
        """Flexible fantasy points calculation"""
        if 'FantasyPoints' not in self.df.columns:
            self.df['FantasyPoints'] = 0
            
            # Batting components
            batting_points = self.df.get('runs', 0) * 1
            batting_points += self.df.get('4s', 0) * 0.5
            batting_points += self.df.get('6s', 0) * 1
            self.df['FantasyPoints'] += batting_points.fillna(0)

            # Bowling components
            bowling_points = self.df.get('wickets', 0) * 25
            economy = self.df.get('economy', 20).clip(upper=20)
            bowling_points += np.where(economy < 6, 12, 0)
            bowling_points += np.where(economy.between(6, 7), 6, 0)
            self.df['FantasyPoints'] += bowling_points.fillna(0)

        return self

    def clean_columns(self):
        """Column cleanup and standardization"""
        # Remove duplicate columns
        self.df = self.df.loc[:, ~self.df.columns.duplicated()]

        # Standardize column names
        self.df.columns = (
            self.df.columns.str.replace(' ', '_')
            .str.lower()
            .str.replace(r'[^a-z0-9_]', '', regex=True)
        )
        
        # Select final columns
        keep_cols = [
            'player_name', 'team', 'credits', 'player_type',
            'batting_strike_rate', 'bowling_strike_rate',
            'venue_avg_score', 'fantasypoints'
        ]
        
        self.df = self.df[[c for c in keep_cols if c in self.df.columns]]
        return self

    def handle_missing(self):
        """Smart missing value handling"""
        # Numeric columns
        num_cols = self.df.select_dtypes(include=np.number).columns
        self.df[num_cols] = self.df[num_cols].fillna(
            self.df[num_cols].median()
        ).clip(lower=0)

        # Categorical columns
        cat_cols = self.df.select_dtypes(exclude=np.number).columns
        for col in cat_cols:
            self.df[col] = self.df[col].fillna('unknown')

        return self

    def save(self, output_path):
        """Save cleaned data"""
        self.df.to_csv(output_path, index=False)
        print(f"💾 Saved cleaned data to {output_path}")
        return self

if __name__ == "__main__":
    try:
        # Step 1: Merge data
        merger = DataMerger()
        merged_data = merger.process_all_matches()
        
        if merged_data.empty:
            raise ValueError("No data merged - check input files")
            
        # Step 2: Clean data
        cleaner = DataCleaner(merged_data)
        (cleaner.calculate_points()
               .clean_columns()
               .handle_missing()
               .save("clean_training_data.csv"))
        
        print("✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        print("Verify:")
        print("1. Input files exist in Data/ directory")
        print("2. Lineup files follow naming convention: Match_XX_Lineup.csv")
        print("3. Required columns in lineup files: Player Name, Team, Credits")