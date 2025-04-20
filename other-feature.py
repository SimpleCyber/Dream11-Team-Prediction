import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
DATA_DIR = Path("Data")
FEATURES_DIR = DATA_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

class FeatureEngineeringPipeline:
    def __init__(self):
        self.df = None
        self.feature_metadata = []
        self.pipeline_steps = [
            self.load_core_data,
            self.process_matchup_data,
            self.process_batter_bowler,
            self.process_partnerships,
            self.create_advanced_features,
            self.validate_features,
            self.save_features
        ]
        self.config = {
            'required_partnership_columns': {
                'batter1': 'player_1',      # Update based on actual column names
                'batter2': 'player_2',
                'runs': 'total_runs',
                'balls': 'balls_played',
                'team_score': 'team_total'
            }
        }

    def _handle_error(self, error_msg, critical=False):
        """统一错误处理"""
        logging.error(error_msg)
        if critical:
            raise RuntimeError(error_msg)

    def load_core_data(self):
        """Load and validate core player data with enhanced checks"""
        try:
            core_path = DATA_DIR / "clean_data/cleaned_player_performance.csv"
            if not core_path.exists():
                self._handle_error("Core data file missing", critical=True)

            self.df = pd.read_csv(core_path)
            required_cols = {'player_name', 'team', 'batting_ipl_runs', 'bowling_ipl_wkts'}
            missing = required_cols - set(self.df.columns)
            if missing:
                self._handle_error(f"Missing required columns: {missing}", critical=True)
            
            logging.info(f"Loaded core data with {len(self.df)} records")
            return True
        except Exception as e:
            self._handle_error(f"Error loading core data: {str(e)}", critical=True)

    def process_matchup_data(self):
        """Process matchup data with date format handling"""

        
        try:
            matchup_dir = DATA_DIR / "matchup"
            if not matchup_dir.exists():
                logging.warning("Matchup directory not found")
                return False

            matchup_files = list(matchup_dir.glob("*.csv"))
            if not matchup_files:
                logging.warning("No matchup files found")
                return False

            matchup_dfs = []
            for file in tqdm(matchup_files, desc="Processing matchup data"):
                try:
                    df = pd.read_csv(file)
                    
                    # Validate required columns
                    if 'Date' not in df.columns:
                        logging.warning(f"Skipping {file.name} - missing Date column")
                        continue
                except Exception as e:
                    logging.warning(f"Error reading file {file.name}: {str(e)}")
                    continue

                    # Convert dates with explicit format
                    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
                    df = df[df['Date'].notna()]  # Remove invalid dates
                    
                    # Calculate form duration
                    df['form_duration'] = (datetime.now() - df['Date']).dt.days
                    
                    # Add matchup type metadata
                    if 'batter' in file.stem.lower():
                        if 'SR' not in df.columns:
                            logging.warning(f"Skipping {file.name} - missing SR column")
                            continue
                        # Convert SR to numeric, coercing errors to NaN
                        df['SR'] = pd.to_numeric(df['SR'], errors='coerce')
                        df['form_strength'] = df['SR'] / 200  # Normalize SR
                    elif 'bowler' in file.stem.lower():
                        if 'ER' not in df.columns:
                            logging.warning(f"Skipping {file.name} - missing ER column")
                            continue
                        # Convert ER to numeric
                        df['ER'] = pd.to_numeric(df['ER'], errors='coerce')
                        df['form_strength'] = 1 - (df['ER'] / 12)  # Normalize ER
                    
                    # Create a mapping for team abbreviations to full names
                    team_mapping = {
                        'CHE': 'Chennai Super Kings',
                        'MI': 'Mumbai Indians',
                        # Add all necessary mappings
                    }

                    # Apply mapping to matchup data before merging
                    df['Team'] = df['Team'].map(team_mapping).fillna(df['Team'])

                    matchup_dfs.append(df[['Name', 'Team', 'form_strength', 'form_duration']])

            if matchup_dfs:
                matchup_df = pd.concat(matchup_dfs)
                self.df = self.df.merge(matchup_df, 
                                      left_on=['player_name', 'team'],
                                      right_on=['Name', 'Team'], 
                                      how='left')
                # Check if merge was successful
                if 'form_strength' not in self.df.columns:
                    logging.warning("Merge failed for matchup data. Using defaults.")
                    self.df['form_strength'] = 0.5  # Default neutral value
                    self.df['form_duration'] = 365  # 1 year default
                else:
                    # Fill missing values post-merge
                    self.df['form_strength'] = self.df['form_strength'].fillna(0.5)
                    self.df['form_duration'] = self.df['form_duration'].fillna(365)
                logging.info("Matchup data merged successfully")
                return True
            return False

        except Exception as e:
            self._handle_error(f"Error processing matchup data: {str(e)}")
            return False

    def process_batter_bowler(self):
        """Process batter vs bowler data with validation"""
        try:
            bvb_path = DATA_DIR / "t20_ipl_data/batter_vs_bowler_summary.csv"
            if not bvb_path.exists():
                logging.warning("Batter vs bowler data not found")
                return False

            bvb = pd.read_csv(bvb_path)
            required_cols = {'batter', 'total_runs', 'total_balls', 'fours', 'sixes', 'dismissals'}
            missing = required_cols - set(bvb.columns)
            if missing:
                logging.warning(f"Skipping batter-bowler processing - missing columns: {missing}")
                return False

            # Calculate features
            bvb['dominance_ratio'] = bvb['total_runs'] / (bvb['total_balls'] + 1e-6)
            bvb['boundary_threat'] = (bvb['fours'] + bvb['sixes']) / (bvb['total_balls'] + 1e-6)
            bvb['survival_rate'] = 1 - (bvb['dismissals'] / (bvb['total_balls'] + 1e-6))
            
            bvb_agg = bvb.groupby('batter').agg(
                avg_dominance=('dominance_ratio', 'mean'),
                max_boundary_threat=('boundary_threat', 'max'),
                mean_survival_rate=('survival_rate', 'mean')
            ).reset_index()
            
            self.df = self.df.merge(bvb_agg, 
                                  left_on='player_name',
                                  right_on='batter', 
                                  how='left')
            logging.info("Batter vs bowler features added")
            return True
        except Exception as e:
            self._handle_error(f"Error processing batter-bowler data: {str(e)}")
            return False

    def process_partnerships(self):
        """Process partnership data with error resilience"""
        try:
            partners_path = DATA_DIR / "ipl_partnerships_last10years.csv"
            if not partners_path.exists():
                logging.warning("Partnership data not found")
                return False

            partners = pd.read_csv(partners_path).rename(columns={
                'batsman1': 'batter1',
                'batsman2': 'batter2',
                'total_runs': 'runs',
                'balls_faced': 'balls',
                'team_total': 'team_score'
            })
            if partners.empty:
                logging.warning("Empty partnership dataframe")
                return False

            # Check for required columns
            required_cols = {'batter1', 'batter2', 'runs', 'balls', 'team_score'}
            missing = required_cols - set(partners.columns)
            if missing:
                logging.warning(f"Skipping partnership processing - missing columns: {missing}")
                return False

            # Calculate safe features
            partners['run_rate'] = partners['runs'] / (partners['balls'] + 1e-6)
            partners['pressure_performance'] = partners['runs'] / (partners['team_score'] + 1e-6)
            
            # Create symmetric partnerships
            partners_reversed = partners.rename(columns={
                'batter1': 'batter2',
                'batter2': 'batter1'
            })
            full_partners = pd.concat([partners, partners_reversed])
            
            partner_agg = full_partners.groupby('batter1').agg(
                partnership_impact=('run_rate', 'mean'),
                avg_pressure_performance=('pressure_performance', 'mean')
            ).reset_index()
            
            self.df = self.df.merge(partner_agg,
                                  left_on='player_name',
                                  right_on='batter1',
                                  how='left')
            logging.info("Partnership features processed")
            return True
        except Exception as e:
            self._handle_error(f"Error processing partnership data: {str(e)}")
            return False

    def create_advanced_features(self):
        """Create features with validation checks"""
        try:
            # Boundary ratio with validation
            if {'batting_ipl_4s', 'batting_ipl_6s', 'batting_ipl_bf'}.issubset(self.df.columns):
                self.df['boundary_ratio'] = (self.df['batting_ipl_4s'] + self.df['batting_ipl_6s']) / (self.df['batting_ipl_bf'] + 1e-6)
                self.df['boundary_ratio'] = self.df['boundary_ratio'].clip(0, 1)
            else:
                logging.warning("Missing columns for boundary ratio calculation")

            # Economy impact calculation
            if {'bowling_ipl_wkts', 'bowling_ipl_econ'}.issubset(self.df.columns):
                self.df['economy_impact'] = self.df['bowling_ipl_wkts'] / (self.df['bowling_ipl_econ'] + 1e-6)
                self.df['economy_impact'] = self.df['economy_impact'].clip(0, 10)
            else:
                logging.warning("Missing columns for economy impact calculation")

            # Composite value index
            components = {
                'boundary_ratio': 0.3,
                'economy_impact': 0.25,
                'form_strength': 0.2,
                'avg_dominance': 0.15,
                'partnership_impact': 0.1
            }

            total_weight = 0
            self.df['value_index'] = 0

            for feature, weight in components.items():
                if feature in self.df.columns:
                    total_weight += weight
                    self.df['value_index'] += self.df[feature].fillna(0) * weight
                else:
                    logging.warning(f"Using fallback for missing feature: {feature}")
                    total_weight += weight  # Keep weight distribution

            # Normalize to 0-1 range
            self.df['value_index'] = self.df['value_index'] / total_weight

            self.df['value_index'] = np.clip(
                self.df['value_index'], 
                0.1,  # Minimum value
                0.9    # Maximum value
            )

            logging.info("Advanced features created")
            return True
        except Exception as e:
            self._handle_error(f"Error creating features: {str(e)}")
            return False

    def validate_features(self):
        """Validate final feature set"""
        try:
            # Add imputation step before validation
            imputation_rules = {
                'boundary_ratio': self.df['boundary_ratio'].median(),
                'economy_impact': 0,
                'value_index': self.df['value_index'].median(),
                'avg_dominance': self.df['avg_dominance'].mean(),
                'partnership_impact': 0.05  # Default partnership impact
            }

            self.df = self.df.fillna(imputation_rules)

            # Check for NaNs
            nan_report = self.df.isna().mean()
            if nan_report.max() > 0.3:
                self._handle_error("Excessive missing values in features")
            
            # Validate value ranges
            validations = {
                'boundary_ratio': (0, 1),
                'economy_impact': (0, 10),
                'value_index': (0, 1)
            }
            
            for feature, (min_val, max_val) in validations.items():
                if feature in self.df.columns:
                    if not self.df[feature].between(min_val, max_val).all():
                        logging.warning(f"Feature {feature} contains out-of-range values")
                        self.df[feature] = self.df[feature].clip(min_val, max_val)
            
            quality_report = {
                'missing_values': self.df.isna().mean().to_dict(),
                'value_index_stats': self.df['value_index'].describe().to_dict(),
                'feature_correlations': self.df.corr()['value_index'].to_dict()
            }

            pd.DataFrame(quality_report).to_csv("Data/features/quality_report.csv")

            logging.info("Feature validation completed")
            return True
        except Exception as e:
            self._handle_error(f"Validation failed: {str(e)}")
            return False

    def save_features(self):
        """Save features with metadata"""
        try:
            output_path = FEATURES_DIR / "final_features.csv"
            self.df.to_csv(output_path, index=False)
            
            # Save pipeline
            pipeline_path = FEATURES_DIR / "feature_pipeline.pkl"
            joblib.dump(self, pipeline_path)
            
            logging.info(f"Features and pipeline saved to {output_path}")
            return True
        except Exception as e:
            self._handle_error(f"Error saving features: {str(e)}")
            return False

    def run_pipeline(self):
        quality_metrics = {}

        for step in self.pipeline_steps:
            success = step()
            quality_metrics[step.__name__] = {
                'success': success,
                'row_count': len(self.df),
                'missing_values': self.df.isna().sum().to_dict()
            }

        # Save quality report
        pd.DataFrame(quality_metrics).to_csv("pipeline_quality_report.csv")
        return True

if __name__ == "__main__":
    pipeline = FeatureEngineeringPipeline()
    if pipeline.run_pipeline():
        logging.info("Feature engineering pipeline completed successfully")
    else:
        logging.error("Feature engineering pipeline failed")