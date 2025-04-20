import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Dict
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data_processing.log"), logging.StreamHandler()],
)


class DataProcessor:
    """Handles data loading and merging with proper error handling"""

    def __init__(self):
        self.data_dirs = {
            "features": Path("Data/features"),
            "matchup": Path("Data/matchup"),
            "partnerships": Path("Data/partnerships_clean.csv"),
            "merged": Path("Data/merged"),
        }
        self._validate_directories()

        self.team_mapping = {
            "SUNRISERS HYDERABAD": "SRH",
            "ROYAL CHALLENGERS BANGALORE": "RCB",
            "CHENNAI SUPER KINGS": "CSK",
            "MUMBAI INDIANS": "MI",
            "KOLKATA KNIGHT RIDERS": "KKR",
            "DELHI CAPITALS": "DC",
            "RAJASTHAN ROYALS": "RR",
            "LUCKNOW SUPER GIANTS": "LSG",
            "GUJARAT TITANS": "GT",
        }

    def _validate_directories(self) -> None:
        """Ensure required directories exist"""
        for path in self.data_dirs.values():
            if path.suffix == "":  # Directory
                path.mkdir(parents=True, exist_ok=True)

    def _load_with_fallback(
        self, path: Path, required_cols: list = None
    ) -> pd.DataFrame:
        """Safe loading with fallback to empty DataFrame"""
        try:
            if not path.exists():
                logging.warning(f"File not found: {path}")
                return pd.DataFrame(columns=required_cols if required_cols else [])

            df = pd.read_csv(path)

            if required_cols:
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    logging.warning(f"Missing columns in {path.name}: {missing}")
                    return pd.DataFrame(columns=required_cols)

            return df

        except Exception as e:
            logging.error(f"Error reading {path}: {str(e)}")
            return pd.DataFrame(columns=required_cols if required_cols else [])

    def _process_matchups(self, players: pd.DataFrame) -> pd.DataFrame:
        """Add hot/cold player flags"""
        for player_type in ["hot_batter", "cold_batter", "hot_bowler", "cold_bowler"]:
            matchup_file = f"{player_type}s.csv"
            matchup_df = self._load_with_fallback(
                self.data_dirs["matchup"] / matchup_file, required_cols=["Name", "Team"]
            )
            players[f"is_{player_type}"] = (
                players["player"]
                .isin(matchup_df["Name"].str.strip().str.lower())
                .astype(int)
                if not matchup_df.empty
                else 0
            )
        return players

    def load_match_data(self, match_num: int) -> Optional[pd.DataFrame]:
        """Main data processing pipeline"""
        try:
            # 1. Load base player data
            players = self._load_with_fallback(
                self.data_dirs["features"] / "player_performance_with_features.csv",
                required_cols=[
                    "player_name",
                    "Team",
                    "batting_ipl_runs",
                    "bowling_ipl_wkts",
                    "batting_ipl_4s",
                    "batting_ipl_6s",
                    "batting_ipl_avg",
                    "bowling_ipl_econ",
                ],
            )
            if players.empty:
                raise ValueError("No player data loaded")

            # 2. Load lineup data for the match
            lineup_path = Path(f"Data/lineups/Match_{match_num}_Lineup.csv")
            lineup = self._load_with_fallback(
                lineup_path, required_cols=["Credits", "Player Name", "Team"]
            )
            if lineup.empty:
                raise ValueError(f"Lineup data missing for match {match_num}")

            # 3. Standardize columns
            players["player"] = players["player_name"].str.strip().str.lower()
            players["team"] = players["Team"].str.strip().str.upper()
            lineup["player"] = lineup["Player Name"].str.strip().str.lower()
            lineup["team"] = lineup["Team"].str.strip().str.upper()

            # Log the columns of the lineup data before merging
            logging.info(f"Lineup data columns: {lineup.columns.tolist()}")

            # Ensure 'IsPlaying' column exists in the lineup data
            if "IsPlaying" not in lineup.columns:
                raise ValueError(
                    "The 'IsPlaying' column is missing from the lineup data."
                )

            # Merge lineup data with player data, ensuring 'IsPlaying' is included
            players = players.merge(
                lineup[["player", "Credits", "IsPlaying"]], on="player", how="left"
            )

            # Log the columns of the player data after merging
            logging.info(
                f"Player data columns after merging: {players.columns.tolist()}"
            )

            # Ensure 'IsPlaying' column is filled with default value if missing
            players["IsPlaying"] = players["IsPlaying"].fillna("NOT_PLAYING")

            # 5. Add matchup flags
            players = self._process_matchups(players)

            # 6. Calculate fantasy points
            players["fantasy_points"] = (
                players["batting_ipl_runs"] * 1
                + players["batting_ipl_4s"] * 0.5
                + players["batting_ipl_6s"] * 1
                + players["bowling_ipl_wkts"] * 25
            )

            # 7. Calculate boundary ratio = (4s + 6s) / runs (avoid division by 0)
            players["boundary_ratio"] = (
                players["batting_ipl_4s"] + players["batting_ipl_6s"]
            ) / players["batting_ipl_runs"].replace(0, 1)

            # 8. Add partnership_avg from CSV
            partnerships_df = self._load_with_fallback(
                self.data_dirs["partnerships"],
                required_cols=["player1", "player2", "runs"],
            )
            if not partnerships_df.empty:
                partnerships_df["player1"] = (
                    partnerships_df["player1"].str.strip().str.lower()
                )
                partnerships_df["player2"] = (
                    partnerships_df["player2"].str.strip().str.lower()
                )

                def calc_partnership_avg(player):
                    mask = (partnerships_df["player1"] == player) | (
                        partnerships_df["player2"] == player
                    )
                    return (
                        partnerships_df[mask]["runs"].mean()
                        if not partnerships_df[mask].empty
                        else 0
                    )

                players["partnership_avg"] = players["player"].apply(
                    calc_partnership_avg
                )
            else:
                players["partnership_avg"] = 0

            # 9. Calculate recent_form (rolling avg over fantasy_points)
            players["recent_form"] = (
                players["fantasy_points"].rolling(window=3, min_periods=1).mean()
            )

            # 10. Final cleanup
            players = players.fillna(0)

            # 11. Save merged data
            merged_path = self.data_dirs["merged"] / f"merged_match_{match_num}.csv"
            players.to_csv(merged_path, index=False)

            return players

        except Exception as e:
            logging.error(f"Failed to process match {match_num}: {str(e)}")
            return None


if __name__ == "__main__":
    processor = DataProcessor()
    match_num = int(input("Enter match number to process: "))
    result = processor.load_match_data(match_num)

    if result is not None:
        print(f"✅ Successfully processed {len(result)} players")
        print(f"Saved to: Data/merged/merged_match_{match_num}.csv")
    else:
        print("❌ Processing failed - check data_processing.log")
