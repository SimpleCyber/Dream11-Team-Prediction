# cleaner.py - FINAL VERSION
import pandas as pd
import numpy as np
from pathlib import Path
import os

def setup_paths():
    base_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    data_dir = base_dir / "Data"
    clean_dir = data_dir / "clean_data"
    os.makedirs(clean_dir, exist_ok=True)
    return data_dir, clean_dir

def clean_file(input_path, output_path, cleaning_func):
    try:
        if not input_path.exists():
            print(f"⚠️ File not found: {input_path}")
            return False

        df = pd.read_csv(input_path)
        print(f"\nProcessing: {input_path.name}")
        print("Original columns:", df.columns.tolist())

        df = cleaning_func(df)

        df.columns = [str(col).strip().replace(" ", "_").lower() for col in df.columns]

        df.to_csv(output_path, index=False)
        print(f"✅ Saved to: {output_path}")
        print("Sample data:")
        print(df.head(2))
        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def clean_batting_stats(data_dir, clean_dir):
    batting_dir = data_dir / "ipl_2024_25_stats" / "batting_stats"

    def clean_most_runs(df):
        required = ['Runs', '4s', '6s', 'Avg', 'SR']
        if not all(col in df.columns for col in required):
            missing = [col for col in required if col not in df.columns]
            raise ValueError(f"Missing columns: {missing}")
        return df.assign(
            boundary_pct=(df['4s'] + df['6s']) / df['Runs'],
            consistency=df['Avg'] / df['SR']
        )

    stats_files = {
        "Most_Runs.csv": clean_most_runs,
        "Highest_Scores.csv": lambda df: df.rename(columns={'Vs': 'opponent'}),
    }

    for file_name, clean_func in stats_files.items():
        input_path = batting_dir / file_name
        output_path = clean_dir / f"cleaned_{file_name}"
        clean_file(input_path, output_path, clean_func)

def clean_bowling_stats(data_dir, clean_dir):
    bowling_dir = data_dir / "ipl_2024_25_stats" / "bowlling_stats"

    stats_files = {
        "Most_Wickets.csv": lambda df: df.assign(
            wickets_per_match=df['Wkts'] / df['Matches'],
            economy_rate=df['Runs'] / df['Overs']
        ),
        "Best_Bowling.csv": lambda df: df.assign(
            bbi=df['BBI'].str.replace("May-", "5/").str.replace("Apr-", "4/")
        ),
    }

    for file_name, clean_func in stats_files.items():
        input_path = bowling_dir / file_name
        output_path = clean_dir / f"cleaned_{file_name}"
        clean_file(input_path, output_path, clean_func)

def clean_other_files(data_dir, clean_dir):
    other_files = {
        "historical_pitch_data.csv": {
            "path": data_dir / "historical_pitch_data.csv",
            "clean_func": lambda df: df.assign(
                pitch_type=np.where(df['spin_wickets'] > df['pace_wickets'], 'spin', 'pace')
            )
        },
        "player_performance.csv": {
            "path": data_dir / "t20_ipl_data" / "player_perfomace.csv",
            "clean_func": lambda df: df.fillna(0)
        },
        "venue_details.csv": {
            "path": data_dir / "venue_match" / "venu_match_detail.csv",
            "clean_func": lambda df: df.rename(columns={
                '1st_inn_avg_runs': 'first_inn_avg_runs',
                '2nd_inn_avg_runs': 'second_inn_avg_runs'
            })
        }
    }

    for file_name, config in other_files.items():
        input_path = config["path"]
        output_path = clean_dir / f"cleaned_{file_name}"
        clean_file(input_path, output_path, config["clean_func"])

def main():
    print("Starting data cleaning process...")
    data_dir, clean_dir = setup_paths()

    print("\n=== Batting Stats ===")
    clean_batting_stats(data_dir, clean_dir)

    print("\n=== Bowling Stats ===")
    clean_bowling_stats(data_dir, clean_dir)

    print("\n=== Other Files ===")
    clean_other_files(data_dir, clean_dir)

    print("\nCleaning complete! Check the 'clean_data' directory.")

if __name__ == "__main__":
    main()
