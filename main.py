# main.py
from model_trainer import ModelTrainer
from data_processor import DataProcessor
from team_optimizer import TeamOptimizer
import logging
import sys
import pandas as pd


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("dream11_optimizer.log"),
            logging.StreamHandler(),
        ],
    )


def run_pipeline(match_num: int):
    """Complete workflow from data to team selection"""
    try:
        # 1. Data Processing
        logging.info("Processing data...")
        processor = DataProcessor()
        data = processor.load_match_data(match_num)
        if data is None:
            raise ValueError("Data processing failed")

        # Load lineup data for the match
        lineup_file = f"Data/lineups/Match_{match_num}_Lineup.csv"
        try:
            lineup_data = pd.read_csv(lineup_file)
            logging.info(f"Loaded lineup data from {lineup_file}")

            # Standardize column names to lowercase and replace spaces with underscores
            lineup_data.columns = lineup_data.columns.str.lower().str.replace(" ", "_")
        except FileNotFoundError:
            raise ValueError(f"Lineup file not found: {lineup_file}")

        # Use only the 'IsPlaying' column from the lineup data
        data = data.merge(
            lineup_data[["player_name", "isplaying"]], on="player_name", how="left"
        )

        # Rename 'isplaying' to 'IsPlaying' for consistency
        data.rename(columns={"isplaying": "IsPlaying"}, inplace=True)

        # Fill missing values
        data["IsPlaying"] = data["IsPlaying"].fillna("NOT_PLAYING")


        # Log the updated DataFrame
        logging.info(
            f"Updated data columns after merging lineup: {data.columns.tolist()}"
        )

        # 2. Model Training
        logging.info("Training model...")
        trainer = ModelTrainer()
        results = trainer.train(data)
        if "error" in results:
            raise ValueError(results["error"])

        # 3. Team Optimization
        logging.info("Optimizing team...")
        data["predicted_points"] = trainer.predict(data)

        # Log the columns of the DataFrame before optimization
        
        data = data.loc[:, ~data.columns.duplicated()]
        logging.info("Data columns before optimization: %s", data.columns.tolist())



        # Check if 'IsPlaying' column exists
        if "IsPlaying" not in data.columns:
            raise ValueError("The 'IsPlaying' column is missing from the input data.")

        # Log a sample of the 'IsPlaying' column
        logging.info(
            f"Sample 'IsPlaying' values: {data['IsPlaying'].value_counts().to_dict()}"
        )

        # Extract teams from data
        teams = data["Team"].unique().tolist()



        # Pass teams to TeamOptimizer
        optimizer = TeamOptimizer(teams)
        team = optimizer.optimize(data)

        if team is None or not team:
            logging.warning("Optimizer failed. Using top 11 PLAYING players by predicted points.")
            fallback_team = (
                data[data["IsPlaying"] == "PLAYING"]
                .sort_values("predicted_points", ascending=False)
                .head(11)
            )

            team = [
                {
                    "player": row["player_name"],
                    "team": row["Team"],
                    "predicted_points": row["predicted_points"],
                }
                for _, row in fallback_team.iterrows()
            ]


        if team is None:
            logging.error("Team optimization failed")
            print("❌ Error: Team optimization failed")
        else:
            print("\n🏆 Optimal Dream11 Team:")
            for i, player in enumerate(
                sorted(team, key=lambda x: x["predicted_points"], reverse=True), 1
            ):
                print(
                    f"{i:2}. {player['player'][:20]:20} {player['team']:4} {player['predicted_points']:5.1f} pts"
                )


        if not team:
            raise ValueError("Team optimization failed")
        

        print("\n🏆 Fallback Dream11 Team:")
        for i, player in enumerate(
            sorted(team, key=lambda x: x["predicted_points"], reverse=True), 1
        ):
            print(
                f"{i:2}. {player['player'][:20]:20} {player['team']:4} {player['predicted_points']:5.1f} pts"
            )


        # 4. Display Results
        print("\n🏆 Optimal Dream11 Team:")
        for i, player in enumerate(
            sorted(team, key=lambda x: x["predicted_points"], reverse=True), 1
        ):
            print(
                f"{i:2}. {player['player'][:20]:20} {player['team']:4} {player['predicted_points']:5.1f} pts"
            )

    except KeyError as e:
        logging.error(f"Missing key in data: {str(e)}")
        print(f"❌ Error: Missing key in data - {str(e)}")
    except ValueError as e:
        logging.error(f"Value error: {str(e)}")
        print(f"❌ Error: {str(e)}")
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    configure_logging()
    match_num = (
        int(sys.argv[1]) if len(sys.argv) > 1 else int(input("Enter match number: "))
    )
    run_pipeline(match_num)
