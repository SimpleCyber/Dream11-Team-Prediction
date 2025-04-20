import pulp
import pandas as pd
import logging
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("optimization.log"), logging.StreamHandler()],
)


class TeamOptimizer:
    def __init__(self, teams: list):
        self.teams = teams
        self.credit_limit = 100.0
        self.role_constraints = {
            "WK": (1, 4),
            "BAT": (3, 6),
            "BOWL": (3, 6),
        }
        self.min_players_per_team = 1
        self.max_players_per_team = 8
        self.team_size = 11

    def _validate_input(self, players: pd.DataFrame) -> bool:
        try:
            # Log the columns of the DataFrame
            logging.info(f"Input DataFrame columns: {players.columns.tolist()}")

            # Check if 'IsPlaying' column exists
            if "IsPlaying" not in players.columns:
                raise ValueError("Missing 'IsPlaying' column in the input data.")

            # Log a sample of the 'IsPlaying' column
            logging.info(
                f"Sample 'IsPlaying' values: {players['IsPlaying'].value_counts().to_dict()}"
            )

            # Check for foreign teams
            foreign_teams = set(players["Team"].unique()) - set(self.teams)
            if foreign_teams:
                raise ValueError(f"Invalid teams detected: {foreign_teams}")

            # Check role coverage
            for role in self.role_constraints:
                if role not in players["Player Type"].unique():
                    logging.warning(f"No players available for role: {role}")

            return True
        except Exception as e:
            logging.error(f"Validation failed: {str(e)}")
            return False

    def optimize(self, players: pd.DataFrame) -> Optional[List[Dict]]:
        try:
            # Filter players to include only those marked as 'PLAYING'
            players = players[players["IsPlaying"] == "PLAYING"]

            if players.empty:
                raise ValueError(
                    "No players available for optimization after filtering for 'PLAYING'."
                )

            if not self._validate_input(players):
                return None

            prob = pulp.LpProblem("Dream11_Optimization", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts("Player", players.index, cat="Binary")

            # Objective: Maximize points
            prob += pulp.lpSum(
                players.loc[i, "predicted_points"] * player_vars[i]
                for i in players.index
            )

            # Credit constraint
            prob += (
                pulp.lpSum(
                    players.loc[i, "Credits"] * player_vars[i] for i in players.index
                )
                <= self.credit_limit
            )

            # Team composition constraints
            for team in self.teams:
                team_players = players[players["Team"] == team].index
                prob += (
                    pulp.lpSum(player_vars[i] for i in team_players)
                    >= self.min_players_per_team
                )
                prob += (
                    pulp.lpSum(player_vars[i] for i in team_players)
                    <= self.max_players_per_team
                )

            # Role constraints
            for role, (min_p, max_p) in self.role_constraints.items():
                role_players = players[players["Player Type"] == role].index
                if not role_players.empty:
                    prob += pulp.lpSum(player_vars[i] for i in role_players) >= min_p
                    prob += pulp.lpSum(player_vars[i] for i in role_players) <= max_p

            # Total players constraint
            prob += pulp.lpSum(player_vars[i] for i in players.index) == 11

            # Solve problem
            prob.solve(pulp.PULP_CBC_CMD(msg=False))

            if pulp.LpStatus[prob.status] != "Optimal":
                raise ValueError(
                    f"No optimal solution found: {pulp.LpStatus[prob.status]}"
                )

            # Extract solution
            selected = [i for i in players.index if player_vars[i].value() == 1]
            team_df = players.loc[selected].sort_values(
                "predicted_points", ascending=False
            )

            # Assign Captain and Vice-Captain
            team_df = team_df.reset_index(drop=True)
            team_df["C/VC"] = "NA"
            if len(team_df) >= 1:
                team_df.at[0, "C/VC"] = "C"
            if len(team_df) >= 2:
                team_df.at[1, "C/VC"] = "VC"

            # Return the optimized team as a list of dictionaries
            return team_df.to_dict("records")

        except Exception as e:
            logging.error(f"Optimization failed: {str(e)}")
            return None
