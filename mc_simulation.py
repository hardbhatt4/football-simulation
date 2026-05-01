import pandas as pd
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt

# Constants/Assumptions
RANDOM_SEED = 42
ELO_K_FACTOR = 20
BASE_DRAW_PROB = 0.3
DRAW_DECAY = 0.003
NUM_SIMULATIONS = 1000

# Import all data as dataframes
df_elo = pd.read_csv("Club Elo.csv", index_col="Club")
club_elo = df_elo.to_dict()["Elo"]
original_elo = club_elo.copy()
matches = pd.read_csv("Matches.csv", parse_dates=["Date"], index_col="Date")
table = pd.read_csv("PL Table.csv", index_col="Squad")

simulation_results = []
random.seed(RANDOM_SEED)


def expected_score(r_a, r_b):
    """Calculate expected score based on elo rating"""
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def update_elo(home_team, away_team, result):
    """Update team elo based on result"""
    r_home = club_elo[home_team]
    r_away = club_elo[away_team]

    # Expected scores
    e_home = expected_score(r_home, r_away)
    e_away = expected_score(r_away, r_home)

    # Actual scores
    if result == "H":
        s_home, s_away = 1, 0
    elif result == "A":
        s_home, s_away = 0, 1
    else:
        s_home, s_away = 0.5, 0.5

    # Update ratings
    club_elo[home_team] += ELO_K_FACTOR * (s_home - e_home)
    club_elo[away_team] += ELO_K_FACTOR * (s_away - e_away)


def simulate_match(home_team, away_team):
    """Simulate a single match and return the result (H for home team win, D for draw and A for away team win)"""

    diff = club_elo[home_team] - club_elo[away_team]

    # Expected score
    expected_match_score = 1 / (1 + 10 ** (-diff / 400))

    # Draw probability
    draw_prob = BASE_DRAW_PROB * math.exp(-DRAW_DECAY * abs(diff))

    # Win probability
    win_prob = expected_match_score - 0.5 * draw_prob

    r = random.random()

    if r < win_prob:
        result = "H"
    elif r < win_prob + draw_prob:
        result = "D"
    else:
        result = "A"

    return result


def run_simulations(num_simulation):
    """Run specified number of simulation and obtain final points table by simulating matches"""
    for i in range(num_simulation):
        club_elo.update(original_elo)
        points_table = table.to_dict()["Pts"]

        # Simulate each match
        for match in matches.itertuples():
            home_team = match.Home
            away_team = match.Away
            result = simulate_match(home_team, away_team)
            if result == "H":
                points_table[home_team] += 3
            elif result == "A":
                points_table[away_team] += 3
            else:
                points_table[home_team] += 1
                points_table[away_team] += 1
            update_elo(home_team, away_team, result)
        simulation_results.append(points_table)


run_simulations(NUM_SIMULATIONS)

# Prepare dataframe plot_df with each team's position for each simulation
dfs = []
for final_table in simulation_results:
    df = pd.DataFrame.from_dict(final_table, orient="index", columns=["Points"])
    df = df.reset_index(names=["Team"])
    df = df.sort_values(by="Points", ascending=False, ignore_index=True)
    dfs.append(df)

records = []
for df in dfs:
    sim_result = {row.Team: row.Index + 1 for row in df.itertuples()}
    records.append(sim_result)

plot_df = pd.DataFrame(records)

# Calculate percentage of finishes for each position
counts = plot_df.apply(pd.Series.value_counts).fillna(0)

mc_percentages = counts.div(len(plot_df)) * 100
mc_percentages.index = mc_percentages.index.astype(int)

# Plot the heatmap
if __name__ == "__main__":
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        mc_percentages.transpose(), annot=True, fmt=".1f", cmap="Reds", linewidths=1
    )
    ax.xaxis.tick_top()
    plt.tight_layout()
    plt.title("Monte Carlo prediction of final position (n=1000)")
    plt.show()
