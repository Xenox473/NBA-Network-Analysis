from nba_api.stats.endpoints import commonallplayers, leagueseasonmatchups, commonteamroster

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle as pk
from tqdm import tqdm
import time

def checkpoint(filename, data):
    # Save data to a pickle file
    filename = "datasets/" + filename + ".pkl"
    with open(filename, 'wb') as f:
        pk.dump(data, f)

def load_checkpoint(filename):
    # Load data from a pickle file
    filename = "datasets/" + filename + ".pkl"
    with open(filename, 'rb') as f:
        return pk.load(f)

def mine_data(season, start_year, end_year):
    all_players = commonallplayers.CommonAllPlayers(season=season).get_data_frames()[0]
    players = all_players[(all_players['TO_YEAR'] >= start_year) & (all_players['FROM_YEAR'] <= end_year)]
    players = players[['PERSON_ID', 'DISPLAY_FIRST_LAST', 'TEAM_ID', 'TEAM_ABBREVIATION']]

    checkpoint(f"players_{season}", players)

    print("Getting matchups...")
    matchups = leagueseasonmatchups.LeagueSeasonMatchups(season=season).get_data_frames()[0]
    matchups = matchups[[
        'OFF_PLAYER_ID',
        'DEF_PLAYER_ID',
        'GP',
        'MATCHUP_MIN',
        'PLAYER_PTS',
        'TEAM_PTS',
        'MATCHUP_AST',
        'MATCHUP_TOV',
        'MATCHUP_BLK',
        'MATCHUP_FGM',
        'MATCHUP_FGA',
        'MATCHUP_FG_PCT',
        'MATCHUP_FG3M',
        'MATCHUP_FG3A',
        'MATCHUP_FG3_PCT',
        'MATCHUP_TIME_SEC',
        'SFL'
    ]]

    checkpoint(f"matchups_{season}", matchups)

    print("Getting rosters...")
    rosters = pd.DataFrame()
    teams = tqdm(players['TEAM_ID'].unique())
    for team in teams:
        if team == 0:
            continue
        roster = commonteamroster.CommonTeamRoster(team_id=team, season=season).get_data_frames()[0]
        rosters = rosters.append(roster)
        time.sleep(1)

    checkpoint(f"rosters_{season}", rosters)

def determine_matchup_winner(matchup1, matchup2):
    player_1 = matchup1['OFF_PLAYER_ID'].values[0]
    player_2 = matchup2['OFF_PLAYER_ID'].values[0]

    score = 0

    # Determine the winner of the matchup based on the following criteria
    # 1. Points -- Offensive player with the most points wins
    # 2. Assists -- Offensive player with the most assists wins
    # 3. Blocks -- Defensive player with the most blocks wins
    # 4. Turnovers -- Offensive player with the least turnovers wins
    # 5. Field goals made -- Offensive player with the most field goals made wins
    # 6. 3-point field goals made -- Offensive player with the most 3-point field goals made wins
    # 7. Shooting fouls -- Offensive player with the most shooting fouls wins
    # All stats are normalized by the number of seconds played in the matchup

    criteria = [
        'MATCHUP_AST',
        'MATCHUP_BLK',
        'MATCHUP_TOV',
        'MATCHUP_FGM',
        'MATCHUP_FG3M',
        'SFL'
    ]

    for c in criteria:
        switch = 1
        if c in ['MATCHUP_TOV', 'MATCHUP_BLK']:
            switch = -1

        player_1_stats = matchup1[c].values[0] / matchup1['MATCHUP_TIME_SEC'].values[0]
        player_2_stats = matchup2[c].values[0] / matchup2['MATCHUP_TIME_SEC'].values[0]

        total_stats = player_1_stats + player_2_stats

        # print(c, player_1_stats, player_2_stats)
        if total_stats == 0:
            continue

        differential = (switch * player_1_stats - switch * player_2_stats)
        score += differential

    # Determine the winner of the matchup
    if score > 0:
        winner = player_1
    elif score < 0:
        winner = player_2
    else:
        winner = None

    total_matchup_time = matchup1['MATCHUP_TIME_SEC'].values[0] + matchup2['MATCHUP_TIME_SEC'].values[0]
    return winner, score, total_matchup_time

def player_id_to_name(player_id, players):
    return players[players['PERSON_ID'] == player_id]['DISPLAY_FIRST_LAST'].values[0]

def calculate_matchup_winners(season):
    matchups = load_checkpoint(f"matchups_{season}")
    players = load_checkpoint(f"players_{season}")

    matchups = matchups[matchups['MATCHUP_TIME_SEC'] > 60]

    matchup_winners =  pd.DataFrame(columns=['Source', 'Target', 'Score ratio', 'Matchup time (sec)'])
    total_matchups = len(matchups)

    for index, row in matchups.iterrows():
        player_1 = row['OFF_PLAYER_ID']
        player_2 = row['DEF_PLAYER_ID']
        player_1_name = player_id_to_name(player_1, players)
        player_2_name = player_id_to_name(player_2, players)

        print(f"{index}/{total_matchups} - {player_1_name} vs. {player_2_name}")

        if len(list(filter(lambda x: set([x[0], x[1]]) == set([player_1_name, player_2_name]), matchup_winners.values))):
            continue
        
        matchup_1 = matchups[(matchups['OFF_PLAYER_ID'] == player_1) & (matchups['DEF_PLAYER_ID'] == player_2)]
        matchup_2 = matchups[(matchups['OFF_PLAYER_ID'] == player_2) & (matchups['DEF_PLAYER_ID'] == player_1)]

        if matchup_2.empty:
            continue

        winner, score, matchup_time = determine_matchup_winner(matchup_1, matchup_2)

        if winner is None:
            continue

        loser = player_1 if winner == player_2 else player_2
        edge_weight = np.abs(score)

        matchup_winners.loc[len(matchup_winners)] = [player_id_to_name(loser, players), player_id_to_name(winner, players), edge_weight, matchup_time]

    checkpoint(f"matchup_winners_{season}", matchup_winners)

def create_graph(season):
    matchup_winners = load_checkpoint(f"matchup_winners_{season}")
    players = load_checkpoint(f"players_{season}")
    rosters = load_checkpoint(f"rosters_{season}")

    # Normalize the score ratio and matchup time
    matchup_winners['Score Normalized'] = matchup_winners['Score ratio'].transform(lambda x: x / x.max())
    matchup_winners['Matchup time normalized'] = matchup_winners['Matchup time (sec)'].transform(lambda x: x / x.max())
    matchup_winners['Matchup time normalized by player'] = matchup_winners.groupby('Target')['Matchup time (sec)'].transform(lambda x: x / x.max())
    matchup_winners['Weighted Score'] = matchup_winners['Score Normalized'] * matchup_winners['Matchup time normalized by player']
    # Normalize Weighted Score
    matchup_winners['Weighted Score'] = matchup_winners['Weighted Score'].transform(lambda x: x / x.max())

    print("Creating graph...")
    matchup_network = nx.from_pandas_edgelist(matchup_winners, 'Source', 'Target', ['Weighted Score', 'Matchup time normalized', 'Matchup time normalized by player'], create_using=nx.DiGraph())

    nodes = tqdm(matchup_network.nodes())
    for node in nodes:
        matchup_network.nodes[node]['name'] = node
        matchup_network.nodes[node]['team'] = players[players['DISPLAY_FIRST_LAST'] == node]['TEAM_ABBREVIATION'].values[0]
        position = rosters[rosters['PLAYER'] == node]['POSITION'].values
        if len(position) > 0:
            matchup_network.nodes[node]['position'] = position[0].split('-')[0]

    checkpoint(f"matchup_network_{season}", matchup_network)
    nx.write_gexf(matchup_network, f"Graphs/matchup_network_{season}.gexf")

def predict_winner(matchup_network, team_1, team_2):
    team_1_players = []
    team_2_players = []

    for node in matchup_network.nodes():
        if matchup_network.nodes[node]['team'] == team_1:
            team_1_players.append(node)
        elif matchup_network.nodes[node]['team'] == team_2:
            team_2_players.append(node)

    team_1_score = 0
    team_2_score = 0

    for player_1 in team_1_players:
        for player_2 in team_2_players:
            if matchup_network.has_edge(player_2, player_1):
                team_1_score += matchup_network.edges[player_2, player_1]['Weighted Score'] 
            elif matchup_network.has_edge(player_1, player_2):
                team_2_score += matchup_network.edges[player_1, player_2]['Weighted Score'] 

    total_score = team_1_score + team_2_score

    if team_1_score > team_2_score:
        return team_1, round(team_1_score / total_score * 100, 2), team_2, round(team_2_score / total_score * 100, 2)
    elif team_2_score > team_1_score:
        return team_2, round(team_2_score / total_score * 100, 2), team_1, round(team_1_score / total_score * 100, 2)
    else:
        return "TIE", round(team_1_score / total_score * 100, 2), "TIE", round(team_2_score / total_score * 100, 2)
    
def playoff_prediction(season):
    matchup_network = load_checkpoint(f"matchup_network_{season}")
    bracket = [
        'ATL',
        'MIA',
        'TOR',
        'PHI',
        'CHI',
        'MIL',
        'BKN',
        'BOS',
    # Western Conference Matchups
        'PHX',
        'NOP',
        'UTA',
        'DAL',
        'DEN',
        'GSW',
        'MEM',
        'MIN',
    ]

    while len(bracket) > 1:
        team_1, team_1_score, team_2, team_2_score = predict_winner(matchup_network, bracket[0], bracket[1])
        print(f'{team_1} ({team_1_score}%) vs {team_2} ({team_2_score}%)')
        matchup_winner = team_1 if team_1_score > team_2_score else team_2
        bracket.append(matchup_winner)
        print("Matchup Winner: ", matchup_winner)
        bracket.pop(0)
        bracket.pop(0)

    print("Champion: ", bracket[0])

def construct_graph():
    season = "2022-23"
    start_year = "2022"
    end_year = "2023"
    mine_data(season, start_year, end_year)
    calculate_matchup_winners(season)
    create_graph(season)

if __name__ == "__main__":
    construct_graph()
    # playoff_prediction("2021-22")