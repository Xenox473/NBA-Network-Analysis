# NBA-Network-Analysis
Network analysis of NBA Matchups for the 2021-2022 regular season

This repo contains all the accompanying files for my Network Science project on NBA matchups. 

File Descriptions:
1. NBA_network_analysis.ipynb: Contains all of the code used for the research. This includes mining the data, determining the matchup winners, and creating the network.
2. nba_script.py: A script version of the most important code from the notebook. Can be used to run the code in one go. Contains two primary methods:
   1. construct_graph: Performs all the neceassary operations to create the network for a given season. Performs the following operations:
      1. mine_data: Mines all the necessary data for the given season
      2. calculate_matchup_winners: Calculates winners of all the matchups
      3. create_graph: Creates the graph
   2. playoff_prediction: For a given bracket, outputs a predicted champion. See code for usage example
3. datasets/: Contains all the datasets mined and generated during script execution and stored as a pickled file.
4. Graphs/: Contains all the created networks in .gexf format for use in softwares like Gephi. Also contains .png images of the graphs created using Gephi.
