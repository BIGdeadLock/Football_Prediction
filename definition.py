"""
The file is used to aggregate all the constants in one place.
The constants will be added under a variable called token.
Once a token is changed, it will affect the entre modules that used it.
"""

# ----------------- Match Table Tokens --------------------
TOKEN_MATCH_HOME_TEAM_ID = 'home_team_api_id'
TOKEN_MATCH_AWAY_TEAM_ID = 'away_team_api_id'
TOKEN_MATCH_ID = 'id'
TOKEN_HOME_TEAM_NAME = 'home'
TOKEN_AWAY_TEAM_NAME = 'away'
TOKEN_MATCH_AWAY_TEAM_SHOTON = 'on_target_shot_away_team'
TOKEN_MATCH_HOME_TEAM_SHOTON = 'on_target_shot_home_team'
TOKEN_MATCH_HOME_TEAM_REDCARD = 'red_card_home_team'
TOKEN_MATCH_AWAY_TEAM_REDCARD = 'red_card_away_team'
TOKEN_MATCH_AWAY_TEAM_YELLOWCARD = 'yellow_card_away_team'
TOKEN_MATCH_HOME_TEAM_YELLOWCARD = 'yellow_card_home_team'
TOKEN_MATCH_HOME_TEAM_CROSSES = 'crosses_home_team'
TOKEN_MATCH_AWAY_TEAM_CROSSES = 'crooses_away_team'
TOKEN_MATCH_HOME_TEAM_CORNERS = 'corner_home_team'
TOKEN_MATCH_AWAY_TEAM_CORNERS = 'corner_away_team'
TOKEN_MATCH_AWAY_TEAM_POSS = 'possession_away_team'

TOKEN_MATCH_HOME_PLAYERS_ID = [f"home_player_{i}" for i in range(1,12)]
TOKEN_MATCH_AWAY_PLAYERS_ID = [f"away_player_{i}" for i in range(1,12)]
TOKEN_MATCH_AWAY_PLAYERS_X_POS = [f"away_player_X{i}" for i in range(1,12)]
TOKEN_MATCH_HOME_PLAYERS_X_POS = [f"home_player_X{i}" for i in range(1,12)]

# -------------- Datasets Table Tokens --------------
TOKEN_DS_HOME_TEAM_ID = "HomeTeamAPI"
TOKEN_DS_AWAY_TEAM_ID = "AwayTeamAPI"
TOKEN_DS_AWAY_TEAM_GOALS = "AwayTeamGoals"
TOKEN_DS_HOME_TEAM_GOALS = "HomeTeamGoals"
TOKEN_DS_GOALDIFF = "GoalDiff"

# -------------- DataFrame TOKENS --------------------
TOKEN_INDEX_AXIS = 'index'
TOKEN_LEFT_JOIN = 'left'
TOKEN_INNER_JOIN = 'inner'


