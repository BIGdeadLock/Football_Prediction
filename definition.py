"""
The file is used to aggregate all the constants in one place.
The constants will be added under a variable called token.
Once a token is changed, it will affect the entre modules that used it.
"""

# ----------------- Match Table Tokens --------------------
TOKEN_MATCH_HOME_TEAM_ID = 'home_team_api_id'
TOKEN_MATCH_AWAY_TEAM_ID = 'away_team_api_id'
TOKEN_MATCH_HOME_PLAYERS_ID = [f"home_player_{i}" for i in range(1,12)]
TOKEN_MATCH_AWAY_PLAYERS_ID = [f"away_player_{i}" for i in range(1,12)]

