"""
The file is used to aggregate all the constants in one place.
The constants will be added under a variable called token.
Once a token is changed, it will affect the entre modules that used it.
"""

TOKEN_CLASS_NAME = 'win'

# ----------------- Match Table Tokens --------------------
TOKEN_MATCH_HOME_TEAM_ID = 'home_team_api_id'
TOKEN_MATCH_AWAY_TEAM_ID = 'away_team_api_id'
TOKEN_MATCH_ID = 'id'
TOKEN_HOME_TEAM = 'home'
TOKEN_HOME_TEAM_NAME = 'home_team'
TOKEN_AWAY_TEAM = 'away'
TOKEN_AWAY_TEAM_NAME = 'away_team'
TOKEN_MATCH_SEASON = 'season'
TOKEN_MATCH_SHOTON = 'shoton'
TOKEN_MATCH_DATE = 'date'
TOKEN_MATCH_CROSS = 'cross'
TOKEN_MATCH_CORNERS= 'cross'
TOKEN_MATCH_POSS = 'possession'
TOKEN_MATCH_CARD= 'card'
TOKEN_MATCH_AWAY_TEAM_SHOTON = 'on_target_shot_away_team'
TOKEN_MATCH_HOME_TEAM_SHOTON = 'on_target_shot_home_team'
TOKEN_MATCH_HOME_TEAM_REDCARD = 'red_card_home_team'
TOKEN_MATCH_AWAY_TEAM_REDCARD = 'red_card_away_team'
TOKEN_MATCH_AWAY_TEAM_YELLOWCARD = 'yellow_card_away_team'
TOKEN_MATCH_HOME_TEAM_YELLOWCARD = 'yellow_card_home_team'
TOKEN_MATCH_HOME_TEAM_CROSSES = 'crosses_home_team'
TOKEN_MATCH_AWAY_TEAM_CROSSES = 'crosses_away_team'
TOKEN_MATCH_HOME_TEAM_CORNERS = 'corner_home_team'
TOKEN_MATCH_AWAY_TEAM_CORNERS = 'corner_away_team'
TOKEN_MATCH_HOME_TEAM_POSS = 'possession_home_team'
TOKEN_MATCH_AWAY_TEAM_POSS = 'possession_away_team'
TOKEN_MATCH_HOME_PLAYERS_ID = [f"home_player_{i}" for i in range(1,12)]
TOKEN_MATCH_AWAY_PLAYERS_ID = [f"away_player_{i}" for i in range(1,12)]
TOKEN_MATCH_AWAY_PLAYERS_X_POS = [f"away_player_X{i}" for i in range(1,12)]
TOKEN_MATCH_HOME_PLAYERS_X_POS = [f"home_player_X{i}" for i in range(1,12)]
TOKEN_MATCH_HOME_PLAYERS_Y_POS = [f"home_player_Y{i}" for i in range(1,12)]
TOKEN_MATCH_AWAY_PLAYERS_Y_POS = [f"away_player_Y{i}" for i in range(1,12)]
TOKEN_MATCH_GOALS = ["home_team_goal", "away_team_goal"]
TOKEN_MATCH_HOME_TEAM_GOAL = 'home_team_goal'
TOKEN_MATCH_AWAY_TEAM_GOAL = 'away_team_goal'


# -------------- Datasets Table Tokens --------------
TOKEN_DS_HOME_TEAM_ID = "HomeTeamAPI"
TOKEN_DS_AWAY_TEAM_ID = "AwayTeamAPI"
TOKEN_DS_AWAY_TEAM_NAME = "AwayTeam"
TOKEN_DS_HOME_TEAM_NAME = "HomeTeam"
TOKEN_DS_HOME_TEAM_ODDS = "HomeTeamsOdds"
TOKEN_DS_AWAY_TEAM_ODDS = "AwayTeamOdds"
TOKEN_DS_AWAY_TEAM_GOALS = "AwayTeamGoals"
TOKEN_DS_HOME_TEAM_GOALS = "HomeTeamGoals"
TOKEN_DS_HOME_TEAM_Rating = "HomeTeamRatings"
TOKEN_DS_AWAY_TEAM_Rating = "AwayTeamRatings"
TOKEN_DS_GOALDIFF = "GoalDiff"
TOKEN_DS_HOME_TEAM_AVG_GOALS = "HomeTeamAvgGoals"
TOKEN_DS_AWAY_TEAM_AVG_GOALS = "AwayTeamAvgGoals"
TOKEN_HOME_TEAM_SPEED = "HomeTeamPlaySpeed"
TOKEN_AWAY_TEAM_SPEED = "AwayTeamPlaySpeed"
TOKEN_HOME_TEAM_DEF = "HomeTeamDefencePressure"
TOKEN_AWAY_TEAM_DEF = "AwayTeamDefencePressure"
TOKEN_HOME_TEAM_SHOOT = "HomeTeamCreatonShooting"
TOKEN_AWAY_TEAM_SHOOT = "AWAYTeamCreatonShooting"

# -------------- DataFrame TOKENS --------------------
TOKEN_INDEX_AXIS = 'index'
TOKEN_LEFT_JOIN = 'left'
TOKEN_INNER_JOIN = 'inner'


# -------------- Player_Attributes Table Tokens --------------
TOKEN_PLAYER_ATTRIB_OVERALL = 'overall_rating'
TOKEN_PLAYER_ID = 'player_api_id'

# -------------- Team_Attributes Table Tokens ---------------
TOKEN_TEAM_ATTR_ID = 'team_api_id'
TOKEN_TEAM_SPEED = 'buildUpPlaySpeed'
TOKEN_TEAM_DEF_PRESS = 'defencePressure'
TOKEN_TEAM_CHANES = 'chanceCreationShooting'