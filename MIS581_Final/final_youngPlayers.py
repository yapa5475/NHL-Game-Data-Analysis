#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# In[5]:


# load data

game_df = pd.read_csv('game.csv')
game_player_df = pd.read_csv('game_skater_stats.csv')
player_df = pd.read_csv('player_info.csv')
scratched_df = pd.read_csv('game_scratches.csv')


# In[8]:


# combine and clean data

# Merge game_player and player dfs
clean_df = game_player_df.merge(player_df[['player_id', 'firstName', 'lastName', 'primaryPosition', 'birthDate']],
              left_on='player_id', right_on='player_id')

# Merge game above with game df 
clean_df = clean_df.merge(game_df[['game_id', 'season', 'type']],
             left_on='game_id', right_on='game_id')

# Merge above with games scratched
clean_df = clean_df.merge(scratched_df, on=['game_id', 'player_id'], how='left', indicator=True)
clean_df['scratched'] = clean_df.pop('_merge').eq('both')

# Drop columns that are not needed
if 'team_id_x' in clean_df.columns:
    clean_df = clean_df.drop(['team_id_x'], axis=1)
# Remove goalies
clean_df = clean_df.loc[clean_df.primaryPosition != 'G']

# Filter for game type R (regular season)
clean_df = clean_df.loc[clean_df.type == 'R']

#Filter out games where player was scratched
clean_df = clean_df.loc[clean_df.scratched == False]
clean_df = clean_df.drop_duplicates()
# Create dictionary for aggregates
agg_dict_sum = {'assists': ['sum'], 'goals': ['sum'], 'shots': ['sum'], 'hits': ['sum'], 'powerPlayGoals': ['sum'], 'powerPlayAssists': ['sum'], 'shortHandedGoals': ['sum'],
     'shortHandedAssists': ['sum'], 'blocked': ['sum'], 'timeOnIce': ['sum'], 'powerPlayTimeOnIce': ['sum'], 'evenTimeOnIce': ['sum'], 'penaltyMinutes': ['sum'], 'player_id' : ['count']}

# Set flag for TOI conversion
toi_converted = 0

# Group data by player_id and season
grouped_df = clean_df.groupby(['player_id', 'firstName', 'lastName', 'primaryPosition', 'birthDate', 'season']).agg(agg_dict_sum).reset_index()

# Rename columns
grouped_df.columns = ['player_id','firstName', 'lastName', 'position', 'birthDate', 'season', 'assists', 'goals', 'shots', 'hits', 'powerPlayGoals',
                     'powerPlayAssists', 'shortHandedGoals', 'shortHandedAssists', 'blocks', 'timeOnIce', 'powerPlayTimeOnIce', 'evenTimeOnIce',
                     'penaltyMinutes', 'gamesPlayed']

# Convert TOI and PP TOI from seconds to minutes
if toi_converted == 0:
    # Convert toi
    grouped_df['timeOnIce'] = grouped_df['timeOnIce']/60
    grouped_df['powerPlayTimeOnIce'] = grouped_df['powerPlayTimeOnIce']/60
    grouped_df['evenTimeOnIce'] = grouped_df['evenTimeOnIce']/60
    
    # Add Shorthanded TOI column 
    grouped_df['shortHandedTimeOnIce'] = round(grouped_df['timeOnIce'] - grouped_df['evenTimeOnIce'] - grouped_df['powerPlayTimeOnIce'], 2)

    # Add per game toi columns
    grouped_df['timeOnIcePerGame'] = grouped_df['timeOnIce']/grouped_df['gamesPlayed']
    grouped_df['evenTimeOnIcePerGame'] = grouped_df['evenTimeOnIce']/grouped_df['gamesPlayed']
    grouped_df['powerPlayTimeOnIcePerGame'] = grouped_df['powerPlayTimeOnIce']/grouped_df['gamesPlayed']
    grouped_df['shortHandedTimeOnIcePerGame'] = grouped_df['shortHandedTimeOnIce']/grouped_df['gamesPlayed']    
    
    toi_converted = 1

# Add points column
grouped_df['points'] = grouped_df['assists'] + grouped_df['goals']

# Add PP points column
grouped_df['powerPlayPoints'] = grouped_df['powerPlayGoals'] + grouped_df['powerPlayAssists']

# Add SH points column
grouped_df['shortHandedPoints'] = grouped_df['shortHandedGoals'] + grouped_df['shortHandedAssists']

# Add Even Strength goals, assists, and points column
grouped_df['evenStrengthGoals'] = grouped_df['goals'] - grouped_df['powerPlayGoals'] - grouped_df['shortHandedGoals']
grouped_df['evenStrengthAssists'] = grouped_df['assists'] - grouped_df['powerPlayAssists'] - grouped_df['shortHandedAssists']
grouped_df['evenStrengthPoints'] = grouped_df['evenStrengthGoals'] + grouped_df['evenStrengthAssists']

# Sort by seasons and points
grouped_df = grouped_df.sort_values(['season', 'points'], 
              ascending = [False, False])

# Concatenate first and last name
if 'firstName' and 'lastName' in grouped_df.columns:
    grouped_df['name'] = grouped_df['firstName'] + ' ' + grouped_df['lastName']

# Drop unnecessary columns
if 'firstName' in grouped_df.columns:
    grouped_df = grouped_df.drop(['firstName'], axis=1)
    
if 'lastName' in grouped_df.columns:   
    grouped_df = grouped_df.drop(['lastName'], axis=1)
    
# Reorder column names
grouped_df = grouped_df.reindex(columns = ['player_id', 'name', 'birthDate', 'position', 'season','goals', 'assists', 'points', 'shots', 'hits', 'blocks',
                                           'powerPlayGoals', 'powerPlayAssists', 'powerPlayPoints', 'shortHandedGoals', 'shortHandedAssists', 'shortHandedPoints',
                                           'evenStrengthGoals', 'evenStrengthAssists', 'evenStrengthPoints','penaltyMinutes', 'timeOnIce', 'evenTimeOnIce',
                                           'powerPlayTimeOnIce', 'shortHandedTimeOnIce', 'timeOnIcePerGame', 'evenTimeOnIcePerGame', 'powerPlayTimeOnIcePerGame',
                                           'shortHandedTimeOnIcePerGame', 'gamesPlayed'])

# Output data
grouped_df.to_csv('skater_data_by_season.csv',index=False)


# In[9]:


# Copy data to work with
skater_all_seasons_df = grouped_df.copy()

# Display top 10 point-scoring skaters from 2019-2020 season
skater_all_seasons_df = skater_all_seasons_df.sort_values(['season', 'points'], ascending = [False, False])
skater_all_seasons_df.loc[skater_all_seasons_df.season == 20192020].head(10)


# In[10]:


# Method to calculate player's age at the start of a season
def get_player_age_in_season(season, birthDate):
    # Assume each season starts in October of that year
    season_year = int(str(season)[:4])
    season_start = datetime.datetime(season_year, 10, 1)
    
    # Format birthDate as date time
    birthDate_dt = datetime.datetime.strptime(birthDate, '%Y-%m-%d %H:%M:%S')
    
    # Calculate age in years at time season started
    return relativedelta(season_start, birthDate_dt).years


# In[12]:


# Global variable for 3 seasons to work with
sample_seasons = [20192020, 20182019, 20172018]

# Seasons to calculate avg toi
sample_seasons_toi = [20192020, 20182019, 20172018, 20162017, 20152016]

# Seasons to look at for age vs toi analysis
sample_seasons_aging = [20192020, 20182019, 20172018, 20162017, 20152016, 20142015, 20132014, 20122013, 20112012, 20102011]

# Global variable - value to iterate through seasons when performing linear regression, since season is stored as an int (i.e. season 2019-2020 is stored as integer: 20192020)
ITERATOR = abs(sample_seasons[0] - sample_seasons[1])

# Global variable - determine how many games to project upcoming season for
GAMES_TO_PLAY = 56

# Global variable - season start date
SEASON_START_DATE = datetime.datetime(2021, 1, 1)

# Global variable - coefficient labels
coefficient_labels = ['Season n-1 coefficient', 'Season n-2 coefficient', 'Season n-3 coefficient']


# In[16]:


all_skaters_2010_df = skater_all_seasons_df.loc[skater_all_seasons_df.season.isin(sample_seasons_aging)][['player_id', 'name', 'birthDate', 'position', 'season', 'goalsPer60', 'assistsPer60', 'shotsPer60', 'pointsPer60', 'timeOnIcePerGame', 'gamesPlayed']].copy()

toi_dictionary = {'timeOnIcePerGame': ['mean'], 'age' : ['count']}
all_skaters_2010_df['age'] = np.vectorize(get_player_age_in_season)(all_skaters_2010_df['season'], all_skaters_2010_df['birthDate']) 

# Group by age to get mean TOI per age
age_2010_df = all_skaters_2010_df.groupby(['age']).agg(toi_dictionary).reset_index()

# Rename columns
age_2010_df.columns = ['age', 'avgTimeOnIcePerGame', 'numberOfPlayers']

# Display age distribution
bar_age = px.bar(
            age_2010_df,
            x='age', y='numberOfPlayers',
            hover_data=['numberOfPlayers', 'numberOfPlayers'],
            labels={'age': 'Age', 'numberOfPlayers':'Number of Players'},
            title = 'Skater Age Distribution (2010-2020)'
        )

bar_age.update_layout(
    title={
                'text': "Skater Age Distribution (2010-2020)",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})


# In[ ]:


# Display age distributed avg time on ice
bar_age_toi = px.bar(
            age_2010_df,
            x='age', y='avgTimeOnIcePerGame',
            hover_data=['numberOfPlayers', 'avgTimeOnIcePerGame'],
            labels={'age': 'Age', 'avgTimeOnIcePerGame':'Average TOI/Game'},
            color='numberOfPlayers',
            color_continuous_scale=px.colors.sequential.Bluered
        )

bar_age_toi.update_layout(
    title={
                'text': "Average TOI/Game Grouped by Age (2010-2020)",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

