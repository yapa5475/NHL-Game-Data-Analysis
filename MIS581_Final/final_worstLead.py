#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# In[3]:


print(os.listdir('.'))


# In[7]:


game_teams_stats = pd.read_csv('game_teams_stats.csv', delimiter=',', usecols=['game_id', 'team_id', 'HoA', 'won', 'settled_in', 'goals', 'shots', 'powerPlayGoals'])


# In[9]:


game_teams_stats.head()


# In[10]:


# combine home and away results into one row

b = game_teams_stats.goals.value_counts().to_frame()
game_teams_stats.goals.value_counts().to_frame().plot.bar()
b = pd.DataFrame([{"goals": int(b.loc[[0, 1, 2, 3] , :].sum())}, {"goals": int(b.loc[[4,5,6,7,8,9,10], :].sum())}], index = ["3 or less goal games", "4 or more goal games"]).plot.bar()
b.set_ylabel("Games")
b.get_legend().remove()
plt.xticks([0,1], ["3 or less goal games", "4 or more goal games"], rotation="horizontal")


# In[12]:


# determine out how many times a team won

print(game_teams_stats.loc[(game_teams_stats["won"] == True)]["team_id"].value_counts().mean())
print(game_teams_stats.loc[(game_teams_stats["won"] == True) & (game_teams_stats["team_id"] == 14)]["team_id"].value_counts())
print(game_teams_stats.loc[(game_teams_stats["won"] == True) & (game_teams_stats["team_id"] == 21)]["team_id"].value_counts())
print(game_teams_stats.loc[(game_teams_stats["won"] == True) & (game_teams_stats["team_id"] == 26)]["team_id"].value_counts())
print(game_teams_stats.loc[(game_teams_stats["won"] == True) & (game_teams_stats["team_id"] == 2)]["team_id"].value_counts())
print(game_teams_stats.loc[(game_teams_stats["won"] == True) & (game_teams_stats["team_id"] == 22)]["team_id"].value_counts())


# In[ ]:


# REALLY STARTS HERE


# In[19]:


game_plays = pd.read_csv('game_plays.csv', delimiter=',', usecols=['play_id', 'game_id', 'play_num', 'team_id_for', 'team_id_against', 'event', 'secondaryType', 'period', 'periodType'])


# In[23]:


game_plays_goals = game_plays.loc[(game_plays["event"] == 'Goal') & (game_plays["periodType"] != 'SHOOTOUT')].sort_values(by = ["game_id", "play_num"])
game_plays_goals.head()


# In[24]:


lead_data = {}
lost_lead_game_data = {}


# In[25]:


for index, row in game_plays_goals.iterrows():
    
    game_id = row["game_id"]
    team_id_for = row["team_id_for"]
    team_id_against = row["team_id_against"]
    
    if game_id not in lead_data:
        lead_data[game_id] = {}
        lead_data[game_id]["largest_lead"] = 0
        lead_data[game_id][team_id_for] = 0
        lead_data[game_id][team_id_against] = 0        

    lead_data[game_id][team_id_for] += 1
    
    score_dif = lead_data[game_id][team_id_for] - lead_data[game_id][team_id_against]
    
       
    if score_dif >= 2:
        # >= because wanna know latest lead
        if score_dif >= lead_data[game_id]["largest_lead"]:
            if game_id in lost_lead_game_data:
                if lost_lead_game_data[game_id]["largest_lead_team"] == team_id_for:
                    lead_data[game_id]["largest_lead"] = score_dif
                    lead_data[game_id]["largest_lead_score"] = str(lead_data[game_id][team_id_for]) + "-" + str(lead_data[game_id][team_id_against])
                    lead_data[game_id]["largest_lead_team"] = team_id_for
            else:
                lead_data[game_id]["largest_lead"] = score_dif
                lead_data[game_id]["largest_lead_score"] = str(lead_data[game_id][team_id_for]) + "-" + str(lead_data[game_id][team_id_against])
                lead_data[game_id]["largest_lead_team"] = team_id_for

            if game_losing_team.loc[game_losing_team["game_id"] == game_id]["team_id"].squeeze() == team_id_for:
                lead_data[game_id]["winning_team"] = team_id_against
                lost_lead_game_data[game_id] = lead_data[game_id]
                lost_lead_game_data[game_id]["settled_in"] = game_losing_team.loc[game_losing_team["game_id"] == game_id]["settled_in"].squeeze()

    elif score_dif == 0 and game_id in lost_lead_game_data:
        if "period_tied" not in lost_lead_game_data[game_id]:
            lost_lead_game_data[game_id]["period_tied"] = row["period"]


# In[ ]:




