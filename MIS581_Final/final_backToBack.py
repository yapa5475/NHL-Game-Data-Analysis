#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd


# In[37]:


# init packages
# library(tidyverse) # metapackage with lots of helpful functions
# library(lubridate) # package for manipulating date


# In[53]:


game_df = pd.read_csv('game.csv')
print(game_df)


# In[59]:


teams_df = pd.read_csv('team_info.csv')
print(teams_df)


# In[70]:


backToback = game_df
filter(type != "P", season != "20122013")


# In[71]:


hockeyBack2Back<-hockeyDF %>%
  filter(type != "P", season != "20122013") %>%
  mutate(date_time = if_else(venue_time_zone_tz == "EDT", format(date_time, tz="America/Toronto"),
                        if_else(venue_time_zone_tz == "CDT", format(date_time, tz="America/Chicago"),
                          if_else (venue_time_zone_tz == "PDT", format(date_time, tz="America/Vancouver"),
                            if_else (venue_time_zone_tz == "MDT",format(date_time, tz="America/Denver"),
                                    format(date_time, tz="America/Chicago")))))) %>%
  select(game_id,season,date_time,away_team_id,home_team_id,outcome) %>%
  gather(home_away,team,away_team_id:home_team_id) %>%
  arrange(team,date_time) %>%
  mutate(gameDate=as.Date(date_time))%>%
  mutate(gameTime=format(as.POSIXct(date_time),format = "%H:%M")) %>%
  mutate(back2backGames = if_else(gameDate - lag(gameDate) == 1, "yes", "no")) %>%
  mutate(outcome= if_else(grepl("away", outcome),"away","home")) %>%
  mutate(home_away = if_else(grepl("away", home_away),"away","home"))%>%
  mutate(winner = if_else(outcome==home_away,"yes","no")) %>%
  left_join(teams,by=c("team" = "team_id")) %>%
  select(-team)


# In[74]:


seasonStats <- hockeyBack2Back %>%
  group_by(season,teamName) %>%
  summarise(totalWins=sum(winner=="yes"),
            totalLosses=sum(winner=="no"),
            totalBack2Backs=sum(back2backGames=="yes"),
            back2BackWins=sum(back2backGames=="yes"&winner=="yes"),
            back2BackLoss=sum(back2backGames=="yes"&winner=="no"),
            winPercentageTotal=(totalWins/82*100),
            winPercentageBack2Back=(back2BackWins/totalBack2Backs*100),
            winDifferential=(winPercentageTotal - winPercentageBack2Back)) %>%
  group_by(teamName,season,totalWins)%>%
  arrange(teamName)


# In[ ]:




