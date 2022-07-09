#!/usr/bin/env python
# coding: utf-8

# In[28]:


# Data import: game.csv, game_teams_stats.csv, drop extra columns, and separate home and away
import pandas as pd

game = pd.read_csv("game.csv", index_col=0)
game_teams_stats = pd.read_csv("game_teams_stats.csv", index_col=0)

team_data = game_teams_stats.drop(['head_coach','goals', "team_id", "powerPlayGoals","startRinkSide"], axis=1)

hmask = team_data.HoA == "home"
home_data = team_data[hmask]
away_data = team_data[~hmask]

# Edit home and away datasets
home_data = home_data.drop(["won", "settled_in", "faceOffWinPercentage"], axis=1)
home_data = home_data.add_prefix("home_")
away_data = away_data.add_prefix("away_")


# In[40]:


# Question 1: To what degree does the home team have an advantage in an NHL game?

# separate the home and away teams to prevent use of overlapping data
# homeMask = game.HoA == "home*"
homeMask = game.outcome == "home*"
home_data = game[homeMask]
away_data = game[~homeMask]

# Edit home and away datasets
# home_data = home_data.drop(["won", "settled_in", "faceOffWinPercentage"], axis=1)
home_data = home_data.add_prefix("home_")
away_data = away_data.add_prefix("away_")

# Add specific home features to the away dataset
team_data = pd.concat([away_data, home_data], axis=1)

# print(away_data)
# print(home_data)
print(team_data)


# In[43]:


# Remove duplicated entries from the 2nd dataset
import re

print("Duplicated entries: {}\n".format(sum(team_data.index.duplicated())))
team_data = team_data[~team_data.index.duplicated()]

# Readjust the timezones (exclude daylight savings)
def timezone_change(tz):
    res = re.search("[A-Z]DT", tz) 
    if res != None:
        return 1
    else:
        return 0 

# Find the offset of the daylight savings timezones
offset = team_data.home_venue_time_zone_tz.apply(timezone_change)


# Adjust the timezones 
team_data.venue_time_zone_offset = (team_data.venue_time_zone_offset - offset)

# Plot the finalized timezones 
plt.hist(team_data.venue_time_zone_offset)
plt.xlabel("Timezone Offset")


# In[ ]:


Duplicated entries: 2570

Text(0.5, 0, 'Timezone Offset')


# In[33]:


print(team_data)


# In[47]:


#set up for log reg
# Collect categorical variables 
categorical = [name for name in data.columns if data[name].dtype == 'object']
categorical = categorical + ["season", "away_timezone", "venue_time_zone_offset", "game_time"]

# Collect numeric variables 
numeric = [column for column in data.columns if column not in categorical]
numeric.remove("away_won")
print(categorical)
print(numeric)

# Set up core pipeline for preprocessing
categorical_transform = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown = "ignore"))])

num_transform = Pipeline(steps=[
    ('scaler', StandardScaler())])

preprocess = ColumnTransformer(
    transformers=[
        ('num', num_transform, numeric),
        ('cat', categorical_transform, categorical)])


# In[48]:


# Variable importance
importances = rf_final.named_steps['rf'].feature_importances_
indices = np.argsort(importances)[::-1]

# PLOTTING 
f,ax = plt.subplots(figsize=(6,8))
plt.title("Variable Importance - XGBoosted Model")
sns.barplot(y=[X_train.columns[i] for i in indices[0:8]] , x=importances[indices[0:8]])


# In[46]:


# logistic regression model
from sklearn.pipeline import Pipeline
import matplotlib
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import seaborn as sns
import sklearn as sk 
import random as rd
import re 
from sklearn.linear_model import LinearRegression, LogisticRegression


log_model = Pipeline([
    ('preprocess',preprocess),
    ('log',LogisticRegression( max_iter=10000))
])

cv_score = cross_val_score(log_model, X_train, y_train, cv=8, scoring='roc_auc')


# In[51]:


log_model = Pipeline([
    ('preprocess',preprocess),
    ('log',LogisticRegression( max_iter=10000))
])

cv_score = cross_val_score(log_model, X_train, y_train, cv=8, scoring='roc_auc')

# print(f"Mean CV AUROC Score: {cv_score.mean()}")

log_model.fit(X_train, y_train)


y_test_pred = log_model.predict(X_test)
y_test_prob = log_model.predict_proba(X_test)

compute_performance(y_test_pred, y_test, log_model.classes_)


# In[ ]:


# Variable importance
importances = rf_final.named_steps['rf'].feature_importances_
indices = np.argsort(importances)[::-1]

# PLOTTING 
f,ax = plt.subplots(figsize=(6,8))
plt.title("Variable Importance - XGBoosted Model")
sns.barplot(y=[X_train.columns[i] for i in indices[0:8]] , x=importances[indices[0:8]])


# In[12]:





# In[14]:





# In[ ]:




