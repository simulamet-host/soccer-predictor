from fileinput import hook_compressed
from pyexpat import features, model
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
from itertools import product
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from statistics import mean
from math import factorial
from scipy.stats import poisson
from scipy.stats import skellam
from PIL import Image
from traitlets import default

###############################################
# Data from google colab
attack_data = pd.read_csv("attack_data.csv")
attack_data.drop(["Unnamed: 0"], axis = 1, inplace=True)

defence_data = pd.read_csv("defence_data.csv")
defence_data.drop(["Unnamed: 0"], axis = 1, inplace= True)
home = pd.read_csv("Home.csv")

home.drop(["Unnamed: 0"], axis = 1, inplace= True)
home = np.array(home["0"])

################################################

##############################################################################

#Predicting results Using PMF of attacking, defensice and home parameter

def score_pred(home_team, away_team, i, j):
    attack_PMF_home = poisson.pmf(int(i), np.exp(home + attack_data[[home_team]].values + defence_data[[away_team]].values))
    attack_PMF_away = poisson.pmf(int(j), np.exp(attack_data[[away_team]].values + defence_data[[home_team]].values))
    pred = np.mean(attack_PMF_home * attack_PMF_away)
    return pred

###################################################

#Score pred 3D plot
def prob_3d_function(home_team, away_team):
    x_axis = np.arange(0, 8, 1)
    y_axis = np.arange(0, 8, 1)
    goals = pd.DataFrame(list(product(x_axis, y_axis)), columns=['home_goals', 'away_goals'])
    goals["prob"] = 0.0
    
    for i in range(0, len(goals)):
        goals["prob"][i] = '{:f}'.format(score_pred(home_team, away_team, goals["home_goals"][i], goals["away_goals"][i])* 100)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111 ,projection = "3d")
    #Positions
    xpos = goals["home_goals"]
    ypos = goals["away_goals"]
    zpos = [0] *len(xpos)
    
    dx = np.ones(len(xpos))
    dy = np.ones(len(ypos))
    dz = list(map(float, goals["prob"]))
    
    #Barplot
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color = "#00ceaa", shade = True)
    ax1.set_ylim(8,0)
    ax1.set_xlim(0, 8)
    ax1.set_zlim(0,21)
    ax1.set_xlabel("Goals scored from " + str(home_team))
    ax1.set_ylabel("Goals scored from " + str(away_team))
    ax1.set_zlabel("Probability score in percentage")
    return fig

#What is the probability of a given home team to win against a random team in the premier leauge: ?
def prob_home_win(home_team, away_team):
    if home_team == away_team: 
        #image = Image.open("Red-Card.jpg")
        return "Not Possible" #disp_col.image(image, caption= "NOT POSSIBLE!")
    else:
        #Hometeam:
        attack_i = attack_data[[home_team]].values
        defence_i = defence_data[[home_team]].values
         
         #Awayteam:
        attack_j = attack_data[[away_team]].values
        defence_j = defence_data[[away_team]].values
         
        atts_i = np.exp(home + attack_i + defence_j)
        atts_j = np.exp(attack_j + defence_i)
         
        prob_W_home = skellam.sf(0, np.mean(atts_i), np.mean(atts_j))
        prob_draw = skellam.pmf(0, np.mean(atts_i), np.mean(atts_j))
        prob_W_away = 1- prob_W_home - prob_draw
    
    return prob_W_home, prob_draw, prob_W_away

############################################################################
#Prob plot
def outcome_plot(home_team, away_team):
  Heights = prob_home_win(home_team, away_team)
  bars = (home_team + " winning", "Draw", away_team + " winning")
  xpos = np.arange(len(bars))

  fig =plt.figure(figsize=(6,4))

  plt.bar(xpos, Heights, color =["green", "yellow", "red"])
  plt.xticks(xpos, bars)
  plt.xlabel("Match_Outcome")
  plt.ylabel("Probability")
  plt.ylim(0, 0.9)
  return fig
  
##################################################3
#Streamlit
header = st.container()
dataset = st.container()
Probability_model = st.container()
score_prediction_model = st.container()

st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html= True
)

with header:
    st.title("Welcome to my summer internship project")
    st.text("In this project im creating a model to predict different outcomes in accosiation fotball")

with dataset:
    st.header("Dataset")
    st.text("The dataset is based on four seasons from premier leauge fotball 18/19-21/22")

with Probability_model:
    st.header("Probabilities of hometeam winning a game")
    st.text("What is the probability of a given team i to beat a team j")

    sel_col, disp_col = st.columns(2)
    teams = list(attack_data.columns)
    teams2 = list(attack_data.columns.sort_values(ascending = False))
    n_estimators = sel_col.selectbox("Please select Home Team ?", options = teams, index = 0)
    n_estimators2 = sel_col.selectbox("Please select the Away Team ?", options = teams2, index = 0)

    disp_col.subheader("Probability of home team winning : ")
    #If statement to avoid error
    disp_col.pyplot(outcome_plot(n_estimators, n_estimators2))

    #disp_col.write(prob_home_win(n_estimators, n_estimators2))

with score_prediction_model:
    st.header("Score prediction probability")
    st.text("Here you can calculate probabilities of different fotball results between home and away team")

    sel_col, set_slide ,disp_col = st.columns(3)

    home_team = sel_col.selectbox("Select Home team", options= teams, index= 0)
    away_team = sel_col.selectbox("Select Away team", options = teams2, index= 0)

    home_goals = set_slide.selectbox("Select number of Home goals:", ("Select",0, 1, 2,3,4,5,6,7,8,9))
    away_goals = set_slide.selectbox("Select number of away goals:", ("Select",0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

    disp_col.subheader("Probability : ")
    if home_goals == "Select" and away_goals == "Select":
        disp_col.pyplot(prob_3d_function(home_team= home_team, away_team= away_team))
    if home_goals == "Select" and away_goals != "Select":
        disp_col.write("")
    #elif home_goals == "Select" or away_goals == "Select":
    #    disp_col.write("Please enter a value for goals scores")
    else:
       disp_col.write(score_pred(home_team, away_team, home_goals, away_goals))
