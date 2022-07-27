from fileinput import hook_compressed
from pyexpat import features, model
from requests import request
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
from itertools import product
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from numpy import random
from datetime import datetime
from streamlit_lottie import st_lottie
import requests

import pandas as pd
import numpy as np
import pymc3 as pm
from statistics import mean
from math import factorial
from scipy.stats import poisson
from scipy.stats import skellam
#from PIL import Image
from traitlets import default

###############################################
# Data from google colab
print(datetime.now().strftime("%H:%M:%S"), "DBG#1: import data")
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
    print(datetime.now().strftime("%H:%M:%S"), "DBG#2: inside score_pred function")
    attack_PMF_home = poisson.pmf(int(i), np.exp(home + attack_data[[home_team]].values + defence_data[[away_team]].values))
    attack_PMF_away = poisson.pmf(int(j), np.exp(attack_data[[away_team]].values + defence_data[[home_team]].values))
    pred = np.mean(attack_PMF_home * attack_PMF_away)
    return pred


###################################################

#Score pred 3D plot
def prob_3d_function(home_team, away_team):
    print(datetime.now().strftime("%H:%M:%S"), "DBG#3: inside prob_3d_function function")
    if home_team == away_team:
        "Not Possible"
    else:
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
    print(datetime.now().strftime("%H:%M:%S"), "DBG#4: inside prob_home_win function")
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
    print(datetime.now().strftime("%H:%M:%S"), "DBG#5: inside outcome_plot function")
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
  

####################################################################################
##                Predicting the 2022/2023 Premier Leauge                         ##

#Step 1 Reading and cleaning the dataset
#print(datetime.now().strftime("%H:%M:%S"), "DBG#6: reading and cleaning the dataset")
#pl_22_23 = pd.read_csv("Pl_22_23.csv")

#pl_22_23.drop(pl_22_23.columns[[0,1, 3]], axis=1, inplace=True)

#pl_22_23['Date'] = pl_22_23['Date'].astype(str)

#for i in range(0, len(pl_22_23)):
#  pl_22_23.Date[i] = pl_22_23.Date[i].split(str(" "), 1)[0]



#pl_22_23["Home Team"] = pl_22_23['Home Team'].replace({'Spurs':'Tottenham', "Man Utd" : "Man United"})
#pl_22_23["Away Team"] = pl_22_23['Away Team'].replace({'Spurs':'Tottenham', "Man Utd" : "Man United"})


#pl_22_23["Home Team"] = pl_22_23['Home Team'].replace({'Nottingham Forest':'Cardiff'})
#pl_22_23["Away Team"] = pl_22_23['Away Team'].replace({'Nottingham Forest':'Cardiff'})

#for i in range(0, 1000):
#  pl_22_23["Sim_" +str(i) + "_HG"] = 0
#  pl_22_23["Sim_" + str(i) + "_AG"] = 0


@st.cache
def test():
    print(datetime.now().strftime("%H:%M:%S"), "DBG#6: reading and cleaning the dataset")
    pl_22_23 = pd.read_csv("Pl_22_23.csv")

    pl_22_23.drop(pl_22_23.columns[[0,1, 3]], axis=1, inplace=True)

    pl_22_23['Date'] = pl_22_23['Date'].astype(str)

    #for i in range(0, len(pl_22_23)):
    #  pl_22_23.Date[i] = pl_22_23.Date[i].split(str(" "), 1)[0]



    pl_22_23["Home Team"] = pl_22_23['Home Team'].replace({'Spurs':'Tottenham', "Man Utd" : "Man United"})
    pl_22_23["Away Team"] = pl_22_23['Away Team'].replace({'Spurs':'Tottenham', "Man Utd" : "Man United"})


    pl_22_23["Home Team"] = pl_22_23['Home Team'].replace({'Nottingham Forest':'Cardiff'})
    pl_22_23["Away Team"] = pl_22_23['Away Team'].replace({'Nottingham Forest':'Cardiff'})

    a = pd.DataFrame([range(0, 2000)])
    pl_22_23 = pd.concat([pl_22_23, a], axis = 1)
    for j in range(0,1000):
        print(datetime.now().strftime("%H:%M:%S"), "DBG#7: inside for loop (0,1000)")
        attack_data_subset = attack_data.iloc[j, :]
        defence_data_subset = defence_data.iloc[j, :]
        
        attack_data_home = attack_data_subset[pl_22_23["Home Team"]].values
        attack_data_away = attack_data_subset[pl_22_23["Away Team"]].values
        
        defence_data_home = defence_data_subset[pl_22_23["Home Team"]].values
        defence_data_away = defence_data_subset[pl_22_23["Away Team"]].values
        
        home_goals = np.exp(np.mean(home) + attack_data_home + defence_data_away)
        away_goals = np.exp(attack_data_away + defence_data_home)
        
        pl_22_23.iloc[:, 4 + 2*j] = random.poisson(home_goals, len(home_goals))
        pl_22_23.iloc[:, 4 + 2*j + 1] = random.poisson(away_goals, len(away_goals))
    return pl_22_23


pl_22_23 = test()

#def Sim_outcome2(sim_number):
#    print(datetime.now().strftime("%H:%M:%S"), "DBG#8: inside Sim_outcome2 function")
#    points = pl_22_23.iloc[:, 1:4]
#    print(datetime.now().strftime("%H:%M:%S"), "DBG#8.1")
#    x = pl_22_23.iloc[:, (4 + 2*sim_number)]
#    y = pl_22_23.iloc[:, (4 + 2 * sim_number + 1)]
#    print(datetime.now().strftime("%H:%M:%S"), "DBG#8.2")

#    points["Home Points"] = (x > y) *3 + 1*(x == y)
#    points["Away Points"] = (x < y) * 3 + 1*(x == y)
#    print(datetime.now().strftime("%H:%M:%S"), "DBG#8.3")
    #points.groupby(['Home Team']).sum().iloc[:, 1] #Home Points pr team
    #points.groupby(["Away Team"]).sum().iloc[:, -1] #Away points pr team

    #Goals scored pr team
#    goals_scored = (pl_22_23.groupby("Home Team").sum().iloc[:, sim_number*2+ 1].values +pl_22_23.groupby("Away Team").sum().iloc[: , sim_number*2 + 2].values) #Goals scored at away 
#    print(datetime.now().strftime("%H:%M:%S"), "DBG#8.4")
#    #Goals conceded pr team
#    goals_conceded = (pl_22_23.groupby("Home Team").sum().iloc[:, sim_number*2 + 2].values + pl_22_23.groupby("Away Team").sum().iloc[: , sim_number*2 + 1].values)
#    print(datetime.now().strftime("%H:%M:%S"), "DBG#8.5")

#    table1 = pd.DataFrame(points["Home Team"].sort_values(ascending = True).unique(), columns= ["Team"])
#    print(datetime.now().strftime("%H:%M:%S"), "DBG#8.6")
#    table1["Games"] = 38
#    table1["Goals Scored"] = goals_scored
#    table1["Goals conceded"] = goals_conceded
#    table1["Goal Difference"] = goals_scored - goals_conceded
#    table1["Total Points"] = points.groupby(['Away Team']).sum().iloc[: ,-1 ].values + points.groupby(['Home Team']).sum().iloc[:,1].values
#    table1["Team"] = table1['Team'].replace({'Cardiff':'Nottingham Forest'})
#    print(datetime.now().strftime("%H:%M:%S"), "DBG#8.7")
#    return table1


def Sim_outcome2(sim_number):
    points = pl_22_23.iloc[:, 1:4]
    x = pl_22_23.iloc[:, (4 + 2*sim_number)]
    y = pl_22_23.iloc[:, (4 + 2 * sim_number + 1)]

    points["Home Points"] = (x > y) *3 + 1*(x == y)
    points["Away Points"] = (x < y) * 3 + 1*(x == y)
    points["Home Goals"] = x
    points["Away Goals"] = y

    table1 = pd.DataFrame(points["Home Team"].sort_values(ascending = True).unique(), columns= ["Team"])
    table1["Games"] = 38
    table1["Total Points"] = points.groupby(['Away Team']).sum().iloc[: ,-1 ].values + points.groupby(['Home Team']).sum().iloc[:,1].values
    table1["Goal Difference"] = points.groupby("Home Team").sum().iloc[:, 3].values + points.groupby("Away Team").sum().iloc[:, 4].values - points.groupby("Home Team").sum().iloc[:, 4].values - points.groupby("Away Team").sum().iloc[:, 3].values 
    table1["Team"] = table1['Team'].replace({'Cardiff':'Nottingham Forest'})

    return table1



random.seed(123)
#@st.cache
def simula(id):
    print(datetime.now().strftime("%H:%M:%S"), "DBG#9: inside simula function")
    test = Sim_outcome2(id)[["Team", "Total Points", "Goal Difference"]]
    test.sort_values(by = ["Total Points", "Goal Difference"], ascending = [False, False], inplace = True, ignore_index = True)
    test.index=  test.index + 1
    test= test["Team"].values
    return test




#def load_model(model_name):
#    a = pd.DataFrame([range(0, 1)])
#    standings = pd.DataFrame(model_name(0))
#    for i in range(1, 1000):
#        standings.insert(i, i, model_name(i))
#    standings.index = standings.index + 1    
#    return standings

#@st.cache(allow_output_mutation=True)
def load_model(func):
    print(datetime.now().strftime("%H:%M:%S"), "DBG#10: inside load_model function")
    test = []
    for i in range(0, 1000):
        test.append(func(i))
    
    test = np.reshape(test, (1000, 20)).T
    test = pd.DataFrame(test)
    return test
standings = load_model(simula)

standings.index = standings.index + 1



random.seed(123)
#index = list(standings.loc[8].value_counts().index)
index = ['Man United','Arsenal', 'Chelsea','Leicester','Tottenham','West Ham','Aston Villa', 'Leeds','Brentford',
 'Wolves','Crystal Palace','Southampton','Everton','Newcastle','Liverpool','Bournemouth','Brighton','Man City',"Fulham" ,'Nottingham Forest']

Standing_probs = pd.DataFrame(index = index)
print(standings.index)

print("********")
print("************* ")
print("************ ")

for i in range(1, 21):
    print(datetime.now().strftime("%H:%M:%S"), "DBG#11: for loop (1,21)")
    Standing_probs = Standing_probs.join(pd.DataFrame(standings.loc[i].value_counts()/1000))

Standing_probs.sort_values(by = 1, ascending = False, inplace = True)
Standing_probs.fillna(0, inplace= True)


#Function 1: What is the probability of team i ending up in position n?

def position_prob(team_name, n):
    print(datetime.now().strftime("%H:%M:%S"), "DBG#12: inside position_prob function")
    prob = Standing_probs.loc[team_name, n]
    return prob


def leage_winning_prob():
    x_bar = list(Standing_probs.index[0:4])
    x_bar.append("Other")
    y_axes = list(Standing_probs.iloc[0:4, 0])
    other = 1 - sum(y_axes)
    y_axes.append(other)
    x_pos = np.arange(len(x_bar))
    fig = plt.figure(figsize=(6,4))    
    plt.bar(x_bar, y_axes, color=("cyan", "red", "blue", "black", "green"))   
    # Create names on the x-axis
    plt.xticks(x_pos, x_bar)
    plt.title("Probability of winning the leauge")
    return fig


def top4():
    Top4_prob = pd.DataFrame(index = list(Standing_probs.index))
    Top4_prob["Top 4 prob"] = Standing_probs.iloc[:, 0].values + Standing_probs.iloc[:, 1].values + Standing_probs.iloc[:, 2].values + Standing_probs.iloc[:, 3].values
    Top4_prob.sort_values(by = "Top 4 prob", ascending = False, inplace = True)
    return Top4_prob.head(8)


def Europa_leauge():
    EU_leauge_prob = pd.DataFrame(index = list(Standing_probs.index))
    EU_leauge_prob["Europa leauge Qualifiers"] = Standing_probs.iloc[:, 4].values + Standing_probs.iloc[:, 5].values
    EU_leauge_prob.sort_values(by = "Europa leauge Qualifiers", ascending = False, inplace = True)
    return EU_leauge_prob.head(6)


def relegation():
    # Summar 4: Relegation
     relegation_prob = pd.DataFrame(index = list(Standing_probs.index))
     relegation_prob["Relegation prob"] = Standing_probs.iloc[:, 17].values + Standing_probs.iloc[:, 18].values + Standing_probs.iloc[:, 19].values
     relegation_prob.sort_values(by = "Relegation prob", ascending = False, inplace = True)
     return relegation_prob.head(6)
    
    

####################################################################################
 
## Lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_coding = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_V72Aqi.json")

 

# #Streamlit
print(datetime.now().strftime("%H:%M:%S"), "DBG#13: streamlit section start")
header = st.container()
Probability_model = st.container()
score_prediction_model = st.container()
Leauge_table_model = st.container()
Outcome_model = st.container()



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
    print(datetime.now().strftime("%H:%M:%S"), "DBG#14: set header properties")
    st.title("Welcome to my summer internship project")
    st.text("This summer I've have developed different regression models to predict different outcomes in accosiation fotball")


with Probability_model:
    print(datetime.now().strftime("%H:%M:%S"), "DBG#15: set probability model properties")
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
    print(datetime.now().strftime("%H:%M:%S"), "DBG#16: set prediction model properties")
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
    elif home_goals == "Select" and away_goals != "Select":
        disp_col.write("please enter number of goals for " + str(home_team))
    elif away_goals == "Select" and home_goals != "Select":
        disp_col.write("Please enter number of goals for " + str(away_team))
    else:
       disp_col.write(score_pred(home_team, away_team, home_goals, away_goals))


with Leauge_table_model:
    st.subheader("Leauge Table probability")
    st.text("What is the probability of a given team to end up in n-th position ? ")

    left_col, display_left_col, right_col = st.columns(3)
    Team = left_col.selectbox("Select a team" ,teams, index = 0)
    position = left_col.selectbox("Select a leauge position" ,range(1, 21))

    display_left_col.subheader("Probability")
    display_left_col.text(position_prob(Team, position))

    with right_col:
        st_lottie(lottie_coding, height= 300, key = "coding")

    
print(index)
print(" ")
print(" ")
print(" ")


with Outcome_model:
    print(datetime.now().strftime("%H:%M:%S"), "DBG#18: set outcome model properties")
    st.subheader("Outcomes for 22/23 season")

    options = ["Winning the premier leauge", "Top 4", "Europa leauge (5th, 6th)", "Relegation"]
    left_col, right_col = st.columns(2)
    
    
    Outcome = left_col.selectbox("Select fotball outcome", options= options, index = 0)
    if Outcome == "Winning the premier leauge":
        left_col.pyplot(leage_winning_prob())
    if Outcome == "Top 4":
        left_col.table(top4())
    
    if Outcome == "Europa leauge (5th, 6th)":
        left_col.table(Europa_leauge())
    
    if Outcome  == "Relegation":
        left_col.table(relegation())

    right_col.text("Simulated probabilities of teams finishing in different leauge positions:")
    right_col.write(Standing_probs)


