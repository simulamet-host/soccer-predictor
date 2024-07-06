import numpy as np

def calculate_win_lose_draw(home_team, away_team):
    # Simulate probabilities for testing purposes
    home_win_prob = np.random.uniform(0.3, 0.6)
    draw_prob = np.random.uniform(0.2, 0.4)
    away_win_prob = 1 - home_win_prob - draw_prob
    
    return home_win_prob, draw_prob, away_win_prob
