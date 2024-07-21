# leagues.py

league = 'epl'

# List of clubs and the number of league positions for each league
clubs_epl = [
    'Arsenal', 'Aston Villa', 'Blackburn Rovers', 'Chelsea', 'Coventry City', 
    'Crystal Palace', 'Everton', 'Ipswich Town', 'Leeds United', 'Liverpool', 
    'Manchester City', 'Manchester United', 'Middlesbrough', 'Norwich City', 
    'Nottingham Forest', 'Oldham Athletic', 'Queens Park Rangers', 
    'Sheffield United', 'Sheffield Wednesday', 'Southampton', 'Tottenham Hotspur', 
    'Wimbledon'
]
num_league_positions_epl = 20

clubs_laliga = [
    'Almeria', 'Athletic Bilbao', 'Atletico Madrid', 'Barcelona', 'Cadiz', 
    'Celta Vigo', 'Elche', 'Espanyol', 'Getafe', 'Girona', 
    'Mallorca', 'Osasuna', 'Rayo Vallecano', 'Real Betis', 'Real Madrid', 
    'Real Sociedad', 'Real Valladolid', 'Sevilla', 'Valencia', 'Villarreal'
]
num_league_positions_laliga = 20

clubs_bundesliga = [
    'FC Bayern München', 'Borussia Dortmund', 'RB Leipzig', '1. FC Union Berlin', 
    'SC Freiburg', 'Bayer 04 Leverkusen', 'Eintracht Frankfurt', 'VfL Wolfsburg', 
    '1. FSV Mainz 05', 'Borussia Mönchengladbach', 'Holstein Kiel', 'TSG Hoffenheim', 
    'SV Werder Bremen', 'VfL Bochum 1848', 'FC Augsburg', 'VfB Stuttgart', 
    'FC St. Pauli', '1. FC Heidenheim 1846'
]
num_league_positions_bundesliga = 18

clubs_ligue1 = [
    'Angers', 'Auxerre', 'Brest', 'Le Havre', 'Lens', 
    'Lille', 'Lyon', 'Marseille', 'Monaco', 'Montpellier', 
    'Nantes', 'Nice', 'Paris Saint-Germain', 'Reims', 'Rennais', 
    'Saint-Étienne', 'Strasbourg', 'Toulouse'
]
num_league_positions_ligue1 = 20

# Define a function to get league metadata
def get_league_metadata(league=league):
    if league == 'epl':
        return clubs_epl, num_league_positions_epl
    elif league == 'laliga':
        return clubs_laliga, num_league_positions_laliga
    elif league == 'bundesliga':
        return clubs_bundesliga, num_league_positions_bundesliga
    elif league == 'ligue1':
        return clubs_ligue1, num_league_positions_ligue1
    else:
        return [], 0
