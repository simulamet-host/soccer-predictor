leagues = 'epl'

# List of clubs and the number of league positions for each league
clubs_epl = [
    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton & Hove Albion',
    'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich Town', 
    'Leicester City', 'Liverpool', 'Manchester City', 'Manchester United',
    'Newcastle United', 'Nottingham Forest', 'Southampton', 'Tottenham Hotspur',
    'West Ham United', 'Wolverhampton Wanderers'
]
num_league_positions_epl = 20

clubs_laliga = [
    'Alavés', 'Athletic Bilbao', 'Atlético Madrid', 'Barcelona', 'Celta Vigo',
    'Espanyol', 'Getafe', 'Girona', 'Las Palmas', 'Leganés',
    'Mallorca', 'Osasuna', 'Rayo Vallecano', 'Real Betis', 'Real Madrid',
    'Real Sociedad', 'Sevilla', 'Valencia', 'Valladolid', 'Villarreal'
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
    'Nantes', 'Nice', 'Paris Saint-Germain', 'Reims', 'Rennes', 
    'Strasbourg', 'Saint-Étienne', 'Toulouse'
]
num_league_positions_ligue1 = 18

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
