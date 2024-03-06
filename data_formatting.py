""" == Authors: Jessica Houghton and Sasha Yousefi
data_formatting.py

This file preprocesses all game, player, and event data for the entire Wyscout 
dataset. Once run, all games are stored in the processed_games folder with the
data necessary to run the model. 

Classes
-------
Wyscout
    A class storing all publicly available Wyscout football match data. Has one
    Game object for each file in the games directory. 
Game
    Processes and stores all information related to one game in the Wyscout dataset.
"""

import argparse
import csv
from datetime import *
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys


class Wyscout:
    """ A class storing all publicly available Wyscout football match data. 

    Up to date datasets can be found on the following website: 
    https://figshare.com/collections/_/4415000. More information about variables
    can be found on the Wyscout API: 
    https://support.wyscout.com/matches-wyid-events

    Attributes
    ----------
    games_dir (str): directory to all games in the wyscout data source
    event_to_subevent (dict) : event to subevent dictionary
    subeventID0index (dict) : subevent IDs to their 0-indexed value
    all_players (df) : all players for every team and their relevant data
    matchPeriodindex (dict) : zero indexed match periods
    tagids2label (dict) : event tag IDs to label
    eventsec_min (float) : minimum seconds of an event across all game data 
    eventsec_max (float) : maximum seconds of an event across all game data 
    playerID0index (dict) : player IDs to their 0-indexed value. In the Wyscout 
                            data, sometimes there are actions without 
                            corresponding players. The value 0 in this 
                            dictionary encompasses all player-less actions. 
    weight_min (float) : minimum weight of all players, measured in kilograms 
    weight_max (float) : maximum weight of all players, measured in kilograms  
    height_min (float) : minimum height of all players, measured in centimeters 
    height_max (float) : maximum height of all players, measured in centimeters 
    age_min (int) : minimum age of all players, measured in years 
    age_max (int) : maximum age of all players, measured in years  
    players_2_touches (dict) : each player and their event counts across the dataset  


    Methods
    -------
    process_subevents(eventid2name_path = 'eventid2name.csv')  
        Save all events and subevents in the event_to_subevent dictionary.
    process_players(player_path = 'players.json')
        Zero index player IDs and store relevant player data in players_clean. 
    process_tags(tag_path = 'tags2name.csv')
        Create a dictionary of tag IDs to the label name for one hot encoding.
    find_eventsec_extremes(directory)
        Find the min and max event seconds in the entire dataset.  
    fill_footedness()
        Manually fill in missing footedness values.   
    fill_height_weight()
        Fill in the mising height and weight values with averages.
    age(born)
        Calculate and return the age in years for a person. 
    process_games()
        Process all games in the dataset. 
    most_freq_players()
        Limit players to those with touch counts above the 25% quantile.
    """
    games_dir = "" 
    event_to_subevent = {} 
    subeventID0index = {}
    all_players = pd.DataFrame() 
    matchPeriod0index = {'1H': 0, '2H': 1, 'E1':2, 'E2': 3}
    tagids2label = {} 
    eventsec_min = 0
    eventsec_max = 0
    playerID0index = {0:0} 
    weight_max = 0
    weight_min = 0
    height_max = 0
    height_min = 0
    age_min = 0
    age_max = 0
    players_2_touches = {} 


    def __init__(self, games_dir, eventid2name_path, player_path, tags_path):
        """Calls on all functions necessary for a functioning Wyscout object."""
        self.games_dir = games_dir
        self.process_subevents(eventid2name_path)
        self.process_players(player_path)
        self.process_tags(tags_path)
        self.find_eventsec_extremes(games_dir)
        self.fill_footedness()
        self.fill_height_weight()
        self.players['age'] = self.players['birthDate'].apply(self.age)

        # create a dictionary or player IDs to weight, height, age,
        self.playerids2weight = dict(zip(self.players.wyId, self.players.weight))
        self.playerids2height = dict(zip(self.players.wyId, self.players.height))
        self.playerids2age = dict(zip(self.players.wyId, self.players.age))
        self.playerids2foot = dict(zip(self.players.wyId, self.players.foot))
        self.playerids = np.sort(self.players['wyId'].unique())

        # find height, weight, and age min and max for normalization
        self.weight_max = max(self.players['weight'])
        self.weight_min = min(self.players['weight'])
        self.height_max = max(self.players['height'])
        self.height_min = min(self.players['height'])
        self.age_max = max(self.players['age'])
        self.age_min = min(self.players['age'])
        self.fill_footedness()

        self.process_games()
        self.most_freq_players()
      
      
    def process_subevents(self, eventid2name_path = 'eventid2name.csv'):
        """Save all events and subevents in the event_to_subevent dictionary.

        eventid2name.csv contains a mapping of all event types to their 
        corresponding subevents. First, zero index all subevents for categorical 
        input into the model. Then, create a dictionary with the keys as event 
        indices and the values as a list of the original (non zero-indexed) 
        subevents. This will be useful when having to manually input subevents 
        that are missing from the data.
        """
        eventid2name = pd.read_csv("eventid2name.csv")

        # zero indexing of subevents
        self.subeventID0index = {k: v for v, k in enumerate(eventid2name['subevent'])}

        # event to subevent dictionary
        for i, j in zip(eventid2name.event,eventid2name.subevent):
            self.event_to_subevent.setdefault(i, []).append(j)


    def process_players(self, player_path = 'players.json'):
        """Zero index player IDs and store relevant player data in players_clean. 

        players.json contains all players that are present on the teams 
        represented in this dataset as well as relevant information such as 
        their national team ID, club team ID, footedness, weight, height, and 
        more. 

        IMPORTANT NOTE: In some events, no player ID is recorded (wyID=0). For 
        now, this is left as is in the dataset. Future work can explore other 
        options for handling missing players.
        """
        self.players = pd.json_normalize(json.load(open(player_path)))

        # 0 indexing of players for categorical encoding
        uniquePlayers = self.players['wyId'].unique()
        for i in range(len(uniquePlayers)):
            self.playerID0index[uniquePlayers[i]] = i+1

        # Create and save a cleaned version for data visualization use later on
        players_clean = self.players[['firstName', 'lastName', 'wyId', 
                                      'role.name', 'birthArea.name']].copy()
        fullname = players_clean['firstName'] + ' ' + players_clean['lastName']
        players_clean['FullName'] = fullname
        players_clean = players_clean.drop(['firstName', 'lastName'], axis=1)
        players_clean.to_csv('players_clean.csv')


    def process_tags(self, tag_path = 'tags2name.csv'):
        """Create a dictionary of tag IDs to the label name for one hot encoding."""
        tags2name = pd.read_csv(tag_path)
        self.tagids2label = dict(zip(tags2name.Tag, tags2name.Label))
    

    def find_eventsec_extremes(self, directory):
        """Find the min and max event seconds in the entire dataset. 

        Input:
            directory (string): directory path to game data
        Output:
            eventsec_min (float): minimum seconds that an event occured
            eventsec_max (float): maximum seconds that an event occured
        """
        self.eventsec_min = np.inf
        self.eventsec_max = 0
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            data = pd.json_normalize(json.load(open(f)))
            game = pd.DataFrame(data['events'][0])
            min = game.iloc[0]['eventSec']
            max = game.iloc[-1]['eventSec']
            if min < self.eventsec_min:
                self.eventsec_min = min
            if max > self.eventsec_max:
                self.eventsec_max = max


    def fill_footedness(self):
        """Manually fill in missing footedness values. 

        Options for footedness inclue 'right', 'left', or 'both'. The majority 
        of these values come from each player's profile on 
        https://www.transfermarkt.us/ .
        """

        self.players.loc[2, 'foot'] = 'right'
        self.players.loc[453, 'foot'] = 'right'
        self.players.loc[536, 'foot'] = 'right'
        self.players.loc[644, 'foot'] = 'right'
        self.players.loc[768, 'foot'] = 'right'
        self.players.loc[810, 'foot'] = 'right'
        self.players.loc[845, 'foot'] = 'left'
        self.players.loc[866, 'foot'] = 'left'
        self.players.loc[1240, 'foot'] = 'right'
        self.players.loc[1395, 'foot'] = 'right'

        self.players.loc[1881, 'foot'] = 'right'
        self.players.loc[2075, 'foot'] = 'left'
        self.players.loc[2138, 'foot'] = 'right'
        self.players.loc[2144, 'foot'] = 'left'
        self.players.loc[2149, 'foot'] = 'right'
        self.players.loc[2231, 'foot'] = 'both'
        self.players.loc[2809, 'foot'] = 'right'
        self.players.loc[2826, 'foot'] = 'right'
        self.players.loc[3160, 'foot'] = 'left'
        self.players.loc[3163, 'foot'] = 'right'

        self.players.loc[3190, 'foot'] = 'left'
        self.players.loc[3195, 'foot'] = 'left'
        self.players.loc[3196, 'foot'] = 'right'
        self.players.loc[3205, 'foot'] = 'right'
        self.players.loc[3211, 'foot'] = 'left'
        self.players.loc[3225, 'foot'] = 'right'
        self.players.loc[3226, 'foot'] = 'right'
        self.players.loc[3227, 'foot'] = 'right'
        self.players.loc[3312, 'foot'] = 'right'
        self.players.loc[3316, 'foot'] = 'right'

        self.players.loc[3317, 'foot'] = 'right'
        self.players.loc[3318, 'foot'] = 'left'
        self.players.loc[3342, 'foot'] = 'left'
        self.players.loc[3371, 'foot'] = 'right'
        self.players.loc[3385, 'foot'] = 'right'
        self.players.loc[3421, 'foot'] = 'left'
        self.players.loc[3426, 'foot'] = 'right'
        self.players.loc[3430, 'foot'] = 'left'
        self.players.loc[3433, 'foot'] = 'right'
        self.players.loc[3436, 'foot'] = 'both'

        self.players.loc[3439, 'foot'] = 'right'
        self.players.loc[3441, 'foot'] = 'left'
        self.players.loc[3447, 'foot'] = 'right'
        self.players.loc[3449, 'foot'] = 'left'
        self.players.loc[3452, 'foot'] = 'left'
        self.players.loc[3453, 'foot'] = 'left'
        self.players.loc[3455, 'foot'] = 'right'
        self.players.loc[3456, 'foot'] = 'left'

        self.players.loc[685, 'foot'] = 'left'
        self.players.loc[865, 'foot'] = 'left'
        self.players.loc[2662, 'foot'] = 'right'
        self.players.loc[3224, 'foot'] = 'left'


    def fill_height_weight(self):
        """Fill in the mising height and weight values with averages. 

        In the interest of time, we did not manually input these values. 
        However, this could be of benefit in the future. 
        """
        avg_height = np.mean(self.players[self.players['height'] != 0]['height'])
        avg_weight = np.mean(self.players[self.players['weight'] != 0]['weight'])
        self.players['height'] = self.players['height'].replace(0, avg_height)
        self.players['weight'] = self.players['weight'].replace(0, avg_weight)


    def age(self, born):
        """Calculate and return the age in years for a person. 

        Input:
            born (string): date of birth in the formed YYYY-MM-DD
        Output:
            age in years (int)
        """
        born = datetime.strptime(born, "%Y-%m-%d").date()
        today = date.today()
        return today.year - born.year - ((today.month,
                                          today.day) < (born.month,
                                                        born.day))


    def process_games(self):
        """Process all games in the dataset. 

        For each game file, create a Game instance, process the game, and add 
        the number of events for each player to players_2_touches. Then, save 
        all processed game events to the processed_games folder. 
        """
        count = 0
        for filename in os.listdir(self.games_dir):
            count += 1
            game = Game(self, filename)
            game.process_game(self.games_dir, True)

            # count the number of events for all players
            num_occurances = game.events.groupby('playerId').count().to_dict(orient='dict')['subEventId']

            # add the number of events to players_2_touches
            for key in num_occurances:
                if key in self.players_2_touches and key != 0: 
                    self.players_2_touches[key] += num_occurances[key]
                elif key != 0: 
                    self.players_2_touches.setdefault(key, num_occurances[key])

            game.events.to_csv('processed_games/{}.csv'.format(filename[:-5]), 
                                index=False)
            print("Processed file #{}, game {}".format(count, filename))


    def most_freq_players(self):
        """
        Limit players to those with touch counts above the 25% quantile.
        Players with minimal plays are unlikely to generate meaningful embeddings.
        """
        quantile = np.quantile(list(self.players_2_touches.values()), q = .25)
        players_to_use = [player[0] for player in self.players_2_touches.items()\
                          if player[1] >= quantile]
        # write players with touches above quantile to file
        file =  open('players_clean_quantile.csv', 'w+', newline ='')
        # writing the data into the file
        with file:
            wr = csv.writer(file, quoting=csv.QUOTE_ALL)
            wr.writerow(players_to_use)


class Game(Wyscout):
    """Processes and stores all information related to one game in the Wyscout dataset.

    This class is a composite class of Wyscout, where Wyscout has many Game 
    objects. 

    Attributes
    ----------
    game_path (str) : path to this game 
    events (df) : all event data for this game

    Methods
    -------
    get_xy_cols()
        Converts raw x and y values into columns in the dataframe.
    get_player_vals()
        Adds 'height', 'weight', 'age', and 'foot' columns to events.
    one_hot_encode_tags()
        One hot encodes all tag values and stores each as columns in events.
    fix_null_subevents()
        Randomly inputs null subevents from the eventId of that action.
    get_score()
        Input the 'score_diff' column in a game event dataframe. 
    normalize()
        Normalize all numerical features in the events dataframe.
    process_game(games_dir, original_IDs = False)
        Call all functions necessary to process the current game. 
    """
    game_path = ""
    events = pd.DataFrame()

    def __init__(self, wyscout, path):
        self.wyscout = wyscout
        self.game_path = path


    def get_xy_cols(self):
        """Converts raw x and y values into columns in the dataframe.

        The x and y values range from [0,100], indicating the percentage of the 
        field from the perspective of the attacking team. Originally, the x and 
        y values are stored in a list of two dictionaries (for example: 
        [{'y': 50, 'x': 51}, {'y': 54, 'x': 35}]). The first dictionary is where 
        the event begins and the second dictionary is where the event ends. For 
        some events, there is only onex and y position. In this case, the center 
        of the field (x=50, y=50) is stored as the x2 and y2.

        Input:
            df (dataframe): dataframe corresponding to all events for one game
        Output:
            x1 (int): the x position of the start of the event
            y1 (int): the y position of the start of the event
            x2 (int): the x position of the end of the event (or 50 in the case 
                      of a one-positional event)
            y2 (int): the y position of the end of the event (or 50 in the case 
                      of a one-positional event)
        """
        x1, y1, x2, y2 = [], [], [], []
        for i, pos in enumerate(self.events['positions']):
            x1.append(pos[0]['x'])
            y1.append(pos[0]['y'])
            if len(pos) == 2:
                x2.append(pos[1]['x'])
                y2.append(pos[1]['y'])
            else:
                x2.append(50)
                y2.append(50)
        self.events['x1'], self.events['y1'], self.events['x2'],\
        self.events['y2'] = x1,y1,x2,y2


    def get_player_vals(self):
        """Adds 'height', 'weight', 'age', and 'foot' columns to events."""
        self.events['height'] = self.events['playerId'].map(self.wyscout.playerids2height)
        self.events['weight'] = self.events['playerId'].map(self.wyscout.playerids2weight)
        self.events['age'] = self.events['playerId'].map(self.wyscout.playerids2age)
        #game['foot'] = game['playerId'].map(playerids2foot)


    def one_hot_encode_tags(self):
        """One hot encodes all tag values and stores each as columns in events."""
        for tag in self.wyscout.tagids2label.values():
            self.events["tags_"+tag] = 0
        for i in range(self.events.shape[0]):
            for tag in self.events['tags'][i]:
                self.events.loc[i, 'tags_'+self.wyscout.tagids2label[tag['id']]] = 1

    
    def fix_null_subevents(self):
        """Randomly inputs null subevents from the eventId of that action."""
        null_indicies = np.ravel(np.where(self.events['subEventId'] == ''))
        for idx in null_indicies:
            self.events.loc[idx,"subEventId"] = np.random.choice(self.wyscout.event_to_subevent[self.events.loc[idx,"eventId"]])
        self.events['subEventId'] = self.events['subEventId'].astype(int)


    def own_goals(self, teams):
        """Add a goal to the opposing team score for any own goals.
        
        Input:
            teams (list of strings): the names of the two teams playing in the 
                                     game.
        """
        if any(self.events['tags_own_goal'].values):
            own_goals = self.events[(self.events['tags_own_goal'] == 1)]
            for index, own_goal in own_goals.iterrows():
                if own_goal['teamId'] == teams[0]:
                    self.events.loc[index:,'team_'+ str(teams[1]) + '_score'] += 1
                else:
                    self.events.loc[index:,'team_'+ str(teams[0]) + '_score'] += 1

    
    def get_score_diff(self, teams):
        """Calculate the 'score_diff' column in the event dataframe.
        
        This value is the difference in score between the team of the player who 
        is completing the action minus the score of the opposing team. For
        example, if the score is 1-0, score_diff will be 1 for all players on 
        the current winning team and -1 for all players on the current losing
        team. If later in the game the score changes to 1-1, score_diff will be 
        set to 0 for all players.

        Input:
            teams (list of strings): the names of the two teams playing in the 
                                     game.
        """
        self.events['score_diff'] = np.zeros(len(self.events.index))
        self.events['score_diff'] = np.where(self.events['teamId'] == teams[0], 
                                             self.events['team_'+ str(teams[0]) 
                                             + '_score'] - self.events['team_'
                                             + str(teams[1]) + '_score'], 
                                             self.events['score_diff'] )
        self.events['score_diff'] = np.where(self.events['teamId'] == teams[1], 
                                             self.events['team_'+ str(teams[1]) 
                                             + '_score'] - self.events['team_'
                                             + str(teams[0]) + '_score'], 
                                             self.events['score_diff'] )



    def get_score(self):
        """Count all goals across each game, and create the 'score_diff' column. 
        
        The score_diff is the difference in score between the team of the player 
        who is completing the action minus the score of the opposing team. For
        example, if the score is 1-0, score_diff will be 1 for all players on 
        the current winning team and -1 for all players on the current losing
        team. If later in the game the score changes to 1-1, score_diff will be 
        set to 0 for all players.

        It is worth noting how goals are kept track of in Wyscout data. For 
        every goal, there will be one action with a 'goal' tag from the player
        who scored the goal. Immediately following, there is a second event that
        accounts for the goalie's save attempt which also has a 'goal' tag. In 
        this function, we make sure to not double count every goal by filtering 
        out events that have a eventId=9 for a save attempt. 
        
        Refer to the following link for the wyscout API which details tags and 
        their corresponding events https://support.wyscout.com/matches-wyid-events.

        Input:
            game (dataframe): dataframe for one game
        Output:
            game (dataframe): dataframe with 'score_diff' column included
        """
        scores = {}
        teams = np.unique(self.events[['teamId']].values)
        self.events['team_' + str(teams[0]) + '_score'] = 0
        self.events['team_' + str(teams[1]) + '_score'] = 0

        # filtering out save attempts (eventId=9) and own goals
        goals = self.events[(self.events['tags_Goal'] == 1) 
                & (self.events['eventId'] != 9) 
                & (self.events['tags_own_goal'] != 1)]
        for team in teams:
            # find the index of all goals for this team
            team_goals = goals[goals['teamId'] == team]['eventSec'].index
            goals_counter = 0

            # input the goal count for this team to every event 
            for ind in team_goals:
                goals_counter += 1
                self.events.loc[ind:,'team_'+ str(team) + '_score'] = goals_counter

        # add any own goals, if necessary
        self.own_goals(teams)

        # compute the score difference for each team at every event
        self.get_score_diff(teams)

        # only keep the score_diff columns                                     
        for team in teams:
            self.events = self.events.drop(['team_'+ str(team) + '_score'], axis = 1)


    def normalize(self):
        """Normalize all numerical features in the events dataframe."""
        self.events['x1'] = self.events['x1']/100
        self.events['y1'] = self.events['y1']/100
        self.events['x2'] = self.events['x2']/100
        self.events['y2'] = self.events['y2']/100
        self.events['height'] = (self.events['height'] - self.wyscout.height_min)/(self.wyscout.height_max 
                                                                           - self.wyscout.height_min)
        self.events['weight'] = (self.events['weight'] - self.wyscout.weight_min)/(self.wyscout.weight_max 
                                                                           - self.wyscout.weight_min)
        self.events['age'] = (self.events['age'] - self.wyscout.age_min)/(self.wyscout.age_max 
                                                                  - self.wyscout.age_min)


    def process_game(self, games_dir, original_IDs = False):
        """Call all functions necessary to process the current game. 

        This includes various operations such as adding x and y columns, player 
        height, weight, age, score differential, one hot encoding tags, and zero
        indexing subevents. Penalties are excluded since the event sequence of 
        penalties differs significantly from the rest of the game. All numerical 
        features are normalized and columns are ordered such that categorical 
        and numerical features are grouped together.

        Input:
            original_IDs: if True, player IDs are left as is. If False, player 
                          IDs are zero indexed. This can be set to False if
                          using all players instead of the top 25% quantile. 
        """
        f = os.path.join(games_dir, self.game_path)
        data = pd.json_normalize(json.load(open(f)))
        self.events = pd.DataFrame(data['events'][0])

        # add x and y positions
        self.get_xy_cols()

        # add height, weight, age, and footedness
        self.get_player_vals()

        # one hot encode tags
        self.one_hot_encode_tags()

        # fix null subevents
        self.fix_null_subevents()
        # 0 index subevents
        self.events = self.events.replace({'subEventId': self.wyscout.subeventID0index})

        # 0 index playerId (if original IDs are not necessary)
        if not original_IDs:
            self.events = self.events.replace({'playerId': self.wyscout.playerID0index})

        # 0 index matchPeriod
        self.events = self.events.replace({'matchPeriod': self.wyscout.matchPeriod0index})

        # add "score_diff" column to df
        self.get_score()

        # drop unecessary features
        self.events = self.events.drop(columns = ['id','matchId','positions', 
                                                  'tags','subEventName',
                                                  'eventName', 'eventId',
                                                  'teamId', 'eventSec'])

        # drop events that occur during penalties
        self.events = self.events[self.events['matchPeriod'] != 'P']

        # switch column order so that categorical features are first, 
        # followed by numerical features
        cols = list(self.events.columns)
        cols[0], cols[1], cols[2] = cols[2], cols[0], cols[1]
        self.events = self.events[cols]

        # normalize numerical features
        self.normalize()

        # fill na values to 0 for now
        self.events = self.events.fillna(0)


def main(argv):
    """Create a Wyscout object and process all games in the directory."""
    wy_ob = Wyscout(argv.games_dir, "eventid2name.csv", "players.json", "tags2name.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games_dir', default='wyscout-soccer-match-event-dataset'\
                                     '/processed/files', 
                                     type=str,
                                     help='Folder where all processed game files'\
                                          ' are stored')
    main(parser.parse_args())
