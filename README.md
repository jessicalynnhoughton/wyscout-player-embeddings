# wyscout-player-embeddings
European football player embedding analysis using LSTM on the publicly available Wyscout dataset. 

This project was completed with equal contribution from Jessica Houghton and Sasha Yousefi, along with mentorship from Vitor Lopes, the Director of Applied Research at Real Madrid Club de Fútbol. 
Please do not hesitate to contact @jessyhoughton@gmail.com and @sashayousefi@gmail.com for comments and questions about this project. 

## How Can Player Vectors Help Real Madrid and other football clubs?
- Enhanced scouting capabilities: Real Madrid can specifically target
players who align with a desired playing style or hole in their roster
- Cost-effective transfers: Real Madrid can identify undervalued or
lesser-known players who possess desired characteristics and potential,
while saving money on transfer fees
- Match preparation: Playing vectors can help identify characteristics and
tendencies of opponents, allowing Real Madrid to tailor their strategy to
exploit specific aspects of the other team’s game

## Goal
### Football clubs must optimize their budget by understanding their existing players and available players in the market
- Currently, insights on playing style relies heavily on subjective opinions
from coaches and football experts
- Instead, player vectors can be constructed by analyzing the actions
performed by each player on the field
- Machine learning models can analyze the patterns and relationships
between players to further enhance the objective measure of playing style
- By utilizing this data-driven approach, a comprehensive and unbiased
estimation of playing style can be achieved

## Approach 
### LSTM with Entity Embeddings
- Keras LSTM model to predict the next subevent in the sequence of subevents
  - Incorporates categorical variables (subevent and playerid) & numerical
variables for additional context
  - LSTM performs sequential analysis and feature extraction on the
combined inputs, enabling the model to capture complex patterns and
dependencies within the game data.
- Entity embeddings for the categorical variables are learned through model training
by means of weight updates
  - Embedding layer learns to represent each player & subevent in a continuous
vector space, capturing inherent relationships between categories


#### Inputs
  - Categorical variables - subevents, players
  - Numerical variables - match period, weight, age, height, xy coordinates, tags
(i.e. yellowcard, ball lost, counterattack).
#### Data Structure
  - Use the previous 5 timesteps to predict the next event
<img width="308" alt="Screen Shot 2024-02-26 at 1 27 11 PM" src="https://github.com/jessicalynnhoughton/wyscout-player-embeddings/assets/60555310/1178ed85-09f1-4ce4-9360-38eb039e66bd">


### Player Embedding Analysis
#### Primary Objective: Can player positions (goalie, defender, midfielder, forward) be accurately determined solely by the extracted player embeddings from the model?
This objective will be approached through three methods:
1. Training a simple logistic regression on player embeddings to predict player position
2. Visualizing player embeddings and positions in both 2D and 3D using TSNE dimensional reduction.
3. Identifying the most numerically similar players to those of interest through PCA dimensional reduction and Euclidean/Manhattan distance metrics.

## Files
### data_formatting.py
This file preprocesses all game, player, and event data for the entire Wyscout dataset in the "processed" folder from [koenvo's wyscout-soccer-match-event-dataset](https://github.com/koenvo/wyscout-soccer-match-event-dataset). Once run, all games are stored in the processed_games folder with the data necessary to run the model. 
### models.ipynb
This file prepares the data for input into the LSTM and proceeds to train, save, and evaluate the LSTM model.
### embeddings.ipynb
This file extracts the player embeddings from the trained LSTM model and evaluates the embeddings using logistic regression, TSNE visualization, and PCA minimum distance analysis. 


