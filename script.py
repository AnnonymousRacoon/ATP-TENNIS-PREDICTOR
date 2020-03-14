
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
plt.rcParams["figure.figsize"] = (8,5)

# ________________________________________________________________
# LOAD AND INVESTIGATE DATA
# ----------------------------------------------------------------
PlayerStats = pd.read_csv('tennis_stats.csv')

# Add Win/Loss ratio
PlayerStats['Win-Loss Ratio'] = PlayerStats.apply(lambda r : float(r.Wins) / r.Losses if r.Losses > 0 else 0, axis = 1)
# View available stats
print('Available Stats:\n')
for stat in PlayerStats.columns.values:
    print(stat)

# Group assumes values are either related to player ID, Variables,or performance indicators 
PlayerID = ['Player','Year']
Variables = ['FirstServe','FirstServePointsWon',
 'FirstServeReturnPointsWon','SecondServePointsWon',
 'SecondServeReturnPointsWon','Aces','BreakPointsConverted',
 'BreakPointsFaced','BreakPointsOpportunities','BreakPointsSaved',
 'DoubleFaults','ReturnGamesPlayed','ReturnGamesWon','ReturnPointsWon',
 'ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
 'TotalServicePointsWon','Win-Loss Ratio']
PerformanceIndicators = ['Wins','Winnings','Ranking']

# ________________________________________________________________
# EXPLORATORY ANALYSIS
# ----------------------------------------------------------------
# #UN-COMMENT TO PLOT GRAPHS FOR ALL RELATIONSHIPS

# for var in Variables:
#     sns.set_style('darkgrid')
#     sns.set_palette('muted')
   
#     plt.subplots(2,2,figsize=(14,14))
# # RANKING
#     ax = plt.subplot(2, 2, 1)
#     ax.scatter(PlayerStats[var],PlayerStats.Ranking,alpha = 0.5)
#     ax.set_xlabel(var)
#     ax.set_ylabel("Ranking")
#     ax.set_title("Player Ranking against {}".format(var)) 
# # WINNINGS
#     ax2 = plt.subplot(2, 2, 2)
#     ax2.scatter(PlayerStats[var],PlayerStats.Winnings,alpha = 0.5)
#     ax2.set_xlabel(var)
#     ax2.set_ylabel("Winnings")
#     ax2.set_title("Player Winnings against {}".format(var))
# # WIN-LOSS RATIO
#     ax3 = plt.subplot(2, 2, 3)
#     ax3.scatter(PlayerStats[var],PlayerStats['Win-Loss Ratio'],alpha = 0.5)
#     ax3.set_xlabel(var)
#     ax3.set_ylabel('Win-Loss Ratio')
#     ax3.set_title("Player Win-Loss Ratio against {}".format(var))
# # WINS
#     ax4 = plt.subplot(2, 2, 4)
#     ax4.scatter(PlayerStats[var],PlayerStats.Wins,alpha = 0.5)
#     ax4.set_xlabel(var)
#     ax4.set_ylabel('Wins')
#     ax4.set_title("Player Wins against {}".format(var))


#     plt.show()

# ___________________________INITIAL THOUGHTS____________________________________
# From the graphs the following show a clear correlation with performance indicators:
# -Aces
# -Break Points Faced
# -Break Point Opportunities
# -Double Faults
# -Return Games Played
# -Return Games Won (Minor correlation)
# -Return Games Played (This is obvious, the more games you play the further into the competition you go)
# -Service Games Played (This is obvious, the more games you play the further into the competition you go)

# ________________________________________________________________
# SINGLE LINEAR REGRESSION
# ----------------------------------------------------------------

# Set the threshold for 'Good' Correlation:
Threshold = 0.3

# lists of 'Good' Correlations for predicting performance indicators:
# N.B in theory (and by experiment) performance indictors should correlate with each other so I've manually added them in
New_vars_wins = ['Winnings','Ranking']
New_vars_winnings = ['Wins','Ranking']
New_vars_ranking = ['Wins','Winnings']

# iterate through vars to see which are good predictors of each PI:
for var in Variables:
    for Indicator in PerformanceIndicators:
        x = PlayerStats[[var]]
        y = PlayerStats[Indicator]

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)

        # build regression model
        mlr = LinearRegression()
        mlr.fit(x_train,y_train)
        y_predict = mlr.predict(x_test)

        # plt.scatter(y_test,y_predict,alpha = 0.5)
        # plt.show()
        print('\nMODEL PERFORMACE FOR: {I} vs. {V}'.format(I = Indicator, V = var))
        print("Train score: {}". format(mlr.score(x_train, y_train)))
        print("Test score: {}". format(mlr.score(x_test, y_test)))

        if mlr.score(x_test, y_test) > Threshold:
            if Indicator == 'Wins':
                New_vars_wins.append(var)
            if Indicator == 'Winnings':
                New_vars_winnings.append(var)
            if Indicator == 'Ranking':
                New_vars_ranking.append(var)
        # Give ranking a reduced threshold
        if mlr.score(x_test, y_test) > Threshold/3 and Indicator == 'Ranking':
            New_vars_ranking.append(var)


print('\nGOOD PREDICTORS FOR WINS ARE:')
for i in New_vars_wins:
    print('-{}'.format(i))
print('\nGOOD PREDICTORS FOR WINNINGS ARE:')
for i in New_vars_winnings:
    print('-{}'.format(i))
print('\nGOOD PREDICTORS FOR RANKINGS ARE:')
for i in New_vars_ranking:
    print('-{}'.format(i))


# __________________________________THOUGHTS____________________________________
# Wins are the easiest to predict, with winnings not that much harder
# Rankings are incredibly hard to predict with a lower threshold require to yield any 'good correlators'


# ________________________________________________________________
# MULTIPLE LINEAR REGRESSION
# ----------------------------------------------------------------

# use our good correlators to predict PIs:
for Indicator, Vars in zip(PerformanceIndicators, [New_vars_wins, New_vars_winnings,New_vars_ranking]):
    x = PlayerStats[Vars]
    y = PlayerStats[Indicator]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

    # build regression model
    mlr = LinearRegression()
    mlr.fit(x_train,y_train)
    y_predict = mlr.predict(x_test)

    # Plot Datapoints
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    ax = plt.subplot()
    plt.scatter(y_test,y_predict,alpha = 0.5)
    ax.set_xlabel('{} Test Values'.format(Indicator))
    ax.set_ylabel('{} Predicted Values'.format(Indicator))
    ax.set_title("Optimised Test vs. Predicted Values for {I} in Tennis\nTest score: {S}".format(I = Indicator,S = mlr.score(x_test, y_test)))
    plt.savefig("Optimised Test vs. Predicted Values for {} in Tennis.png".format(Indicator))
    plt.show()
    print('\nOPTIMISED MODEL PERFORMACE FOR: {I}'.format(I = Indicator))
    print("Train score: {}". format(mlr.score(x_train, y_train)))
    print("Test score: {}". format(mlr.score(x_test, y_test)))

# __________________________________THOUGHTS____________________________________
# -Improvements for Wins and Winnings
# -Rankings still very hard to predict from the current data
# -the Model Predicts negative rankings, which is impossible
# -Other Performance indicators were not considered as variables, but in theory they should correleate with each other.
# -Adding them in improves the test score by about 3-4% 

