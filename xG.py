# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.calibration import calibration_curve

import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn


def create_eventId(df):
    """
    Sorts all events, then provides a unique row identifier in the order that the event occurred within a match
    """

    df = df.sort_values(['competition','season','matchId','periodId','gameTime'], ascending=[True,True,True,True,True])\
            .reset_index(drop=True)

    return df.index.values + 1


def possession_indicator(df):
    """
    Function which identifies which team is in possession.

    NaN's will be forward filled, so we just need to provide very clear sides that an event is on,
    and the forward fill will work in between.

    """

    # team identifiers
    teamId = df['playerTeamId']
    homeTeamId = df['homeTeamId']
    awayTeamId = df['awayTeamId']
    teams = set([homeTeamId,awayTeamId])
    otherTeamId = list(teams - set([teamId]))[0]

    # events & subevents
    eventType = df['eventType']
    eventSubType = df['eventSubType']

    # assigning possessionTeamId
    ## Basically picking the things in DEFENCE that should actually be the TEAMID in possession
    ## (But including Pass because of it's prevelence to reduce having to go further down the loop)
    if eventSubType in ['Pass','Ball Recovery','Catch','Clearance','Interception','Ball Claim']:
        possessionTeamId = teamId

    # picking things in ATTACK that should actually be seen as the OTHERTEAMID in possession
    elif eventSubType in ['Lost Aerial Duel','Lost Possession']:
        possessionTeamId = otherTeamId

    elif eventSubType in  ['Aerial Duel']:
        possessionTeamId = np.NaN

    elif eventType in ['attack','shot']:
        possessionTeamId = teamId

    elif eventType in ['defence', 'press']:
        possessionTeamId = otherTeamId

    else:
        possessionTeamId = np.NaN

    return possessionTeamId



def xG_contextual_feature_engineering(df):

    # 1) producing eventId
    df['eventId'] = create_eventId(df)

    # 2) producing possessionTeamId marker -> quite a few other advanced features hang off of this
    df['possessionTeamId'] = df.apply(possession_indicator, axis=1)
    # forward filling NaNs
    df['possessionTeamId'] = df.possessionTeamId.fillna(method='ffill')
    # converting to int
    df['possessionTeamId'] = df['possessionTeamId'].astype(int)

    # 3) Sequencing the possessions (each possession  will have it's own index per match)
    #print ('Applying possessionSequenceIndex...')
    ## initiate sequence at 0
    df['possessionSequenceIndex'] = 0
    ## every time there's a change in sequence (or a change in half), you set a value of 1
    df.loc[( (df['possessionTeamId'] != df['possessionTeamId'].shift(1)) | (df['periodId'] != df['periodId'].shift(1)) | (df['matchId'] != df['matchId'].shift(1)) ), 'possessionSequenceIndex'] = 1
    ## take a cumulative sum of the 1s per match
    df['possessionSequenceIndex'] = df.groupby('matchId')['possessionSequenceIndex'].cumsum()

    # 4) Getting the time that the team has been in possession until the pass has been made (1) takes a while, but allows 2) to be vectorised)
    #print ('Applying possessionStartSec...')
    ## getting the time since the possession started
    df['possessionStartTime'] = df.loc[df.groupby(['matchId','possessionSequenceIndex'])['timeStamp'].transform('idxmin'), 'timeStamp'].values
    ## calculating the time of the posession
    df['possessionTimeSec'] = (df['timeStamp'] - df['possessionStartTime']) / pd.Timedelta(1, 's')


    # 5) Getting the time that the player has been in possession
    #print ('Applying playerPossessionTimeSec...')
    ## 1) initialising at 0
    df['playerPossessionTimeSec'] = 0
    ## 2) checks that the previous event was part of the same possession sequence within the same match, and if it is, calculates possession time in seconds
    df.loc[( (df['matchId'] == df['matchId'].shift(1)) & (df['possessionSequenceIndex'] == df['possessionSequenceIndex'].shift(1)) ), 'playerPossessionTimeSec'] = df['possessionTimeSec'] - df['possessionTimeSec'].shift(1)

    ################################################################################################
    ################################################################################################

    # 6) Game State (The +/- Number of Goals)
    # print ('Applying gameState...')
    ## getting goals scored flag
    df['goalScoredFlag'] = df.eventSubType.apply(lambda x: 1 if x == 'Goal' else 0)

    # querying goals
    df_goals = df.loc[df['goalScoredFlag'] == 1, ['matchId','eventId','playerTeamId']]

    # a list of eventId's that occur right after a goal for the other team that we'll be added a conceded flag
    lst_concededEventId = []

    # this is basically a really ugly cross apply
    for idx, cols in df_goals.iterrows():
        matchId, eventId, teamId = cols
        try:
            concededEventId = df.loc[(df['matchId'] == matchId) & (df['eventId'] > eventId) & \
                                     (df['playerTeamId'] != teamId)]\
                                .sort_values('eventId', ascending=True)\
                                .head(100)['eventId'].values[0]

            # appending eventId to list
            lst_concededEventId.append(concededEventId)
        except:
            continue

    # setting goals conceded flag
    df['goalsConcededFlag'] = 0
    df.loc[(df['eventId'].isin(lst_concededEventId)), 'goalsConcededFlag'] = 1

    ## Cumulatively summing the goals scored
    df['goalsScored'] = df.sort_values(['matchId','periodId','timeStamp'], ascending=[True, True, True])\
                                        .groupby(['matchId','playerTeamId'])\
                                        ['goalScoredFlag'].cumsum()

    ## Cumulatively summing the goals conceded
    df['goalsConceded'] = df.sort_values(['matchId','periodId','timeStamp'], ascending=[True, True, True])\
                                        .groupby(['matchId','playerTeamId'])\
                                        ['goalsConcededFlag'].cumsum()

    ## Calculating the goal delta
    df['goalDelta'] = df['goalsScored'] - df['goalsConceded']

    ################################################################################################
    ################################################################################################

    # 7) Number Red Cards (Very similar method above)
    # print ('Applying numReds...')
    ## Applying red card flag
    df['redCardFlag'] = df.eventSubType.apply(lambda x: -1 if x == 'Red Card' else 0)

    ## Applying Excess Player flag to the other team
    df_reds = df.loc[df['redCardFlag'] == -1, ['matchId','eventId','playerTeamId']]

    lst_redEventId = []

    for idx, cols in df_reds.iterrows():
        matchId, eventId, teamId = cols
        try:
            redEventId = df.loc[(df['matchId'] == matchId) & (df['eventId'] > eventId) & \
                                     (df['playerTeamId'] != teamId)]\
                                .sort_values('eventId', ascending=True)\
                                .head(100)['eventId'].values[0]

            lst_redEventId.append(redEventId)
        except:
            continue

    df.loc[df['eventId'].isin(lst_redEventId), 'redCardFlag'] = 1

    ## Cumulatively summing the number of red cards on a team throughout a game
    df['numReds'] = df.sort_values(['matchId','periodId','timeStamp'], ascending=[True, True, True])\
                                    .groupby(['matchId','playerTeamId'])\
                                    ['redCardFlag'].cumsum()

    return df[['competition', 'season', 'seasonIndex', 'gameMonthIndex', 'matchId',
       'playerId', 'playerName', 'position', 'detailedPosition',
       'playerTeamId', 'minsPlayed', 'subIn', 'subOut',
       'replacedReplacingPlayerId', 'booking', 'eventId', 'eventType', 'eventSubType',
       'eventTypeId', 'x1', 'y1', 'x2', 'y2', 'gameTime', 'timeStamp',
       'periodId', 'homeTeamName', 'homeTeamId', 'awayTeamName', 'awayTeamId',
       'kickOffDateTime', 'minute', 'second', 'x1_m', 'y1_m', 'x2_m', 'y2_m',
       'possessionTeamId', 'possessionSequenceIndex',
       'possessionStartTime', 'possessionTimeSec', 'playerPossessionTimeSec',
       'goalDelta', 'numReds','goalScoredFlag','xT']].copy()


def xG_geometric_shot_feature_engineering(df):
    """
    Calculating angles and distances.

    Intentionally not including data after the shot is taken (i.e. involving final shot position.)
    """

    # filtering shots
    df_shots = df.loc[df['eventType'] == 'shot'].reset_index(drop=True).copy()

    ## getting the x-dimension distance (and squared distance) to goal
    df_shots['x_dist_goal'] = 105 - df_shots['x1_m']
    df_shots['x_dist_goal_2'] = df_shots['x_dist_goal']**2

    ## getting some central y stats and squared stats (same definitions as in David Sumpter's code from MSc Mathematical Modelling of Football course at Uppsala University)
    df_shots['c1_m'] = abs(df_shots['y1_m'] - 34)
    df_shots['c1_m_2'] = df_shots['c1_m']**2

    ## getting distance to goal
    df_shots['vec_x'] = df_shots['x_dist_goal']
    df_shots['vec_y'] = 34 - df_shots['c1_m']
    df_shots['D'] = np.sqrt(df_shots['vec_x']**2 + df_shots['vec_y']**2)
    df_shots['Dsquared'] = df_shots.D**2
    df_shots['Dcubed'] = df_shots.D**3

    ## DQ step: getting rid of events where the vec_x = vec_y = 0 (look like data errors)
    df_shots = df_shots.loc[~((df_shots['vec_x'] == 0) & (df_shots['vec_y'] == 0))].copy()

    ## calculating passing angle in radians
    df_shots['a'] = np.arctan(df_shots['vec_x'] / abs(df_shots['vec_y']))

    ## calculating shooting angle from initial position
    df_shots['aShooting'] = np.arctan(7.32 * df_shots['x_dist_goal'] / (df_shots['x_dist_goal']**2 + df_shots['c1_m']**2 - (7.32/2)**2))
    df_shots['aShooting'] = df_shots.aShooting.apply(lambda x: x+np.pi if x<0 else x)


    ## getting number of defensive players applying pressure to the ball
    ## this takes a long  time to run...
    # storing list of counts of pressures on shots
    lst_pressureCountOnShot = []

    # getting list of all shot eventId's
    shot_eventIds = df_shots.eventId.values

    # iterating through shots
    for n, shot in enumerate(shot_eventIds):

        if n % 1000 == 0:
            print (f'Processed {n} shots out of {len(shot_eventIds)}')

        # getting shot meta to query the full event dataframe with: matchId,  teamId, timestamp
        matchId, teamId, timeStamp = df_shots.loc[df_shots['eventId'] == shot, ['matchId','playerTeamId','timeStamp']].values[0]

        # producing a dataframe of pressures on each shot, where the pressure must be coming from a player on the other team to the one taking a shot
        # and of course must be happening in the same match
        # and has been recorded two seconds either side of the shot (this is the timedelta logic)
        ## THIS IS A VERY EXPENSIVE QUERY TO RUN
        df_shotPressure = df.loc[(df['eventSubType'] == 'Pressure on Shot') & (df['playerTeamId'] != teamId) & (df['matchId'] ==  matchId) &\
                      (df['timeStamp'] > timeStamp - pd.Timedelta(2,'s')) & (df['timeStamp'] < timeStamp + pd.Timedelta(2,'s'))]

        # and we set the pressureCountOnShot metric to simply be the count of the players that are applying pressure to the shot
        lst_pressureCountOnShot.append(len(df_shotPressure))

    # updating df_shots dataframe all at once
    df_shots['pressureCountOnShot'] = lst_pressureCountOnShot


    ## and finally generating a penalty flag (takes about a minute)
    # storing  0/1 list of penalties
    df_shots['penaltyFlag'] = 0

    # getting subset of of all shot eventId's that could be penalties due to the location of the shot (92.925,34.0)
    pen_eventIds = df_shots.loc[(df_shots['x1_m'] == 92.925) & (df_shots['y1_m'] == 34.000)].eventId.values

    # iterating through shots
    for shot in pen_eventIds:

        # getting shot meta to query the full event dataframe with: matchId,  teamId, timestamp
        matchId, teamId, timeStamp = df_shots.loc[df_shots['eventId'] == shot, ['matchId','playerTeamId','timeStamp']].values[0]

        # producing a dataframe of pressures on each shot, where the pressure must be coming from a player on the other team to the one taking a shot
        # and of course must be happening in the same match
        # and has been recorded 5 minutes before the shot
        df_penalty = df.loc[(df['playerTeamId'] != teamId) & (df['matchId'] ==  matchId) &\
                      (df['timeStamp'] > timeStamp - pd.Timedelta(300,'s')) & (df['timeStamp'] < timeStamp) &\
                      (df['eventSubType'].isin(['Conceded Penalty','Foul for Penalty']))]

        # and we set the pressureCountOnShot metric to simply be the count of the players that are applying pressure to the shot
        df_shots.loc[df_shots['eventId'] == shot, 'penaltyFlag'] = 1 if len(df_penalty) > 0 else 0

    return df_shots


def xG_geometric_synthetic_shot_feature_engineering(df):
    """
    Calculating angles and distances.

    Intentionally not including data after the shot is taken (i.e. involving final shot position.)
    """

    # filtering shots
    df_shots = df.loc[df['eventType'] == 'shot'].reset_index(drop=True).copy()

    ## getting the x-dimension distance (and squared distance) to goal
    df_shots['x_dist_goal'] = 105 - df_shots['x1_m']
    df_shots['x_dist_goal_2'] = df_shots['x_dist_goal']**2

    ## getting some central y stats and squared stats (same definitions as in David Sumpter's code from MSc Mathematical Modelling of Football course at Uppsala University)
    df_shots['c1_m'] = abs(df_shots['y1_m'] - 34)
    df_shots['c1_m_2'] = df_shots['c1_m']**2

    ## getting distance to goal
    df_shots['vec_x'] = df_shots['x_dist_goal']
    df_shots['vec_y'] = 34 - df_shots['c1_m']
    df_shots['D'] = np.sqrt(df_shots['vec_x']**2 + df_shots['vec_y']**2)
    df_shots['Dsquared'] = df_shots.D**2
    df_shots['Dcubed'] = df_shots.D**3

    ## DQ step: getting rid of events where the vec_x = vec_y = 0 (look like data errors)
    df_shots = df_shots.loc[~((df_shots['vec_x'] == 0) & (df_shots['vec_y'] == 0))].copy()

    ## calculating passing angle in radians
    df_shots['a'] = np.arctan(df_shots['vec_x'] / abs(df_shots['vec_y']))

    ## calculating shooting angle from initial position
    df_shots['aShooting'] = np.arctan(7.32 * df_shots['x_dist_goal'] / (df_shots['x_dist_goal']**2 + df_shots['c1_m']**2 - (7.32/2)**2))
    df_shots['aShooting'] = df_shots.aShooting.apply(lambda x: x+np.pi if x<0 else x)

    return df_shots



# applying basic, added, advanced, and synthetic models to test data
def apply_xG_model_to_test(df_shots_test, models):
    """
    Applying the four different logistic regression models to produce four xG values
    """
    log_basic, log_added, log_adv, log_syn, log_adv_on_syn = models

    print ('Applying models...')
    df_shots_test['xG_basic'] = log_basic.predict(df_shots_test)
    df_shots_test['xG_added'] = log_added.predict(df_shots_test)
    df_shots_test['xG_adv'] = log_adv.predict(df_shots_test)
    df_shots_test['xG_syn'] = log_syn.predict(df_shots_test)
    df_shots_test['xG_adv_on_syn'] = log_adv_on_syn.predict(df_shots_test)
    print (f'Done applying {len(models)} models.')

    return df_shots_test


def plot_calibration_curve(df_shots_test, numBins=25, alpha=0.6, saveOutput=0, plotName='xG_calibration_plot', calibrationType='uniform'):
    """
    Calibration plots for xG models
    """
    fig = plt.figure(figsize=(10, 15))

    # splitting figure into two subplots
    gs = fig.add_gridspec(ncols=1, nrows=2, height_ratios=(2/3, 1/3))

    # defining axes of subplots
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # getting colourbline palette
    palette = seaborn.color_palette('colorblind', 6).as_hex()

    # Plotting perfect calibration (line y=x)
    ax1.plot([0, 1], [0, 1], 'k:', label='Perfectly Calibrated Model', alpha=alpha)

    # FOUR calibration curves - Tricky to plot all four at a time, so just do a Simple Vs Advanced
    ## 1) Simple Model
    fraction_of_positives, mean_predicted_value = calibration_curve(df_shots_test.goalScoredFlag, df_shots_test.xG_basic, n_bins=numBins, strategy=calibrationType)
    ax1.plot(mean_predicted_value, fraction_of_positives, marker="o", markersize=10, label='Basic Model', alpha = alpha, lw=1, color=palette[4])

    ## 2) Added Model
    fraction_of_positives, mean_predicted_value = calibration_curve(df_shots_test.goalScoredFlag, df_shots_test.xG_added, n_bins=numBins, strategy=calibrationType)
    ax1.plot(mean_predicted_value, fraction_of_positives, marker="o", markersize=10, label='Added Features', alpha = alpha, lw=2, color=palette[2])

    ## 3) Advanced Model: Canonical (Logit) Link function
    fraction_of_positives, mean_predicted_value = calibration_curve(df_shots_test.goalScoredFlag, df_shots_test.xG_adv, n_bins=numBins, strategy=calibrationType)
    ax1.plot(mean_predicted_value, fraction_of_positives, marker="o", markersize=10, label='Advanced Features', alpha = alpha, lw=3, color=palette[1])

    ## 4) Advanced Model: Using Synthetic data to train BUT NOT TEST
    fraction_of_positives, mean_predicted_value = calibration_curve(df_shots_test.goalScoredFlag, df_shots_test.xG_syn, n_bins=numBins, strategy=calibrationType)
    ax1.plot(mean_predicted_value, fraction_of_positives, marker="o", markersize=10, label='Advanced Features + Synthetic Shots', alpha=alpha, lw=4, color=palette[0])

    ax1.set_title('Calibration Plot', fontsize=20, pad=10)
    ax1.set_ylabel('Fraction of Successful Test Shots', fontsize=16)
    ax1.set_xlabel('Mean xG', fontsize=16)

    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlim([-0.05, 1.05])

    ax1.legend(loc="lower right", fontsize=16)

    ax1.tick_params(labelsize=16)

    # now plotting histogram
    seaborn.distplot(df_shots_test.xG_syn, color=palette[0], label='Advanced Features + Synthetic Shots', kde=False, ax=ax2)

    ax2.set_title('Distribution of xG Test Predictions (Real Shots Only)', fontsize=20, pad=10)
    ax2.set_ylabel('Number of Test Shots', fontsize=16)
    ax2.set_xlabel('Predicted xG', fontsize=16)
    ax2.tick_params(labelsize=16)
    ax2.legend(fontsize=16)

    plt.tight_layout()

    if saveOutput == 1:
        plt.savefig(f'/Users/christian/Desktop/University/Birkbeck MSc Applied Statistics/Project/Plots/xG Calibration/{plotName}.pdf', dpi=300, format='pdf', bbox_inches='tight')

    return plt.show()


def calculate_model_metrics(df_shots_test, xGtype='xG_adv', log_reg_decision_threshold = 0.5):
    """
    Applies Logistic Regression Decision Threshold (i.e. applying the model to attribute whether a pass would or would have not been successful)
    And calculates a bunch of related metrics
    """

    df_shots_test['predictedSuccess'] = df_shots_test[xGtype].apply(lambda x: 1 if x > log_reg_decision_threshold else 0)

    brierScore = metrics.brier_score_loss(df_shots_test.goalScoredFlag, df_shots_test[xGtype])

    # strongly advised by https://github.com/CleKraus/soccer_analytics/blob/master/notebooks/expected_goal_model_lr.ipynb
    logLossScore = metrics.log_loss(df_shots_test.goalScoredFlag, df_shots_test[xGtype])

    # precision = TRUE POSITIVE / (TRUE POSITIVE + FALSE POSITIVE)
    # ratio of correctly positive observations / all predicted positive observations
    precisionScore = metrics.precision_score(df_shots_test.goalScoredFlag, df_shots_test.predictedSuccess)

    # recall = TRUE POSITIVE / (TRUE POSITIVE + FALSE NEGATIVE)
    # ratio of correctly positive observations / all true positive observations (that were either correctly picked TP or missed FN)
    recallScore = metrics.recall_score(df_shots_test.goalScoredFlag, df_shots_test.predictedSuccess)

    # weighted average of precision and recall
    f1Score = metrics.f1_score(df_shots_test.goalScoredFlag, df_shots_test.predictedSuccess)

    AUCScore = metrics.roc_auc_score(df_shots_test.goalScoredFlag, df_shots_test[xGtype])

    # overall accuracy score: ratio of all correct over count of all observations
    accuracyScore = metrics.accuracy_score(df_shots_test.goalScoredFlag, df_shots_test.predictedSuccess)

    return print (f'LogLoss Score: {logLossScore}\n\nBrier Score: {brierScore}\n\nPrecision Score: {precisionScore}\n\nRecall Score: {recallScore}\n\nF1 Score: {f1Score}\n\nAUC Score: {AUCScore}\n\nAccuracyScore: {accuracyScore}')
