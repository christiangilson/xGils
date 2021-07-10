# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


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

    return df_shots
