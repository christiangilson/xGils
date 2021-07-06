# -*- coding: utf-8 -*-


def apply_datetimes(df):

    df['kickOffDateTime'] = pd.to_datetime(df['kickOffDateTime'])
    df['timeStamp'] = pd.to_datetime(df['timeStamp'])

    return df


def create_game_month_index(df):
    """

    In order to cycle through the data month by month using the Beta-Binomial conjugate analysis, will need to assign indices for the seasons and the months that will enable us to cyle through as part of the systematic updating

    """

    months = df.timeStamp.dt.month
    years = df.timeStamp.dt.year

    idx = (years*12) + months

    df['gameMonthIndex'] = idx

    return df


def opta_infer_dribble_end_coords(df):
    """
    There are two dribble related actions: "Dribble" and "Failed Dribble"

    (Successful) Dribbles are followed up by the action that that player did next

    Failed Dribbles are followed up usually by a tackle and a ball recovery on the defensive team

    With Successful Dribble's we're going to infer the end co-ords of the dribble based on the starting position of what the player did next

    With Failed Dribble, we're going to model them as not having moved (this looks to be how Opta model it implicitly)

    KEY NOTE: Even though "Ball Recovery" is a DEFENCE type event, it's co-ordinates are w.r.t. ATTACKING frame of reference

    """

    # failed dribbles
    df['x2'] = df.apply(lambda x: x.x1 if x.eventSubType == 'Failed Dribble' else x.x2, axis=1)
    df['y2'] = df.apply(lambda x: x.y1 if x.eventSubType == 'Failed Dribble' else x.y2, axis=1)

    # successful dribbles: looking at the next, next+1, and next+2 actions
    df['x2_1'] = None
    df['y2_1'] = None
    df['x2_2'] = None
    df['y2_2'] = None
    df['x2_3'] = None
    df['y2_3'] = None

    df['x2_same_attack'] = None
    df['y2_same_attack'] = None
    df['x2_same_recovery'] = None
    df['y2_same_recovery'] = None
    df['x2_opp_recovery'] = None
    df['y2_opp_recovery'] = None
    df['x2_same_defence'] = None
    df['y2_same_defence'] = None
    df['x2_opp_defence'] = None
    df['y2_opp_defence'] = None
    df['x2_opp_attack'] = None
    df['y2_opp_attack'] = None

    # next (same player)
    df['x2_1'][((df['playerId'] == df['playerId'].shift(-1)) & (df['eventSubType'] == 'Dribble'))] = df['x1'].shift(-1)
    df['y2_1'][((df['playerId'] == df['playerId'].shift(-1)) & (df['eventSubType'] == 'Dribble'))] = df['y1'].shift(-1)

    # next+1 (same player)
    df['x2_2'][((df['playerId'] == df['playerId'].shift(-2)) & (df['eventSubType'] == 'Dribble'))] = df['x1'].shift(-2)
    df['y2_2'][((df['playerId'] == df['playerId'].shift(-2)) & (df['eventSubType'] == 'Dribble'))] = df['y1'].shift(-2)

    # next+2 (same player)
    df['x2_3'][((df['playerId'] == df['playerId'].shift(-3)) & (df['eventSubType'] == 'Dribble'))] = df['x1'].shift(-3)
    df['y2_3'][((df['playerId'] == df['playerId'].shift(-3)) & (df['eventSubType'] == 'Dribble'))] = df['y1'].shift(-3)

    # next (any same team player, any attacking event)
    df['x2_same_attack'][((df['playerTeamId'] == df['playerTeamId'].shift(-1)) & (df['eventType'].shift(-1).isin(['attack','shot'])) & (df['eventSubType'] == 'Dribble'))] = df['x1'].shift(-1)
    df['y2_same_attack'][((df['playerTeamId'] == df['playerTeamId'].shift(-1)) & (df['eventType'].shift(-1).isin(['attack','shot'])) & (df['eventSubType'] == 'Dribble'))] = df['y1'].shift(-1)

    # next (any same team player, ball recovery)
    df['x2_same_recovery'][((df['playerTeamId'] == df['playerTeamId'].shift(-1)) & (df['eventSubType'].shift(-1) == 'Ball Recovery') & (df['eventSubType'] == 'Dribble'))] = df['x1'].shift(-1)
    df['y2_same_recovery'][((df['playerTeamId'] == df['playerTeamId'].shift(-1)) & (df['eventSubType'].shift(-1) == 'Ball Recovery') & (df['eventSubType'] == 'Dribble'))] = df['y1'].shift(-1)

    # next (any opposition player, ball recovery)
    df['x2_opp_recovery'][((df['playerTeamId'] != df['playerTeamId'].shift(-1)) & (df['eventSubType'].shift(-1) == 'Ball Recovery') & (df['eventSubType'] == 'Dribble'))] = df['x1'].shift(-1)
    df['y2_opp_recovery'][((df['playerTeamId'] != df['playerTeamId'].shift(-1)) & (df['eventSubType'].shift(-1) == 'Ball Recovery') & (df['eventSubType'] == 'Dribble'))] = df['y1'].shift(-1)

    # next (any same team player, any defensive event)
    df['x2_same_defence'][((df['playerTeamId'] == df['playerTeamId'].shift(-1)) & (df['eventType'].shift(-1).isin(['defence','press'])) & (df['eventSubType'] == 'Dribble'))] = df['x1'].shift(-1)
    df['y2_same_defence'][((df['playerTeamId'] == df['playerTeamId'].shift(-1)) & (df['eventType'].shift(-1).isin(['defence','press'])) & (df['eventSubType'] == 'Dribble'))] = df['y1'].shift(-1)

    # next (any opposition player, any defensive event)
    df['x2_opp_defence'][((df['playerTeamId'] != df['playerTeamId'].shift(-1)) & (df['eventType'].shift(-1).isin(['defence','press'])) & (df['eventSubType'] == 'Dribble'))] = df['x1'].shift(-1)
    df['y2_opp_defence'][((df['playerTeamId'] != df['playerTeamId'].shift(-1)) & (df['eventType'].shift(-1).isin(['defence','press'])) & (df['eventSubType'] == 'Dribble'))] = df['y1'].shift(-1)

    # next (any opposition player, any attacking event)
    df['x2_opp_attack'][((df['playerTeamId'] != df['playerTeamId'].shift(-1)) & (df['eventType'].shift(-1).isin(['attack','shot'])) & (df['eventSubType'] == 'Dribble'))] = df['x1'].shift(-1)
    df['y2_opp_attack'][((df['playerTeamId'] != df['playerTeamId'].shift(-1)) & (df['eventType'].shift(-1).isin(['attack','shot'])) & (df['eventSubType'] == 'Dribble'))] = df['y1'].shift(-1)

    # applying those x2 and y2's
    df['x2'] = df.apply(lambda x: x.x2 if x.eventSubType != 'Dribble' else\
                                  x.x2_1 if x.x2_1 != None else\
                                  x.x2_2 if x.x2_2 != None else\
                                  x.x2_3 if x.x2_3 != None else\
                                  x.x2_same_attack if x.x2_same_attack != None else\
                                  x.x2_same_recovery if x.x2_same_recovery != None else\
                                  100-x.x2_opp_recovery if x.x2_opp_recovery != None else\
                                  100-x.x2_same_defence if x.x2_same_defence != None else\
                                  x.x2_opp_defence if x.x2_opp_defence != None else\
                                  100-x.x2_opp_attack if x.x2_opp_attack != None else None, axis=1)

    df['y2'] = df.apply(lambda x: x.y2 if x.eventSubType != 'Dribble' else\
                                  x.y2_1 if x.y2_1 != None else\
                                  x.y2_2 if x.y2_2 != None else\
                                  x.y2_3 if x.y2_3 != None else\
                                  x.y2_same_attack if x.y2_same_attack != None else\
                                  x.y2_same_recovery if x.y2_same_recovery != None else\
                                  100-x.y2_opp_recovery if x.y2_opp_recovery != None else\
                                  100-x.y2_same_defence if x.y2_same_defence != None else\
                                  x.y2_opp_defence if x.y2_opp_defence != None else\
                                  100-x.y2_opp_attack if x.y2_opp_attack != None else None, axis=1)

    return df


def coords_in_metres(df, x1, x2, y1, y2, pitch_length = 105.0, pitch_width = 68.0):
    """
    Convert Opta co-ordinates from x in [0,100], y in [0, 100] to x' in [0, 105], y' in [0, 68]
    """

    df['x1_m'] = (df.x1 / 100.0) * pitch_length
    df['y1_m'] = (df.y1 / 100.0) * pitch_width
    df['x2_m'] = (df.x2 / 100.0) * pitch_length
    df['y2_m'] = (df.y2 / 100.0) * pitch_width

    # tidying up extra cols created to get the final dribble positions
    return df[['competition','season','seasonIndex','gameMonthIndex','matchId', 'playerId', 'playerName', 'position', 'detailedPosition','playerTeamId'\
               ,'minsPlayed', 'subIn', 'subOut','replacedReplacingPlayerId', 'booking'\
               ,'eventType', 'eventSubType','eventTypeId', 'x1', 'y1', 'x2', 'y2'\
               ,'gameTime', 'timeStamp','periodId', 'homeTeamName', 'homeTeamId', 'awayTeamName', 'awayTeamId','kickOffDateTime', 'minute', 'second', 'x1_m', 'y1_m', 'x2_m', 'y2_m']].copy()



def wyscout_coords_in_metres(df_wyscout, start_x, end_x, start_y, end_y, pitch_length = 105.0, pitch_width = 68.0):
    """
    Convert Wyscout co-ordinates from x in [0,100], y in [100,0] (y inverted w.r.t. Opta) to x' in [0,105], y' in [0,68]
    """

    # 1) firstly, getting rid of erronous data where co-ords fall outside of the Wyscout range
    df_wyscout = df_wyscout.loc[df_wyscout[start_y] <= 100].copy()
    df_wyscout = df_wyscout.loc[df_wyscout['end_y'] <= 100].copy()

    df_wyscout = df_wyscout.loc[df_wyscout[start_y] >= 0].copy()
    df_wyscout = df_wyscout.loc[df_wyscout[end_y] >= 0].copy()

    df_wyscout = df_wyscout.loc[df_wyscout[start_x] <= 100].copy()
    df_wyscout = df_wyscout.loc[df_wyscout[end_x] <= 100].copy()

    df_wyscout = df_wyscout.loc[df_wyscout[start_x] >= 0].copy()
    df_wyscout = df_wyscout.loc[df_wyscout[end_x] >= 0].copy()

    # 2) and then
    df_wyscout['x1_m'] = (df_wyscout.start_x / 100.0) * pitch_length
    df_wyscout['y1_m'] = ( (100.0 - df_wyscout.start_y) / 100.0) * pitch_width
    df_wyscout['x2_m'] = (df_wyscout.end_x / 100.0) * pitch_length
    df_wyscout['y2_m'] = ( (100.0 - df_wyscout.end_y) / 100.0) * pitch_width

    return df_wyscout


import pandas as pd
import numpy as np
