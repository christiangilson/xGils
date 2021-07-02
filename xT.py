# -*- coding: utf-8 -*-


def get_cell_indexes(x, y, l = 18, w = 12, pitch_length = 105.0, pitch_width = 68.0):
    """
    Vectorised transformation of (x,y) pitch co-ordinates into MxN zones

    Get the fraction of pitch length / width and then multiply by N / M to get the zone

    But that zone will be a float. Want to get the floor of the float (i.e. round down to the nearest integer),
    and make sure that the floored zone is between 0 and M-1 for widths, and 0 and N-1 for lengths using the clip method (.clip(lower bound, upper bound))

    Opta coords are x: [0,100] and y: [0,100], with the origin in the bottom left of the pitch

    Datatype note: x and y should be Pandas Series' objects
    """
    x_zone = (x / pitch_length) * l
    y_zone = (y / pitch_width) * w
    x_zone = x_zone.astype(int).clip(0, l - 1)
    y_zone = y_zone.astype(int).clip(0, w - 1)

    return x_zone, y_zone



def get_flat_indexes(x, y, l = 18, w = 12, pitch_length = 105.0, pitch_width = 68.0):
    """
    Collapsing the M, N indices to a single flat index, z

    N = l = length of pitch (x axis)
    M = w = width of pitch (y axis)

    Remember x and y indices go from 0 -> N-1 and 0 -> M-1

    Will be a unique number per zone

    0,0 -> 0,0 will have zone z = (12 - 1 - 0) * 18 + 0 = 198 (if N = 18 and M = 12)
    105,0 -> 17,0 will have zone z = (12 - 1 - 0) * 18 + 17 = 215 (if N = 18 and M = 12)

    0,68 -> 0,11 -> (12 - 1 - 11) * 18 + 0 -> 0
    105,68 -> 17,11 -> (12 - 1 - 11) * 18 + 17 -> 17

    So:
    * top left has z = 0
    * top right has z = 17
    * bottom left has z = 198
    * bottom right has z = 215

    So our MxN zones are indexed with the origin bottom left

    But our z indices start top left, and go left to right.
    """
    x_zone, y_zone = get_cell_indexes(x, y, l, w, pitch_length, pitch_width)

    # clever bit: this is ordered such that you can easily unpack into an MxN matrix
    return l * (w - 1 - y_zone) + x_zone



## Amazing function taken from Karun Singh implementation
def count(x, y, l = 18, w = 12, pitch_length = 105.0, pitch_width = 68.0):
    """
    Firstly getting rid of any NaN positions

    Then aggregating counts by zone

    Then transforming zone counts to MxN counts, to produce an MxN Numpy matrix

    Count the number of actions occurring in each cell of the grid, where the top left corner is the origin.
    """
    # ensuring that the counts are non-NULL
    x = x[~np.isnan(x) & ~np.isnan(y)]
    y = y[~np.isnan(x) & ~np.isnan(y)]

    # aggregating around the cell index and producing a count per index
    zone_counts = get_flat_indexes(x, y, l, w, pitch_length, pitch_width).value_counts(sort=False)

    # producing a vector of zeros of length m x n
    m_x_n_counts = np.zeros(w * l)

    # and then populating that m x n matrix, using the flat index, with the counts
    m_x_n_counts[zone_counts.index] = zone_counts

    # and now reshaping the vector of length m x n into a matrix with w=m rows (i.e. the width of the pitch) and l=n columns (i.e. the length of the pitch)
    return m_x_n_counts.reshape((w, l))



def safe_divide(a, b):
    """
    This function allows the safe division of one vector or matrix by another vector or matrix or float, whereby if you encounter an element-wise
    division that would result in a zero denominator, then you just return zero
    """

    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)



def p_score_if_shoot(df, successful_shot_events, failed_shot_events, event_column_name, l = 18, w = 12, pitch_length = 105.0, pitch_width = 68.0):
    """
    Takes in the events dataframe and extracts counts of shots and goals

    Those counts are then used to calculate the expected goals (xG) per zone

    This is a highly simplistic approach to calculating xG - sophistication can surely be added here?

    Outputs an M x N matrix of xG
    """
    all_shot_events = successful_shot_events + failed_shot_events

    df_shots = df.loc[df[event_column_name].isin(all_shot_events)]
    df_goals = df.loc[df[event_column_name].isin(successful_shot_events)]

    shot_matrix = count(df_shots.x1_m, df_shots.y1_m, l, w, pitch_length, pitch_width)
    goal_matrix = count(df_goals.x1_m, df_goals.y1_m, l, w, pitch_length, pitch_width)

    return safe_divide(goal_matrix, shot_matrix)



def get_df_all_moves(df, successful_pass_events, failed_pass_events, successful_dribble_events, failed_dribble_events, event_column_name):

    """
    Filter events dataframe to just the move events (successful + failed)
    """

    move_events = successful_pass_events + failed_pass_events + successful_dribble_events + failed_dribble_events

    return df.loc[df[event_column_name].isin(move_events)].copy()



def get_df_successful_moves(df, successful_pass_events, successful_dribble_events, event_column_name):
    """
    Filter events dataframe to just the successful move events
    """

    successful_move_events = successful_pass_events + successful_dribble_events

    return df.loc[df[event_column_name].isin(successful_move_events)].copy()



def p_shoot_or_move(df, successful_shot_events, failed_shot_events, successful_pass_events, failed_pass_events, successful_dribble_events, failed_dribble_events, event_column_name, l = 18, w = 12, pitch_length = 105.0, pitch_width = 68.0):
    """
    Takes in the events dataframe, and outputs two MxN matrices:
        1. The first is the choice-to-shoot matrix, where each element is the probability of a player choosing to shoot from that location
        2. The second is the choice-to-move matrix, where each element represents the probability of a player choosing to move the ball (either by passing, crossing, or dribbling)
    """

    all_shot_events = successful_shot_events + failed_shot_events

    df_moves = get_df_all_moves(df, successful_pass_events, failed_pass_events, successful_dribble_events, failed_dribble_events, event_column_name)
    df_shots = df.loc[df[event_column_name].isin(all_shot_events)]

    move_matrix = count(df_moves.x1_m, df_moves.y1_m, l, w, pitch_length, pitch_width)
    shot_matrix = count(df_shots.x1_m, df_shots.y1_m, l, w, pitch_length, pitch_width)
    total_matrix = move_matrix + shot_matrix

    return safe_divide(shot_matrix, total_matrix), safe_divide(move_matrix, total_matrix)



def move_transition_matrix(df, successful_pass_events, failed_pass_events, successful_dribble_events, failed_dribble_events, event_column_name, l = 18, w = 12, pitch_length = 105.0, pitch_width = 68.0):
    """
    Computation of transition matrix
    """

    df_moves = get_df_all_moves(df, successful_pass_events, failed_pass_events, successful_dribble_events, failed_dribble_events, event_column_name)

    # apply flat index to df_moves, to start and end positions
    df_moves['z'] = get_flat_indexes(df_moves.x1_m, df_moves.y1_m, l, w, pitch_length=105, pitch_width=68)
    df_moves['z_prime'] = get_flat_indexes(df_moves.x2_m, df_moves.y2_m, l, w, pitch_length=105, pitch_width=68)

    # getting successful move events (will filter on these when calculating )
    successful_moves = successful_pass_events + successful_dribble_events

    """
    REMEMBER, THE FLAT INDEXES ARE left to right, top to bottom.
    So the top left index is 0, the top right index is 215, and the bottom right index is 46655
    AND IN THE TRANSITION MATRIX, ROWS REPRESENT STARTING LOCATIONS, AND COLUMNS REPRESENT ENDING LOCATIONS SO WE'LL GET SUCCESS COUNTS PER CELL (SUCCESS FROM z TO z')
    AND THEN WE'LL DIVIDE EACH CELL BY THE SUM OF ALL MOVES FROM THE STARTING LOCATION

    WHICH MEANS WE'LL BE DIVIDING EACH CELL (REPRESENTING A SUCCESSFUL MOVE) ---  WHERE EACH ROW REPRESENTS THE STARTING LOCATION,  z ---
    BY THE ROW SUM OF TOTAL MOVES (NOT JUST SUCCESSFUL)
    """

    ## CREATING EMPTY TRANSITION MATRIX + DENOMINATOR VECTOR

    # this is an MxN by MxN matrix
    # so it's an all-zones by all-zones grid
    transition_matrix = np.zeros((w * l, w * l))
    transition_matrix_denom = np.zeros(w*l)

    ## CALCULATING SUCCESS COUNTS
    df_successful_counts = df_moves.loc[df_moves[event_column_name].isin(successful_moves)]\
        .groupby(['z','z_prime'])\
        .agg({'playerId':'count'})\
        .rename(columns={'playerId':'successful_counts'})\
        .reset_index()

    ## CALCULATING DENOMINATOR COUNTS
    df_denom = df_moves.groupby('z')\
        .agg({'playerId':'count'})\
        .rename(columns={'playerId':'total_z_counts'})\
        .reset_index()

    # TRANSFORMING DATAFRAME -> NUMPY MATRIX
    transition_matrix[df_successful_counts.z, df_successful_counts.z_prime] = df_successful_counts.successful_counts
    transition_matrix_denom[df_denom.z] = df_denom.total_z_counts

    ## TWO METHODS TO PERFORM THIS DIVISION
    #1) = xT.safe_divide(transition_matrix.T, transition_matrix_denom).T
    #2) transition_matrix = xT.safe_divide(transition_matrix, transition_matrix_denom.reshape(l*w,1))

    return safe_divide(transition_matrix, transition_matrix_denom.reshape(l*w,1))



## TODO: put the loops in your own code
def xT_surface(df, successful_shot_events, failed_shot_events, successful_pass_events, failed_pass_events, successful_dribble_events, failed_dribble_events, event_column_name, l=18, w=12, pitch_length=105, pitch_width=68):
    """
    Iteratively calculates MxN xT value surface.

    Origin: top left of the pitch
    """

    epsilon = 1e-5

    heatmaps = []

    xT = np.zeros((w, l))

    print ('Calculating xG...')
    xG = p_score_if_shoot(df, successful_shot_events, failed_shot_events, event_column_name, l, w, pitch_length=105, pitch_width=68)

    print ('Calculating pShoot & pMove...')
    pS, pM = p_shoot_or_move(df, successful_shot_events, failed_shot_events, successful_pass_events, failed_pass_events, successful_dribble_events, failed_dribble_events, event_column_name, l, w, pitch_length=105, pitch_width=68)

    print ('Calculating transition matrix...')
    transition_matrix = move_transition_matrix(df, successful_pass_events, failed_pass_events, successful_dribble_events, failed_dribble_events, event_column_name, l, w, pitch_length=105, pitch_width=68)

    delta = 1e6

    # iteration zero: xT is MxN of zeros
    it = 0
    heatmaps.append(xT)

    print ('Calculating xT value surface...')
    # running this until every element of xT has converged
    while np.any(delta > epsilon):

        print (f'Running {it+1} iteration of xT...')

        total_payoff = np.zeros((w, l))

        for y in range(0, w):
            for x in range(0, l):
                for q in range(0, w):
                    for z in range(0, l):
                        total_payoff[y, x] += (transition_matrix[l * y + x, l * q + z] * xT[q, z])

        xT_new = (pS * xG) + (pM * total_payoff)
        delta = xT_new - xT
        xT = xT_new
        heatmaps.append(xT.copy())
        it += 1

    print (f'# iterations: {it}')

    return xT, heatmaps


# 1. go from x, y -> x_zone, y_zone
# 2. do this for x_final and y_final
# 3. get xT values for both and take start from final
# 4. interpolator

def bilinear_interp_xT(xT, l=18, w=12, pitch_length=105, pitch_width=68, interpolation_factor=100):
    """

    Applies bilinear interpolation https://en.wikipedia.org/wiki/Bilinear_interpolation to our MxN xT surface to take advantage of the additional
    location precision provided in the Opta data

    """

    zone_length = 105 / l
    zone_width = 68 / w

    # getting the centres of the MxN zones
    zone_x_centres = np.arange(0.0, pitch_length, zone_length) + 0.5 * zone_length
    zone_y_centres = np.arange(0.0, pitch_width, zone_width) + 0.5 * zone_width

    # linear interpolation of our xT surface
    interp_xT = interp2d(x=zone_x_centres, y=zone_y_centres, z=xT, kind='linear', bounds_error=False)

    interp_x = np.linspace(0, pitch_length, l*interpolation_factor)
    interp_y = np.linspace(0, pitch_width, w*interpolation_factor)

    return interp_xT(interp_x, interp_y)



def apply_xT(df, xT, successful_pass_events, failed_pass_events, successful_dribble_events, failed_dribble_events, l=18, w=12, pitch_length=105, pitch_width=68, interpolation_factor=100, xT_mode = 1):
    """

    Mode 1: Only applies to successful actions

    Mode 2: Applies to both successful and negative actions: negative scoring is just the opposite sign of the action being successful

    Mode 3: Applies to both successful and negative actions: negative scoring is implemented by difference in xT's being between the opposite team having the ball at the end position (in their 100-x, 100-y ref frame) and the starting xT for the attacking team (who loses the ball)

    """

    interp_xT = bilinear_interp_xT(xT, l, w, pitch_length, pitch_width, interpolation_factor)

    l = l*interpolation_factor
    w = w*interpolation_factor

    successful_actions = successful_pass_events + successful_dribble_events
    failed_actions = failed_pass_events + failed_dribble_events
    all_actions = successful_actions + failed_actions

    x1, y1 = df.x1_m, df.y1_m
    x2, y2 = df.x2_m, df.y2_m

    # for events that have nan x2 & y2, setting them to being the same as x1 & y1
    x2[np.isnan(x2)] = x1[np.isnan(x2)]
    y2[np.isnan(y2)] = y1[np.isnan(y2)]

    x_start, y_start = get_cell_indexes(x1, y1, l, w)
    x_end, y_end = get_cell_indexes(x2, y2, l, w)

    actions = df.eventSubType.values

    # might need to double check the orientation here
    # this is vectorised so is efficient to calculate

    if xT_mode == 1:

        # only looking at successful actions
        x_start = pd.Series([i if j in successful_actions else 0 for i, j in zip(x_start, actions)])
        x_end = pd.Series([i if j in successful_actions else 0 for i, j in zip(x_end, actions)])
        y_start = pd.Series([i if j in successful_actions else 0 for i, j in zip(y_start, actions)])
        y_end = pd.Series([i if j in successful_actions else 0 for i, j in zip(y_end, actions)])

        # w -  1  -  y_start is how we convert between y values as a cell index and the index in the interpolated xT matrix (basically y-inverted)
        # y represents rows in interp_xT matrix
        # x represents columns in interp_xT matrix
        xT_start = interp_xT[w - 1 - y_start, x_start]
        xT_end = interp_xT[w - 1 - y_end, x_end]
        xT_delta = xT_end - xT_start

    elif xT_mode == 2:

        # looking at all actions
        x_start = pd.Series([i if j in all_actions else 0 for i, j in zip(x_start, actions)])
        x_end = pd.Series([i if j in all_actions else 0 for i, j in zip(x_end, actions)])
        y_start = pd.Series([i if j in all_actions else 0 for i, j in zip(y_start, actions)])
        y_end = pd.Series([i if j in all_actions else 0 for i, j in zip(y_end, actions)])

        xT_start = interp_xT[w - 1 - y_start, x_start]
        xT_end = interp_xT[w - 1 - y_end, x_end]
        xT_delta = xT_end - xT_start

        # if the action is failed, either give the player the negative score of the intended action, or 0 (as there shouldn't be a reward for an unsuccessful attempt to move the ball into a less threatening area)
        xT_delta = [min([-i, 0]) if j in failed_actions else i for i, j in zip(xT_delta, actions)]


    elif xT_mode == 3:

        ## calculating the difference in xT between attacking team position and unsuccessful action that provides the ball to the defensive team (altering x_end)

        # looking at all actions
        x_start = pd.Series([i if j in all_actions else 0 for i, j in zip(x_start, actions)])
        x_end = pd.Series([i if j in all_actions else 0 for i, j in zip(x_end, actions)])
        y_start = pd.Series([i if j in all_actions else 0 for i, j in zip(y_start, actions)])
        y_end = pd.Series([i if j in all_actions else 0 for i, j in zip(y_end, actions)])

        # x_end modification
        x_end = pd.Series([(l-1)-i if j in failed_actions else i for i, j in zip(x_end, actions)])

        xT_start = interp_xT[w - 1 - y_start, x_start]
        xT_end = interp_xT[w - 1 - y_end, x_end]

        # if successful, it's the delta between finish and starting attacking position
        # if unsuccessful, it's the delta between the oppositions position (now in attack) and the starting attack position
        # xT_delta = xT_end - xT_start
        xT_delta = [i-j if k not in failed_actions else min([-(j+i),0])/10 for i, j, k in zip(xT_end, xT_start, actions)]

    return xT_delta


import pandas as pd
import numpy as np
from scipy.interpolate import interp2d
