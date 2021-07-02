# -*- coding: utf-8 -*-

"""
## **xT Implementation**
#1. Indexing Functions;
#2. Counting Functions;
#3. Dataframe Filtering Functions;
#4. Probability Matrix Functions;
#5. Solving xT Function;
#6. Applying xT Functions.
"""


"""
## Bayesian Probability Matrix Functions

# 1. Bayesian xG (bayes_p_score_if_shoot)
# 2. Bayesian Shoot or Move (bayes_p_shoot_or_move)
# 3. Bayesian Transition Matrix (bayes_move_transition_matrix)
"""


def successful_move_count_matrix(df, successful_pass_events, successful_dribble_events, event_column_name, l=18, w = 12, pitch_length = 105.0, pitch_width = 68.0):
    """
    Getting count matrix for SUCCESSFUL MOVES for given events dataframe
    """

    all_successful_moves = successful_pass_events + successful_dribble_events

    df_successful_moves = df.loc[df[event_column_name].isin(all_successful_moves)]

    successful_move_matrix = count(df_successful_moves.x1_m, df_successful_moves.y1_m, l, w, pitch_length, pitch_width)

    return successful_move_matrix


def failed_move_count_matrix(df, failed_pass_events, failed_dribble_events, event_column_name, l=18, w = 12, pitch_length = 105.0, pitch_width = 68.0):
    """
    Getting count matrix for FAILED MOVES for given events dataframe
    """

    all_failed_moves = failed_pass_events + failed_dribble_events

    df_failed_moves = df.loc[df[event_column_name].isin(all_failed_moves)]

    failed_move_matrix = count(df_failed_moves.x1_m, df_failed_moves.y1_m, l, w, pitch_length, pitch_width)

    return failed_move_matrix


def successful_shot_count_matrix(df, successful_shot_events, event_column_name, l=18, w = 12, pitch_length = 105.0, pitch_width = 68.0):
    """
    Getting count matrix for SUCCESSFUL SHOTS for given events dataframe
    """

    df_goals = df.loc[df[event_column_name].isin(successful_shot_events)]

    goal_matrix = count(df_goals.x1_m, df_goals.y1_m, l, w, pitch_length, pitch_width)

    return goal_matrix


def failed_shot_count_matrix(df, failed_shot_events, event_column_name, l=18, w = 12, pitch_length = 105.0, pitch_width = 68.0):
    """
    Getting count matrix for FAILED SHOTS for given events dataframe
    """

    df_misses = df.loc[df[event_column_name].isin(failed_shot_events)]

    miss_matrix = count(df_misses.x1_m, df_misses.y1_m, l, w, pitch_length, pitch_width)

    return miss_matrix


def bayes_p_score_if_shoot(prior_successful_shot_matrix, data_successful_shot_matrix, prior_failed_shot_matrix, data_failed_shot_matrix):
    """
    Takes in the events dataframe and extracts counts of shots and goals

    Those counts are then used to calculate the expected goals (xG) per zone

    This is a highly simplistic approach to calculating xG - sophistication can surely be added here?

    Outputs an M x N matrix of xG
    """

    posterior_successful_shots = prior_successful_shot_matrix + data_successful_shot_matrix
    posterior_failed_shots = prior_failed_shot_matrix + data_failed_shot_matrix
    posterior_total_shots = posterior_successful_shots + posterior_failed_shots

    return safe_divide(posterior_successful_shots, posterior_total_shots)




def bayes_p_shoot_or_move(prior_successful_shot_matrix, data_successful_shot_matrix, prior_failed_shot_matrix, data_failed_shot_matrix, prior_successful_move_matrix, data_successful_move_matrix, prior_failed_move_matrix, data_failed_move_matrix):
    """
    Takes in the events dataframe, and outputs two MxN matrices:
        1. The first is the choice-to-shoot matrix, where each element is the probability of a player choosing to shoot from that location
        2. The second is the choice-to-move matrix, where each element represents the probability of a player choosing to move the ball (either by passing, crossing, or dribbling)
    """

    posterior_successful_moves = prior_successful_move_matrix + data_successful_move_matrix
    posterior_failed_moves = prior_failed_move_matrix + data_failed_move_matrix
    posterior_successful_shots = prior_successful_shot_matrix + data_successful_shot_matrix
    posterior_failed_shots = prior_failed_shot_matrix + data_failed_shot_matrix

    posterior_total_moves = posterior_successful_moves + posterior_failed_moves
    posterior_total_shots = posterior_successful_shots + posterior_failed_shots

    posterior_total = posterior_total_moves + posterior_total_shots

    return safe_divide(posterior_total_shots, posterior_total), safe_divide(posterior_total_moves, posterior_total)


"""
THIS IS THE FUNCTION WITH A BUG IN IT!!!
"""
def bayes_move_transition_matrices(df, successful_pass_events, failed_pass_events, successful_dribble_events, failed_dribble_events, event_column_name, l=18, w = 12, pitch_length = 105.0, pitch_width = 68.0):
    """
    Computation of transition matrices outputting two matrics: counts of successes and counts of failures

    This is the same code as the all-in-one transition matrix function except that it produces two MxN x MxN matrices of counts (rather than probabilities)

    """
    df_moves = get_df_all_moves(df, successful_pass_events, failed_pass_events, successful_dribble_events, failed_dribble_events, event_column_name)

    # apply flat index to df_moves, to start and end positions
    df_moves['z'] = get_flat_indexes(df_moves.x1_m, df_moves.y1_m, N, M, pitch_length=105, pitch_width=68)
    df_moves['z_prime'] = get_flat_indexes(df_moves.x2_m, df_moves.y2_m, N, M, pitch_length=105, pitch_width=68)

    # getting successful and failed move events (will filter on these when calculating )
    successful_moves = opta_successful_pass_events + opta_successful_dribble_events
    failed_moves = opta_failed_pass_events + opta_failed_dribble_events

    # there's a chance you may not have counts in all zones, so we need to start all zones, and then left join the counts onto df_z
    df_z = pd.DataFrame(np.arange(0, l*w), columns=['z'])
    df_z_prime = pd.DataFrame(np.arange(0, l*w), columns=['z_prime'])

    # getting the starting counts per zone, z
    df_z_counts = df_moves.sort_values('z', ascending=True).groupby('z').agg({'playerId':'count'}).reset_index().rename(columns={'playerId':'count'})

    # applying counts to empty grid
    df_z_counts = df_z.merge(df_z_counts, how='left', on='z').fillna(0).astype(int)

    # this is an MxN by MxN matrix
    # so it's an all-zones by all-zones grid
    transition_matrix_success = np.zeros((w * l, w * l))
    transition_matrix_failure = np.zeros((w * l, w * l))

    # iterating through starting zones
    for i in np.arange(0, w * l):

        # dataframe of z_prime counts, for successful moves (so it's the probability of successfully moving from z to z')
        df_z_prime_success_counts = df_moves.loc[(df_moves['z'] == i) & (df_moves['eventSubType'].isin(successful_moves))].groupby('z_prime').agg({'playerId':'count'}).reset_index().rename(columns={'playerId':'count'})
        df_z_prime_success_counts = df_z_prime.merge(df_z_prime_success_counts, how='left', on='z_prime').fillna(0).astype(int)

        # updating the transition matrix
        # rows indexed on initial zone
        # columns indexed on final zone
        transition_matrix_success[i, df_z_prime_success_counts.z_prime.values] = df_z_prime_success_counts['count'].values

        # replicating the above but for failed counts
        df_z_prime_failed_counts = df_moves.loc[(df_moves['z'] == i) & (df_moves['eventSubType'].isin(failed_moves))].groupby('z_prime').agg({'playerId':'count'}).reset_index().rename(columns={'playerId':'count'})
        df_z_prime_failed_counts = df_z_prime.merge(df_z_prime_failed_counts, how='left', on='z_prime').fillna(0).astype(int)

        # failed transition count matrix
        transition_matrix_failure[i, df_z_prime_failed_counts.z_prime.values] = df_z_prime_failed_counts['count'].values

    return transition_matrix_success, transition_matrix_failure


"""
## Bayesian Conjugate Updating: Beta-Binomial Analysis
"""

def posterior_bayes_mean(prior_success_counts, data_success_counts, prior_fail_counts, data_fail_counts):

    posterior_alpha = prior_success_counts + data_success_counts
    posterior_beta = prior_fail_counts + data_fail_counts

    posterior_mean = safe_divide(posterior_alpha, (posterior_alpha + posterior_beta))

    return posterior_mean


"""
Solving xT
"""

## TODO: put the loops in your own code
def xT_surface(df, successful_shot_events, failed_shot_events, successful_pass_events, failed_pass_events, successful_dribble_events, failed_dribble_events, event_column_name, l=18, w = 12, pitch_length=105, pitch_width=68):
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


def bayes_xT_surface(xG, pS, pM, T, l=18, w = 12):
    """
    For the Bayesian application of the xT surface we're going to be solving the xT equation once per month, and we'll be inputting the posterior xG, pS, pM, T matrices into this function
    """

    # Initial set up
    epsilon = 1e-5

    heatmaps = []

    xT = np.zeros((w, l))

    delta = 1e6

    # iteration zero: xT is MxN of zeros
    it = 0
    heatmaps.append(xT)

    #print ('Calculating xT value surface...')
    # running this until every element of xT has converged
    while np.any(delta > epsilon):

        #print (f'Running {it+1} iteration of xT...')

        total_payoff = np.zeros((w, l))

        for y in range(0, w):
            for x in range(0, l):
                for q in range(0, w):
                    for z in range(0, l):
                        total_payoff[y, x] += (T[l * y + x, l * q + z] * xT[q, z])

        xT_new = (pS * xG) + (pM * total_payoff)
        delta = xT_new - xT
        xT = xT_new
        heatmaps.append(xT.copy())
        it += 1

    #print (f'# iterations: {it}')

    return xT, heatmaps


"""
Bilinear Interpolation
"""

def bilinear_interp_xT(xT, l=18, w = 12, pitch_length=105, pitch_width=68, interpolation_factor=100):
    """

    Applies bilinear interpolation https://en.wikipedia.org/wiki/Bilinear_interpolation to our MxN xT surface to take advantage of the additional
    location precision provided in the Opta data

    """

    zone_length = pitch_length / l
    zone_width = pitch_width / w

    # getting the centres of the MxN zones
    zone_x_centres = np.arange(0.0, pitch_length, zone_length) + 0.5 * zone_length
    zone_y_centres = np.arange(0.0, pitch_width, zone_width) + 0.5 * zone_width

    # linear interpolation of our xT surface
    interp_xT = interp2d(x=zone_x_centres, y=zone_y_centres, z=xT, kind='linear', bounds_error=False)

    interp_x = np.linspace(0, pitch_length, l*interpolation_factor)
    interp_y = np.linspace(0, pitch_width, w*interpolation_factor)

    return interp_xT(interp_x, interp_y)


"""
# Applying xT To Events Data
"""


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
