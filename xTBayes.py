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

    successful_move_matrix = xT.count(df_successful_moves.x1_m, df_successful_moves.y1_m, l, w, pitch_length, pitch_width)

    return successful_move_matrix


def failed_move_count_matrix(df, failed_pass_events, failed_dribble_events, event_column_name, l=18, w = 12, pitch_length = 105.0, pitch_width = 68.0):
    """
    Getting count matrix for FAILED MOVES for given events dataframe
    """

    all_failed_moves = failed_pass_events + failed_dribble_events

    df_failed_moves = df.loc[df[event_column_name].isin(all_failed_moves)]

    failed_move_matrix = xT.count(df_failed_moves.x1_m, df_failed_moves.y1_m, l, w, pitch_length, pitch_width)

    return failed_move_matrix


def successful_shot_count_matrix(df, successful_shot_events, event_column_name, l=18, w = 12, pitch_length = 105.0, pitch_width = 68.0):
    """
    Getting count matrix for SUCCESSFUL SHOTS for given events dataframe
    """

    df_goals = df.loc[df[event_column_name].isin(successful_shot_events)]

    goal_matrix = xT.count(df_goals.x1_m, df_goals.y1_m, l, w, pitch_length, pitch_width)

    return goal_matrix


def failed_shot_count_matrix(df, failed_shot_events, event_column_name, l=18, w = 12, pitch_length = 105.0, pitch_width = 68.0):
    """
    Getting count matrix for FAILED SHOTS for given events dataframe
    """

    df_misses = df.loc[df[event_column_name].isin(failed_shot_events)]

    miss_matrix = xT.count(df_misses.x1_m, df_misses.y1_m, l, w, pitch_length, pitch_width)

    return miss_matrix


def bayes_p_score_if_shoot(prior_successful_shot_matrix, data_successful_shot_matrix, prior_failed_shot_matrix, data_failed_shot_matrix, l=18, w=12, pitch_length = 105.0, pitch_width = 68.0, use_synthetic = 0, df_synthetic = None):
    """
    If synthetic NOT used:
    Takes in the events dataframe and extracts counts of shots and goals
    Those counts are then used to calculate the expected goals (xG) per zone
    Outputs an M x N matrix of xG

    If synthetic shots used as ANOTHER prior:
    We combine synthetic and real shots to produce an xG grid where the posterior contains a prior from the previous data, as well as a prior from the synthetic shots, and the new data.
    """

    # if we're not using synthetic shot counts to deal with outliers
    if use_synthetic == 0:
        posterior_successful_shots = prior_successful_shot_matrix + data_successful_shot_matrix
        posterior_failed_shots = prior_failed_shot_matrix + data_failed_shot_matrix
        posterior_total_shots = posterior_successful_shots + posterior_failed_shots

    # elif you're using synthetic counts
    else:
        # querying synthetic shot dataframe to produce synthetic goal dataframe, and calculating synthetic success and failures matrix
        df_synthetic_successful_shots = df_synthetic.loc[df_synthetic['goal'] == 1].copy()
        df_synthetic_failed_shots = df_synthetic.loc[df_synthetic['goal'] == 0].copy()
        synthetic_successful_shot_matrix = xT.count(df_synthetic_successful_shots.x1_m, df_synthetic_successful_shots.y1_m, l, w, pitch_length, pitch_width)
        synthetic_failed_shot_matrix = xT.count(df_synthetic_failed_shots.x1_m, df_synthetic_failed_shots.y1_m, l, w, pitch_length, pitch_width)

        # now combining the real (prior and data) and synthetic matrices
        posterior_successful_shots = prior_successful_shot_matrix + data_successful_shot_matrix + synthetic_successful_shot_matrix
        posterior_failed_shots = prior_failed_shot_matrix + data_failed_shot_matrix + synthetic_failed_shot_matrix
        posterior_total_shots = posterior_successful_shots + posterior_failed_shots

    return xT.safe_divide(posterior_successful_shots, posterior_total_shots)




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

    return xT.safe_divide(posterior_total_shots, posterior_total), xT.safe_divide(posterior_total_moves, posterior_total)



def bayes_move_transition_matrices(df, successful_pass_events, failed_pass_events, successful_dribble_events, failed_dribble_events, event_column_name, l=18, w = 12, pitch_length = 105.0, pitch_width = 68.0):
    """
    Computation of transition matrices outputting two matrics: counts of successes and counts of failures

    This is the same code as the all-in-one transition matrix function except that it produces two MxN x MxN matrices of counts (rather than probabilities)

    """
    df_moves = xT.get_df_all_moves(df, successful_pass_events, failed_pass_events, successful_dribble_events, failed_dribble_events, event_column_name)

    # apply flat index to df_moves, to start and end positions
    df_moves['z'] = xT.get_flat_indexes(df_moves.x1_m, df_moves.y1_m, l, w, pitch_length=105, pitch_width=68)
    df_moves['z_prime'] = xT.get_flat_indexes(df_moves.x2_m, df_moves.y2_m, l, w, pitch_length=105, pitch_width=68)

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
    transition_matrix_counts = np.zeros((w * l, w * l))
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

    # TRANSFORMING DATAFRAMES -> NUMPY MATRICES
    transition_matrix_counts[df_successful_counts.z, df_successful_counts.z_prime] = df_successful_counts.successful_counts
    transition_matrix_denom[df_denom.z] = df_denom.total_z_counts

    return transition_matrix_counts, transition_matrix_denom


"""
Updating xT surface via Beta-Binomial conjugate analysis
"""

def bayes_xT_surface(xG, pS, pM, T, l=18, w =12):
    """
    For the Bayesian application of the xT surface we're going to be solving the xT equation once per month, and we'll be inputting the posterior xG, pS, pM, T matrices into this function
    """

    # Initial set up
    epsilon = 1e-5

    heatmaps = []

    xT_surface = np.zeros((w, l))

    delta = 1e6

    # iteration zero: xT is MxN of zeros
    it = 0
    heatmaps.append(xT_surface)

    #print ('Calculating xT value surface...')
    # running this until every element of xT has converged
    while np.any(delta > epsilon):

        #print (f'Running {it+1} iteration of xT...')

        total_payoff = np.zeros((w, l))

        for y in range(0, w):
            for x in range(0, l):
                for q in range(0, w):
                    for z in range(0, l):
                        total_payoff[y, x] += (T[l * y + x, l * q + z] * xT_surface[q, z])

        xT_new = (pS * xG) + (pM * total_payoff)
        delta = xT_new - xT_surface
        xT_surface = xT_new
        heatmaps.append(xT_surface.copy())
        it += 1

    #print (f'# iterations: {it}')

    return xT_surface


import pandas as pd
import numpy as np
from scipy.interpolate import interp2d
import xGils.xT as xT
