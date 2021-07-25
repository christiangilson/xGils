# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:38:46 2020

Christian's dual duel mean Elo implementation
"""

import math
import numpy as np
import pandas as pd

def expected_win(r1, r2):
    """
    Expected probability of player 1 beating player 2
    if player 1 has rating 1 (r1) and player 2 has rating 2 (r2)
    """
    return 1.0 / (1 + math.pow(10, (r2-r1)/400))


def update_rating(R, k, P, d):
    """
    * k represents k factor
    * R represents the players rating (before the match)
    * d represents the ACTUAL outcome of a match
    * P represents the expected probability that the player would win the match

    -> Therefore d-P is the difference between actual and expected
    -> k modifies this
    -> And you add this to the original pre-match rating as an update

    d = 1 = WIN
    d = 0 = LOSS
    """
    return R + k*(d-P)


def elo(Ra, Rb, k, d):
    """
    A single set of Elo updates, given the outcome of a single match between two players
    ,with the ratings for each player provided

    Inputs: takes the pre-match ratings of player A and B, the k-factor, and whether A won/lost (via d=1/0)

    Outputs: updated ratings for players A and B following the result


    d = 1 when player A wins
    d = 0 when player B wins
    """

    Pa = expected_win(Ra, Rb)
    Pb = expected_win(Rb, Ra)

    # update if A wins
    if d == 1:
        Ra = update_rating(Ra, k, Pa, d)
        Rb = update_rating(Rb, k, Pb, d-1)

    # update if B wins
    elif d == 0:
        Ra = update_rating(Ra, k, Pa, d)
        Rb = update_rating(Rb, k, Pb, d+1)

    return Pa, Pb, Ra, Rb


def elo_sequence(things, initial_rating, k, results):
    """
    Initialises score dictionary, and runs through sequence of pairwise results, returning final dictionary of Elo rankings
    """

    # initialise the ratings of all players in the universe of players
    # and give each of those players an initial rating (usually 100)
    dic_scores = {i:initial_rating for i in things}

    # then looping through each pairwise  comparison
    for result in results:

        # to make life easier, the results should be fed in as winner / loser
        winner, loser = result
        # getting the pre-match scores per player from the dictionary
        Ra, Rb = dic_scores[winner], dic_scores[loser]
        # getting the updated scores via the elo function
        _, _, newRa, newRb = elo(Ra, Rb, k, 1)
        # storing those updated scores
        dic_scores[winner], dic_scores[loser] = newRa, newRb

    # outputting the dictionary of scores after a full sequence of results
    return dic_scores


def elo_attack_defence_sequence(things, initial_rating, k, results):
    """
    Initialises score dictionaries for attack and defence, and runs through sequence of pairwise results, returning final dictionaries with Elo rankings for both attack (dribblers) and defence (of dribblers)
    """

    dic_scores_attack = {i:initial_rating for i in things}
    dic_scores_defence = {i:initial_rating for i in things}

    for result in results:

        winner, loser, dribble_outcome = result

        # winner = attacker, loser = defender
        if dribble_outcome == 1:
            Ra, Rb = dic_scores_attack[winner], dic_scores_defence[loser]
            _, _, newRa, newRb = elo(Ra, Rb, k, 1)
            dic_scores_attack[winner], dic_scores_defence[loser] = newRa, newRb

        # winner = defender, loser = attacker
        elif dribble_outcome == 0:
            Ra, Rb = dic_scores_defence[winner], dic_scores_attack[loser]
            _, _, newRa, newRb = elo(Ra, Rb, k, 1)
            dic_scores_defence[winner], dic_scores_attack[loser] = newRa, newRb

    return dic_scores_attack, dic_scores_defence



def mElo(things, initial_rating, k, results, numEpochs):
    """
    Randomises the order of the pairwise comparisons, running the Elo sequence in a random
    sequence for a number of epochs

    Returns the mean Elo ratings over the randomised epoch sequences
    """

    lst_outcomes = []

    for i in np.arange(numEpochs):
        np.random.shuffle(results)
        lst_outcomes.append(elo_sequence(things, initial_rating, k, results))

    return pd.DataFrame(lst_outcomes).mean().sort_values(ascending=False)


def mElo_attack_defence(things, initial_rating, k, results, numEpochs):
    """
    Randomises the sequence of the pairwise comparisons, running the Elo sequence in a random
    sequence for a number of epochs

    Returns the mean Elo ratings over the randomised epoch sequences
    """

    lst_outcomes_attack = []
    lst_outcomes_defence = []

    for i in np.arange(numEpochs):
        np.random.shuffle(results)
        dic_scores_attack, dic_scores_defence = elo_attack_defence_sequence(things, initial_rating, k, results)
        lst_outcomes_attack.append(dic_scores_attack)
        lst_outcomes_defence.append(dic_scores_defence)

    df_attack = pd.DataFrame(lst_outcomes_attack).mean().sort_values(ascending=False).to_frame(name='eloAttack')
    df_attack['player'] = df_attack.index
    df_attack = df_attack.reset_index(drop=True)[['player','eloAttack']]

    df_defence = pd.DataFrame(lst_outcomes_defence).mean().sort_values(ascending=False).to_frame(name='eloDefence')
    df_defence['player'] = df_defence.index
    df_defence = df_defence.reset_index(drop=True)[['player','eloDefence']]

    df_elo = df_attack.merge(df_defence).sort_values('eloAttack', ascending=False)
    df_elo['eloDribbleRank'] = df_elo.index+1

    df_elo = df_elo.sort_values('eloDefence', ascending=False).reset_index(drop=True)
    df_elo['eloDribbleDefenceRank'] = df_elo.index+1

    return df_elo.sort_values('eloDribbleRank').reset_index(drop=True)

"""

EXAMPLE

initial_rating = 400
k = 100

things = ['Malted Milk','Rich Tea','Hobnob','Digestive']

results = np.array([('Malted Milk','Rich Tea'),('Malted Milk','Digestive'),('Malted Milk','Hobnob')\
            ,('Hobnob','Rich Tea'),('Hobnob','Digestive'),('Digestive','Rich Tea')])

mElo(things, initial_rating, k, results, 1000)

"""
