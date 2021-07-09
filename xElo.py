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
    d = 1 = WIN
    d = 0 = LOSS
    """
    return R + k*(d-P)


def elo(Ra, Rb, k, d):
    """
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

    dic_scores = {i:initial_rating for i in things}

    for result in results:

        winner, loser = result
        Ra, Rb = dic_scores[winner], dic_scores[loser]
        _, _, newRa, newRb = elo(Ra, Rb, k, 1)
        dic_scores[winner], dic_scores[loser] = newRa, newRb

    return dic_scores


def mElo(things, initial_rating, k, results, numEpochs):
    """
    Randomises the sequence of the pairwise comparisons, running the Elo sequence in a random
    sequence for a number of epochs

    Returns the mean Elo ratings over the randomised epoch sequences
    """

    lst_outcomes = []

    for i in np.arange(numEpochs):
        np.random.shuffle(results)
        lst_outcomes.append(elo_sequence(things, initial_rating, k, results))

    return pd.DataFrame(lst_outcomes).mean().sort_values(ascending=False)


"""

EXAMPLE

initial_rating = 400
k = 100

things = ['Malted Milk','Rich Tea','Hobnob','Digestive']

results = np.array([('Malted Milk','Rich Tea'),('Malted Milk','Digestive'),('Malted Milk','Hobnob')\
            ,('Hobnob','Rich Tea'),('Hobnob','Digestive'),('Digestive','Rich Tea')])

mElo(things, initial_rating, k, results, 1000)

"""
