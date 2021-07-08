# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.cm as cm
from mplsoccer.pitch import Pitch


def produce_df_teams_ref(df):
    # producing a dataframe for team ref
    df_teams = df[['homeTeamName','homeTeamId']]\
                .drop_duplicates()\
                .sort_values('homeTeamId')\
                .reset_index(drop=True)\
                .rename(columns={'homeTeamName':'teamName','homeTeamId':'teamId'})

    return df_teams


def produce_df_matches_ref(df):
    """
    Produces data frame of match reference data from events dataframe for the season of interest.
    """

    return df[['matchId','homeTeamId','awayTeamId']].drop_duplicates().reset_index(drop=True).copy()


def xT_grid(df, events, eventCol='eventSubType', bins=(18,12), exclCorners = 1):
    """
    Input should be a dataframe containing x1_m, y1_m, xT columns for a given team/season (whatever combination you want to aggregate xT over in to bins).

    To make life easier when plotting / aggregating statistics between Opta and Wyscout, we'll use standardised Uefa coordinate system throughout.

    Key: the only columns you need in the event dataframe are [x1_m, y1_m, xT]
    """

    # can be useful to exclude corners from this analysis
    ## 1) statistically: corners don't really characterise the way a team threatens the opponent
    ## 2) visually: as corners are such focussed events w.r.t. location, so they dominate and detract from the team's grid if they're included
    if exclCorners == 1:
        df = df.loc[~((df['x1_m'] >= 104.4) & (df['y1_m'] >= 67.5))].reset_index(drop=True)
        df = df.loc[~((df['x1_m'] >= 104.4) & (df['y1_m'] <= 0.5))].reset_index(drop=True)

    # instantiate a Uefa pitch object (pitch dimensions: [[0,105],[0,68]]): we'll subsequently use the binned_statistic method to calculate the summed counts, and also to help plot the pitches later
    pitch = Pitch(pitch_type='uefa', figsize=(16,9), pitch_color='white', line_zorder=2, line_color='gray')

    # filtering by actions
    df_events = df.loc[df[eventCol].isin(events)].reset_index(drop=True).copy()

    # binned statistics
    ## standardised=True means we're using Uefa pitch range for the binning, i.e. in the underlying scipy code, it'll define a range=[[0,105],[0,68]]
    bs = pitch.bin_statistic(df_events.x1_m, df_events.y1_m, values=df_events.xT, statistic='sum', bins=bins, standardized=True)

    # returning an array, where the first element is the transposed matrix of counts, and the second object is a chunkier object with all of the metadata required for the pitch plots
    ## the second element will be what we use as an input to the plotting function
    ## and the third will be the filtered dataframe on the specific events of interest
    return (bs['statistic'], bs,  df_events)


def plot_xT_pitch(df, events, filepath, transparent=True, teamOrPlayer='team', eventCol='eventSubType', bins=(18,12), figsize=(16,9), fontsize=14, vmax_override=None, cmap=cm.coolwarm, exclCorners=1, scatter=1):
    """
    Plotting an xT pitch using the aggregated statistics from the team_xT_grid function.

    Optional argument to plot a scatter of the underlying events.
    """

    # getting bin statistic bundle from team_xT_grid function
    bs, bs_obj, df_events = xT_grid(df, events, eventCol, bins, exclCorners)

    # instantiate a Uefa pitch object (pitch dimensions: [[0,105],[0,68]]): we'll subsequently use the binned_statistic method to calculate the summed counts, and also to help plot the pitches later
    pitch = Pitch(pitch_type='uefa', pitch_color='white', line_zorder=2, line_color='gray')

    # setting up the figure
    fig, ax = pitch.draw(figsize=figsize)
    fig.patch.set_facecolor('white')

    # setting min/max of the values
    vmax = bs.max()
    vmin = 0

    # using a value override if specified in the function
    if vmax_override != None:
        vmax = vmax_override

    # producing heatmap
    hm = pitch.heatmap(bs_obj, ax=ax, cmap=cmap, edgecolors='white', vmin=vmin, vmax=vmax)

    # optional extra to plot a scatter of the individual events
    if scatter == 1:
        sc = pitch.scatter(df_events.x1_m, df_events.y1_m, c='white', s=2, ax=ax, alpha=0.3)

    # producing the colourbar to the right of the plot
    team_cbar = fig.colorbar(hm, ax=ax)
    team_cbar.set_label('xT', rotation=270, fontsize=fontsize-2)

    # generating plot title
    ## getting team name and season name from df_events dataframe
    df_teams = produce_df_teams_ref(df_events)

    teamId = df_events.playerTeamId.values[0]
    teamName = df_teams.loc[df_teams['teamId'] == teamId].teamName.values[0]
    season = df_events.season.values[0]
    playerName = df_events.playerName[0]

    if teamOrPlayer == 'team':
        ax.set_title(f'{teamName}: {season}', x=0.5, y=0.98, fontsize=fontsize, color='black')
    elif teamOrPlayer == 'player':
        ax.set_title(f'{playerName} ({teamName}): {season}', x=0.5, y=0.98, fontsize=fontsize, color='black')

    return fig.savefig(filepath, transparent=transparent, dpi=300)



def plot_xT_multi_pitch(df, events, lst_teams, lst_seasons, lst_cmaps, filepath, transparent=True, teamOrPlayer='team', eventCol='eventSubType', bins=(18,12), figsize=(18,18), fontsize=14, vmax_override=None, exclCorners=1, scatter=1):
    """
    Plotting an array of xT pitch's using the aggregated statistics from the team_xT_grid function.

    Optional argument to plot a scatter of the underlying events.
    """

    # creating an array of plotting metadata
    numTeams = len(lst_teams)
    numSeasons = len(lst_seasons)

    ## plotting array
    lst_plots = []
    for teamId, cmap in zip(lst_teams, lst_cmaps):
        for season in lst_seasons:
            lst_plots.append((teamId,season,cmap))

    pitch = Pitch(pitch_type='uefa', pitch_color='white', line_zorder=2, line_color='gray')
    # creating numTeams x numSeasons set of pitches
    fig, axs = pitch.draw(figsize=figsize, nrows=numTeams, ncols=numSeasons)

    for ax, p in zip(axs.flat, lst_plots):

        # unpacking plotting metadata
        teamId, season, cmap = p

        # getting team / season dataframe
        df_plot = df.loc[((df['playerTeamId'] == teamId) & (df['season'] == season))]

        # getting bin statistic bundle from team_xT_grid function
        bs, bs_obj, df_events = xT_grid(df_plot, events, eventCol, bins, exclCorners)

        #fig.patch.set_facecolor('white')

        # setting min/max of the values
        vmax = bs.max()
        vmin = 0

        # using a value override if specified in the function
        if vmax_override != None:
            vmax = vmax_override

        # producing heatmap
        hm = pitch.heatmap(bs_obj, ax=ax, cmap=cmap, edgecolors='white', vmin=vmin, vmax=vmax)

        # optional extra to plot a scatter of the individual events
        if scatter == 1:
            sc = pitch.scatter(df_plot.x1_m, df_plot.y1_m, c='white', s=2, ax=ax, alpha=0.3)

        # producing the colourbar to the right of the plot
        team_cbar = fig.colorbar(hm, ax=ax)
        team_cbar.set_label('xT', rotation=270, fontsize=fontsize-2)

        # generating plot title
        ## getting team name and season name from df_events dataframe
        df_teams = produce_df_teams_ref(df)
        teamName = df_teams.loc[df_teams['teamId'] == teamId].teamName.values[0]

        if teamOrPlayer == 'team':
            ax.set_title(f'{teamName}: {season}', x=0.5, y=0.98, fontsize=fontsize, color='black')
        elif teamOrPlayer == 'player':
            ax.set_title(f'{playerName} ({teamName}): {season}', x=0.5, y=0.98, fontsize=fontsize, color='black')

    return fig.savefig(filepath, transparent=transparent, dpi=300)



def delta_xT_grid(df, teamId, events, eventCol='eventSubType', bins=(18,12), exclCorners = 1):

    """
    Input should entire events dataframe for the competition / season of interest.

    Must specify team of interest to calculate delta xT statistics for.

    To make life easier when plotting / aggregating statistics between Opta and Wyscout, we'll use standardised Uefa coordinate system throughout.

    Key: the required columns in the event dataframe are [...]
    """

    # instantiate a Uefa pitch object (pitch dimensions: [[0,105],[0,68]])
    pitch = Pitch(pitch_type='uefa', pitch_color='white', line_zorder=2, line_color='gray')

    # can be useful to exclude corners from this analysis
    ## 1) statistically: corners don't really characterise the way a team threatens the opponent
    ## 2) visually: as corners are such focussed events w.r.t. location, so they dominate and detract from the team's grid if they're included
    if exclCorners == 1:
        df = df.loc[~((df['x1_m'] >= 104.4) & (df['y1_m'] >= 67.5))].reset_index(drop=True)
        df = df.loc[~((df['x1_m'] >= 104.4) & (df['y1_m'] <= 0.5))].reset_index(drop=True)

    # starting by generating dataframe of matches for ALL teams in the competition per season
    df_matches = produce_df_matches_ref(df)

    # getting list of OTHER TEAMS (i.e. the 19 other teamId's that aren't the team of interest)
    # we just ask for the unique list of awayTeamId's when the teamId is the home team.
    lst_otherTeamId = df_matches.loc[df_matches['homeTeamId'] == teamId, 'awayTeamId'].drop_duplicates().values

    # creating a dictionary of matches (want something that's easily queryable to get matches involving
    # the team of interest and "other" matches)
    # dict: {matchId: [home, away]}
    dic_matches = {i:[j,k] for i,j,k in zip(df_matches.matchId,df_matches.homeTeamId,df_matches.awayTeamId)}

    # keeping track of each of the delta matrices
    lst_matrix_Delta = []

    # looping through opponents
    for otherTeamId in lst_otherTeamId:

        # we'll be storing the xT grids for matches VS our team of interest and matches against OTHER teams in lists
        lst_matrix_Vs = []
        lst_matrix_Other = []

        # getting list and number of matchId's VS our team of interest
        lst_matchId_Vs = [i for i in dic_matches.keys() if otherTeamId in dic_matches[i] and teamId in dic_matches[i]]
        numVs = len(lst_matchId_Vs)

        # getting list and number of matchId's against OTHER teams
        lst_matchId_Other = [i for i in dic_matches.keys() if otherTeamId in dic_matches[i] and teamId not in dic_matches[i]]
        numOther = len(lst_matchId_Other)

        # 1) matches VS
        for matchId in lst_matchId_Vs:
            # producing dataframe just for that match, purely containing actions by the other team
            df_xT_match = df.loc[(df['matchId'] == matchId) & (df['playerTeamId'] == otherTeamId) & (df[eventCol].isin(events))].copy()
            # appending binned sums of xT (a matrix) to a list of matrices
            lst_matrix_Vs.append(pitch.bin_statistic(df_xT_match.x1_m, df_xT_match.y1_m, df_xT_match.xT, statistic='sum', bins=bins)['statistic'])

        # 2) matches against OTHER
        for matchId in lst_matchId_Other:
            # producing dataframe just for that match, purely containing actions by the other team
            df_xT_match = df.loc[(df['matchId'] == matchId) & (df['playerTeamId'] == otherTeamId) & (df[eventCol].isin(events))].copy()
            lst_matrix_Other.append(pitch.bin_statistic(df_xT_match.x1_m, df_xT_match.y1_m, df_xT_match.xT, statistic='sum', bins=bins)['statistic'])

        # calculating averages: numpy performing element wise averages per bin
        mean_matrix_Vs = np.mean(lst_matrix_Vs, axis=0)
        mean_matrix_Other = np.mean(lst_matrix_Other, axis=0)

        # calculating delta xT
        ## subtracting the xT average against OTHER teams from VS
        ## this means when there's excess xT (a positive value) left, then other teams are specifically targeting that area
        ## and negative xT is when other teams specifically don't target those regions against the team of interest, but they normally would against other opponents
        matrix_Delta = mean_matrix_Vs - mean_matrix_Other

        lst_matrix_Delta.append(matrix_Delta)

    # and now calculating the average delta
    mean_matrix_Delta = np.mean(lst_matrix_Delta, axis=0)
    var_matrix_Delta = np.var(lst_matrix_Delta, axis=0)

    # and calculating the numDeltas as this is our sample size, if we want to calculate the standard error in the mean
    numDeltas = len(lst_matrix_Delta)

    # hacky bit that'll save a lot of headaches when it comes to plotting this
    ## 1) calculate arbitrary bin statistic with the bin definition that you're using
    bs = pitch.bin_statistic(df_xT_match.x1_m, df_xT_match.y1_m, df_xT_match.xT, statistic='sum', bins=bins)

    ## 2) override the statistic part of the bin statistic data object with the mean_matrix_Delta
    bs['statistic'] = mean_matrix_Delta

    return (bs, mean_matrix_Delta, var_matrix_Delta, numDeltas)



def plot_delta_xT_pitch(df, teamId, events, filepath, transparent=True, eventCol='eventSubType', bins=(18,12), exclCorners = 1, figsize=(16,9), fontsize=14, symmetricDeltaVizOverride=1):
    """
    Plotting the delta xT pitch, given the delta xT payload from delta_xT_grid
    """

    delta_xT_payload = delta_xT_grid(df, teamId, events, eventCol, bins, exclCorners)

    bin_statistic_object, mean_matrix_Delta, var_matrix_Delta, numDeltas = delta_xT_payload

    pitch = Pitch(pitch_type='uefa', pitch_color='white', line_zorder=2, line_color='gray')
    fig, ax = pitch.draw(figsize=figsize)

    # Choice of whether to force symmetry between positive and negative delta xT, or to use the natural min and max
    if symmetricDeltaVizOverride == 1:
        vmin, vmax = -0.01, 0.01
    else:
        vmin, vmax = (mean_matrix_Delta).min(), (mean_matrix_Delta).max()

    # Delta xT heatmap
    hm = pitch.heatmap(bin_statistic_object, ax=ax, cmap=cm.coolwarm, edgecolors='white', vmin=vmin, vmax=vmax)
    # inverting x-  and y- axes so it appears like it's the target team's defence
    hm.axes.invert_yaxis()
    hm.axes.invert_xaxis()

    # Delta xT = DxT colorbar
    cbar = fig.colorbar(hm, ax=ax)
    cbar.set_label('DxT', rotation=270, fontsize=fontsize-2)

    # plot title
    df_teams = produce_df_teams_ref(df)
    teamName = df_teams.loc[df_teams['teamId'] == teamId, 'teamName'].values[0]
    season = df.season.values[0]

    # set title
    ax.set_title(f'{teamName}: {season}', x=0.5, y=0.98, fontsize=fontsize, color='black')

    return fig.savefig(filepath, transparent=transparent, dpi=300)



def plot_delta_xT_multi_pitch(df, events, lst_teams, lst_seasons, lst_cmaps, filepath, transparent=True, eventCol='eventSubType', bins=(18,12), exclCorners=1, figsize=(18,18), fontsize=14, symmetricDeltaVizOverride=1):
    """
    Plotting an array of Delta xT pitch's using the aggregated statistics from the delta_xT_grid function.
    """

    # getting dataframe of match reference data (matchId | homeTeamId | homeTeamName | awayTeamId | awayTeamName)
    df_teams = produce_df_teams_ref(df)

    # creating an array of plotting metadata
    numTeams = len(lst_teams)
    numSeasons = len(lst_seasons)

    ## plotting array
    lst_plots = []
    for teamId, cmap in zip(lst_teams, lst_cmaps):
        for season in lst_seasons:
            lst_plots.append((teamId,season,cmap))

    pitch = Pitch(pitch_type='uefa', pitch_color='white', line_zorder=2, line_color='gray')
    # creating numTeams x numSeasons set of pitches
    fig, axs = pitch.draw(figsize=figsize, nrows=numTeams, ncols=numSeasons)

    for ax, p in zip(axs.flat, lst_plots):

        # unpacking plotting metadata
        teamId, season, cmap = p

        # getting season dataframe
        df_season = df.loc[(df['season'] == season)].copy()

        # getting bin statistic bundle from team_xT_grid function
        bin_statistic_object, mean_matrix_Delta, var_matrix_Delta, numDeltas = delta_xT_grid(df_season, teamId, events, eventCol, bins, exclCorners)

        #fig.patch.set_facecolor('white')

        # Choice of whether to force symmetry between positive and negative delta xT, or to use the natural min and max
        if symmetricDeltaVizOverride == 1:
            vmin, vmax = -0.01, 0.01
        else:
            vmin, vmax = (mean_matrix_Delta).min(), (mean_matrix_Delta).max()

        # producing heatmap
        hm = pitch.heatmap(bin_statistic_object, ax=ax, cmap=cmap, edgecolors='white', vmin=vmin, vmax=vmax)
        # inverting x-  and y- axes so it appears like it's the target team's defence
        hm.axes.invert_yaxis()
        hm.axes.invert_xaxis()

        # Delta xT = DxT colorbar
        cbar = fig.colorbar(hm, ax=ax)
        cbar.set_label('DxT', rotation=270, fontsize=fontsize-2)

        # plot title
        teamName = df_teams.loc[df_teams['teamId'] == teamId, 'teamName'].values[0]

        # set title
        ax.set_title(f'{teamName}: {season}', x=0.5, y=0.98, fontsize=fontsize, color='black')

    return fig.savefig(filepath, transparent=transparent, dpi=300)
