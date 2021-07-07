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


def plot_xT_pitch(df, events, filepath, transparent=True, teamOrPlayer='team', eventCol='eventSubType', bins=(18,12), figsize=(16,9), fontsize=14, vmax_override=None, cmap=cm.coolwarm, scatter=1):
    """
    Plotting an xT pitch using the aggregated statistics from the team_xT_grid function.

    Optional argument to plot a scatter of the underlying events.
    """

    # getting bin statistic bundle from team_xT_grid function
    bs, bs_obj, df_events = xT_grid(df, events, eventCol, bins)

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
    team_cbar.set_label('xT', rotation=270, fontsize=fontsize)

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


def plot_xT_multi_pitch(df, events, lst_teams, lst_seasons, lst_cmaps, filepath, transparent=True, teamOrPlayer='team', eventCol='eventSubType', bins=(18,12), figsize=(18,18), fontsize=14, vmax_override=None, scatter=1):
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
        bs, bs_obj, df_events = xT_grid(df_plot, events, eventCol, bins)

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
        team_cbar.set_label('xT', rotation=270, fontsize=fontsize)

        # generating plot title
        ## getting team name and season name from df_events dataframe
        df_teams = produce_df_teams_ref(df)
        teamName = df_teams.loc[df_teams['teamId'] == teamId].teamName.values[0]

        if teamOrPlayer == 'team':
            ax.set_title(f'{teamName}: {season}', x=0.5, y=0.98, fontsize=fontsize, color='black')
        elif teamOrPlayer == 'player':
            ax.set_title(f'{playerName} ({teamName}): {season}', x=0.5, y=0.98, fontsize=fontsize, color='black')

    return fig.savefig(filepath, transparent=transparent, dpi=300)
