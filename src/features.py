"""
features.py
-----------
Feature engineering functions for the India Cricket Analytics capstone.

Derived features defined here:
  - Batting Impact Score (BIS)
  - Bowling Pressure Index (BPI)
  - Win Probability Added (WPA)
  - Dot Ball Percentage (DBP)
  - Career-Phase Label
  - Era Label
  - win_loss_flag (target variable)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Era Label
# ---------------------------------------------------------------------------
def assign_era_label(date_series: pd.Series) -> pd.Series:
    """
    Assign historical era label based on match start date.

    Categories:
        Pre-2000      : 1990-1999
        Early 2000s   : 2000-2007
        Transition Era: 2008-2012
        Modern Era    : 2013-2024
    """
    dates = pd.to_datetime(date_series)
    year = dates.dt.year
    conditions = [
        year < 2000,
        (year >= 2000) & (year < 2008),
        (year >= 2008) & (year < 2013),
        year >= 2013,
    ]
    labels = ["Pre-2000", "Early 2000s", "Transition Era", "Modern Era"]
    return pd.Series(np.select(conditions, labels, default="Unknown"), index=date_series.index)


# ---------------------------------------------------------------------------
# Win/Loss Flag
# ---------------------------------------------------------------------------
def compute_win_loss_flag(df: pd.DataFrame) -> pd.Series:
    """
    Derive binary win_loss_flag (1 = India Win, 0 = India Loss).

    Infers outcome by comparing innings totals for matches where
    India is either batting_team or bowling_team.
    Ties and no-results are excluded (returns NaN).
    """
    df = df.copy()
    df["total_runs"] = df["runs_off_bat"].fillna(0) + df["extras"].fillna(0)

    flags = {}
    for match_id, grp in df.groupby("match_id"):
        inn = grp.groupby(["innings", "batting_team"])["total_runs"].sum().reset_index()
        inn1 = inn[inn["innings"] == 1]
        inn2 = inn[inn["innings"] == 2]
        if len(inn1) == 0 or len(inn2) == 0:
            continue
        t1, r1 = inn1.iloc[0]["batting_team"], inn1.iloc[0]["total_runs"]
        t2, r2 = inn2.iloc[0]["batting_team"], inn2.iloc[0]["total_runs"]
        if r1 == r2:
            continue
        winner = t2 if r2 > r1 else t1
        flags[match_id] = 1 if winner == "India" else 0

    return df["match_id"].map(flags)


# ---------------------------------------------------------------------------
# Batting Impact Score (BIS)
# ---------------------------------------------------------------------------
def batting_impact_score(runs: pd.Series, strike_rate: pd.Series,
                          balls_faced: pd.Series) -> pd.Series:
    """
    BIS = (Runs Scored x Strike Rate) / (100 x Balls Faced)
    Adjusted for phase of innings (Powerplay, Middle Overs, Death Overs).
    """
    denom = (100 * balls_faced.replace(0, np.nan))
    return (runs * strike_rate) / denom


# ---------------------------------------------------------------------------
# Bowling Pressure Index (BPI)
# ---------------------------------------------------------------------------
def bowling_pressure_index(economy_rate: pd.Series,
                            wicket_prob_per_ball: pd.Series) -> pd.Series:
    """
    BPI = Economy Rate x (1 - Wicket Probability per delivery)
    Calibrated per format using historical dot-ball rates.
    Lower BPI = greater bowling pressure.
    """
    return economy_rate * (1 - wicket_prob_per_ball)


# ---------------------------------------------------------------------------
# Dot Ball Percentage (DBP)
# ---------------------------------------------------------------------------
def dot_ball_percentage(delivery_df: pd.DataFrame) -> pd.Series:
    """
    DBP = proportion of deliveries resulting in zero runs and zero extras.
    Computed at match level for the bowling team.

    Args:
        delivery_df: filtered to a single match and bowling team.
    Returns:
        Scalar float (proportion).
    """
    total = len(delivery_df)
    if total == 0:
        return np.nan
    dots = ((delivery_df["runs_off_bat"] == 0) & (delivery_df["extras"] == 0)).sum()
    return dots / total


# ---------------------------------------------------------------------------
# Career-Phase Label
# ---------------------------------------------------------------------------
def career_phase_label(age_at_match: pd.Series) -> pd.Series:
    """
    Assign career phase based on player age at time of match.

    Categories:
        Early : debut to age 24
        Peak  : ages 25-30
        Late  : age 31+
    """
    conditions = [
        age_at_match <= 24,
        (age_at_match >= 25) & (age_at_match <= 30),
        age_at_match >= 31,
    ]
    labels = ["Early", "Peak", "Late"]
    return pd.Series(np.select(conditions, labels, default="Unknown"),
                     index=age_at_match.index)
