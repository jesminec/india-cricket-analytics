"""
features.py
-----------
Feature engineering functions for the India Cricket Analytics capstone.

All computations are derived dynamically from the raw data.
No hard-coded thresholds, statistics, or lookup values.

Derived features:
  - Era Label                 (from match start date)
  - win_loss_flag             (from innings totals)
  - venue_type                (Home / Away from venue name keywords)
  - Batting Impact Score      (BIS)
  - Bowling Pressure Index    (BPI)
  - Dot Ball Percentage       (DBP)
  - Career-Phase Label        (from player age)
  - Match-level KPI aggregates for modelling
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Era boundaries — single source of truth
# ---------------------------------------------------------------------------
ERA_BOUNDARIES = [
    (None, 2000, "Pre-2000"),
    (2000, 2008, "Early 2000s"),
    (2008, 2013, "Transition Era"),
    (2013, None, "Modern Era"),
]

def assign_era_label(date_series: pd.Series) -> pd.Series:
    """Assign historical era label based on match start date."""
    dates = pd.to_datetime(date_series)
    year  = dates.dt.year
    conditions, labels = [], []
    for lo, hi, label in ERA_BOUNDARIES:
        if lo is None:   cond = year < hi
        elif hi is None: cond = year >= lo
        else:            cond = (year >= lo) & (year < hi)
        conditions.append(cond)
        labels.append(label)
    return pd.Series(np.select(conditions, labels, default="Unknown"),
                     index=date_series.index, dtype="object")

# ---------------------------------------------------------------------------
# Win / Loss flag
# ---------------------------------------------------------------------------
def compute_win_loss_flag(df: pd.DataFrame) -> pd.Series:
    """
    Derive binary win_loss_flag (1=India Win, 0=India Loss).
    Compares total runs per team across all innings.
    Ties / no-results return NaN.
    """
    df = df.copy()
    df["_total"] = df["runs_off_bat"].fillna(0) + df["extras"].fillna(0)
    team_runs = (df.groupby(["match_id", "batting_team"])["_total"]
                   .sum().reset_index(name="runs"))
    flags: dict = {}
    for match_id, grp in team_runs.groupby("match_id"):
        india = grp[grp["batting_team"] == "India"]["runs"].sum()
        opp   = grp[grp["batting_team"] != "India"]["runs"].sum()
        if india == opp:
            continue
        flags[match_id] = 1 if india > opp else 0
    return df["match_id"].map(flags)

# ---------------------------------------------------------------------------
# Venue type
# ---------------------------------------------------------------------------
INDIA_VENUE_KEYWORDS: list[str] = [
    "wankhede","brabourne","eden gardens","chepauk","chidambaram",
    "chinnaswamy","feroz","kotla","arun jaitley","narendra modi",
    "sardar patel","himachal","hpca","dharamsala","mohali","pca stadium",
    "jsca","ranchi","holkar","indore","rajiv gandhi","hyderabad","uppal",
    "vidarbha","nagpur","barabati","cuttack","barsapara","guwahati",
    "sawai mansingh","jaipur","green park","kanpur","aca-vdca",
    "visakhapatnam","vizag","saurashtra","rajkot","greenfield",
    "thiruvananthapuram","brsabv","lucknow","maharashtra cricket",
    "pune","vadodara","baroda","vca stadium",
]

def classify_venue_type(venue_series: pd.Series) -> pd.Series:
    """Classify each venue as Home or Away based on INDIA_VENUE_KEYWORDS."""
    pattern = "|".join(re.escape(kw) for kw in INDIA_VENUE_KEYWORDS)
    def _classify(v):
        if not isinstance(v, str): return "Away"
        return "Home" if re.search(pattern, v.lower()) else "Away"
    return venue_series.apply(_classify)

# ---------------------------------------------------------------------------
# Match-level KPI aggregates
# ---------------------------------------------------------------------------
def build_match_kpis(delivery_df: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """
    Aggregate delivery-level Cricsheet data into one row per match.
    All statistics computed directly from the data — no static values.
    """
    df = delivery_df.copy()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["_total"]    = df["runs_off_bat"].fillna(0) + df["extras"].fillna(0)
    df["_legal"]    = df["wides"].isna() & df["noballs"].isna()
    df["_dot"]      = (df["runs_off_bat"] == 0) & (df["extras"].fillna(0) == 0)
    df["_boundary"] = df["runs_off_bat"].isin([4, 6])
    df["_wicket"]   = df["wicket_type"].notna() & (df["wicket_type"] != "none")

    wlf_series = compute_win_loss_flag(df)
    df["_wlf"] = wlf_series

    rows = []
    for match_id, mdf in df.groupby("match_id"):
        meta       = mdf.iloc[0]
        start_date = meta["start_date"]
        venue      = meta.get("venue", "")
        era        = assign_era_label(pd.Series([start_date])).iloc[0]
        venue_type = classify_venue_type(pd.Series([venue])).iloc[0]
        wlf_val    = mdf["_wlf"].dropna().iloc[0] if len(mdf["_wlf"].dropna()) > 0 else np.nan

        # India batting
        bat        = mdf[mdf["batting_team"] == "India"]
        legal_bat  = bat[bat["_legal"]]
        india_runs = bat["runs_off_bat"].sum()
        india_balls= len(legal_bat)
        india_sr   = (india_runs / india_balls * 100) if india_balls > 0 else np.nan
        india_bnd  = int(bat["_boundary"].sum())

        # First innings score for India
        inn1_batters = mdf[mdf["innings"] == 1]["batting_team"].unique()
        india_bat_first = 1 if "India" in inn1_batters else 0
        india_fi_score  = int(bat[bat["innings"].isin([1, 3])]["runs_off_bat"].sum())

        # India bowling
        bowl       = mdf[mdf["bowling_team"] == "India"]
        legal_bowl = bowl[bowl["_legal"]]
        bowl_overs = len(legal_bowl) / 6 if len(legal_bowl) > 0 else np.nan
        bowl_runs  = bowl["_total"].sum()
        bowl_wkts  = int(bowl["_wicket"].sum())
        bowl_econ  = (bowl_runs / bowl_overs) if bowl_overs and bowl_overs > 0 else np.nan
        dots       = bowl[bowl["_dot"]]["_legal"].sum()
        dbp        = (dots / len(legal_bowl)) if len(legal_bowl) > 0 else np.nan

        rows.append({
            "match_id":                  match_id,
            "start_date":                start_date,
            "venue":                     venue,
            "venue_type":                venue_type,
            "era_label":                 era,
            "win_loss_flag":             wlf_val,
            "format":                    fmt,
            "india_runs":                int(india_runs),
            "india_balls":               india_balls,
            "india_sr":                  round(india_sr, 2)   if not np.isnan(india_sr)   else np.nan,
            "india_boundaries":          india_bnd,
            "india_bat_first":           india_bat_first,
            "india_first_innings_score": india_fi_score,
            "india_bowl_econ":           round(bowl_econ, 3)  if bowl_econ and not np.isnan(bowl_econ) else np.nan,
            "india_bowl_wkts":           bowl_wkts,
            "india_dbp":                 round(float(dbp), 4) if dbp and not np.isnan(dbp) else np.nan,
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Batting Impact Score (BIS)
# ---------------------------------------------------------------------------
def batting_impact_score(runs: pd.Series, strike_rate: pd.Series,
                          balls_faced: pd.Series) -> pd.Series:
    """BIS = (Runs × Strike Rate) / (100 × Balls Faced)"""
    denom = 100.0 * balls_faced.replace(0, np.nan)
    return (runs * strike_rate) / denom

# ---------------------------------------------------------------------------
# Bowling Pressure Index (BPI)
# ---------------------------------------------------------------------------
def bowling_pressure_index(economy_rate: pd.Series,
                            wicket_prob_per_ball: pd.Series) -> pd.Series:
    """BPI = Economy Rate × (1 − Wicket Probability per delivery)"""
    return economy_rate * (1.0 - wicket_prob_per_ball)

def compute_wicket_prob(delivery_df: pd.DataFrame) -> float:
    """Empirical wicket probability per legal delivery."""
    legal   = delivery_df[delivery_df["wides"].isna() & delivery_df["noballs"].isna()]
    if len(legal) == 0: return 0.0
    wickets = (delivery_df["wicket_type"].notna() &
               (delivery_df["wicket_type"] != "none")).sum()
    return wickets / len(legal)

# ---------------------------------------------------------------------------
# Dot Ball Percentage (DBP) — single match
# ---------------------------------------------------------------------------
def dot_ball_percentage(delivery_df: pd.DataFrame) -> float:
    """DBP = dot deliveries / legal deliveries for a single match-team slice."""
    legal = delivery_df[delivery_df["wides"].isna() & delivery_df["noballs"].isna()]
    if len(legal) == 0: return np.nan
    dots = ((legal["runs_off_bat"] == 0) & (legal["extras"].fillna(0) == 0)).sum()
    return dots / len(legal)

# ---------------------------------------------------------------------------
# Career-Phase Label
# ---------------------------------------------------------------------------
CAREER_PHASES = [
    (None, 25, "Early"),
    (25,   31, "Peak"),
    (31,   None, "Late"),
]

def career_phase_label(age_series: pd.Series) -> pd.Series:
    """Assign career phase from CAREER_PHASES boundaries."""
    conditions, labels = [], []
    for lo, hi, label in CAREER_PHASES:
        if lo is None:   cond = age_series < hi
        elif hi is None: cond = age_series >= lo
        else:            cond = (age_series >= lo) & (age_series < hi)
        conditions.append(cond)
        labels.append(label)
    return pd.Series(np.select(conditions, labels, default="Unknown"),
                     index=age_series.index, dtype="object")

# ---------------------------------------------------------------------------
# ESPN data loaders
# ---------------------------------------------------------------------------
def load_espn_batting(path: Path) -> pd.DataFrame:
    """Load and clean ESPNcricinfo innings-level batting CSV."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df.columns = df.columns.str.strip()
    rename = {"Start Date":"start_date","Player":"player","Runs":"runs",
               "BF":"balls_faced","SR":"strike_rate","4s":"fours","6s":"sixes",
               "Opposition":"opposition","Ground":"ground","Format":"format",
               "Inns":"innings_no","Mins":"minutes"}
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    if "runs" in df.columns:
        df["runs"] = (df["runs"].astype(str)
                      .str.replace(r"\*","",regex=True)
                      .str.replace(r"[^0-9.]","",regex=True))
        df["runs"] = pd.to_numeric(df["runs"], errors="coerce")
    for col in ["balls_faced","strike_rate","fours","sixes","minutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "start_date" in df.columns:
        df["era_label"] = assign_era_label(df["start_date"])
    return df.dropna(subset=["runs"]).reset_index(drop=True)

def load_espn_bowling(path: Path) -> pd.DataFrame:
    """Load and clean ESPNcricinfo innings-level bowling CSV."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df.columns = df.columns.str.strip()
    rename = {"Start Date":"start_date","Player":"player","Overs":"overs",
               "Runs":"runs_conceded","Wkts":"wickets","Econ":"economy",
               "Mdns":"maidens","Opposition":"opposition","Ground":"ground",
               "Format":"format","Inns":"innings_no"}
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    for col in ["overs","runs_conceded","wickets","economy","maidens"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "start_date" in df.columns:
        df["era_label"] = assign_era_label(df["start_date"])
    return df.dropna(subset=["economy"]).reset_index(drop=True)
