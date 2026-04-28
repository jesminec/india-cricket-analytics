# Performance Analytics and Pattern Intelligence of the Indian Cricket Team Across Formats and Eras

**QM640 V1: Data Analytics Capstone | Walsh College | Fall 2025 Term 3**
**Author:** Jesmine Chaudhuri
**Mentor:** Dr. Keya Choudhury Ganguli

---

## Project Overview

This capstone project systematically applies statistical inference, machine learning, and data visualization techniques to approximately 35 years of Indian cricket data (1990-2024). All five research questions share a unified target variable: Match Outcome (Win = 1, Loss = 0).

### Research Questions

| RQ | Question |
|----|----------|
| RQ1 | Do era-wise batting performance metrics significantly predict match outcome? |
| RQ2 | How do format-specific bowling patterns predict match outcome across Test, ODI, and T20I? |
| RQ3 | Which combination of batting and bowling KPIs most accurately classifies match outcomes? |
| RQ4 | Do toss decision, venue type, and opposition ICC ranking significantly predict match outcome? |
| RQ5 | Can squad-level player career-phase features predict match outcome? |

---

## Repository Structure

```
india-cricket-analytics/
├── README.md
├── requirements.txt
├── environment.yml
├── data/
│   ├── raw/
│   │   ├── cricsheet_tests.csv          # 423,385 rows
│   │   ├── cricsheet_odis.csv           # 281,002 rows
│   │   ├── cricsheet_t20is.csv          # 61,791 rows
│   │   ├── cricsheet_ipl.csv            # 279,126 rows
│   │   ├── espn_india_test_batting.csv
│   │   ├── espn_india_odi_batting.csv
│   │   ├── espn_india_t20_batting.csv
│   │   ├── espn_india_test_bowling.csv
│   │   ├── espn_india_odi_bowling.csv
│   │   └── espn_india_t20_bowling.csv
│   └── processed/
│       └── match_outcomes.csv           # win_loss_flag per match
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_rq1_era_batting_outcome.ipynb
│   ├── 04_rq2_bowling_patterns_outcome.ipynb
│   ├── 05_rq3_multi_kpi_classifier.ipynb
│   ├── 06_rq4_contextual_factors_outcome.ipynb
│   └── 07_rq5_career_phase_outcome.ipynb
├── src/
│   ├── india_cricket_downloader_v4.py
│   ├── features.py
│   └── models.py
└── reports/
    ├── figures/
    └── synopsis_final.pdf
```

---

## Data Sources

- **Cricsheet** (https://cricsheet.org/) — Ball-by-ball CSV2 format, open access
- **ESPNcricinfo Statsguru** (https://stats.espncricinfo.com/) — Innings-level batting and bowling statistics

---

## Setup

### Using pip
```bash
pip install -r requirements.txt
```

### Using conda
```bash
conda env create -f environment.yml
conda activate india-cricket
```

---

## Usage

Run notebooks in order:
```bash
jupyter notebook notebooks/01_data_acquisition.ipynb
```

---

## License

For academic use only. Data sourced from Cricsheet (open access) and ESPNcricinfo Statsguru.
