# CSE558 Midterm Project: Did the Impact Player rule change the IPL?

Group: **Null and Void** (Snehil Modi, Vasu Mehta, Ishaan Dhull, Harshita Garg, Ishika Dhama). Repository: https://github.com/Snehil-Modi04/DSC_Project_IPL_Dataset

This project runs five hypothesis tests on 19 seasons of IPL ball-by-ball data (2008–2026). The data covers 1,243 matches and 295,732 deliveries.

| # | Question | Test | Result (α = 0.05) |
|---|---|---|---|
| H1 | Are first-innings totals higher in the Impact era (2023–26) than in 2018–22? | Welch t-test | +22.3 runs (95% CI 16.9 to 27.7), p ≈ 1.5×10⁻¹⁵: reject H₀ |
| H2 | Do teams hit more sixes per first innings? | Welch t-test | +2.29 sixes (CI 1.63 to 2.95), p ≈ 1.3×10⁻¹¹: reject H₀ |
| H3 | Does the toss winner win more than half of matches? | One-proportion z-test | 51.6% (CI 48.8 to 54.4), p = 0.14: fail to reject H₀ |
| H4 | Has the chasing side's win share changed across three eras? | Chi-square test, 3×2 | 54.4% / 55.8% / 53.0%, p = 0.80: fail to reject H₀ |
| H5 | Does losing 3 or more powerplay wickets lower the batting-first team's win share? | Two-proportion z-test | 23.6% vs 49.4% (−25.8 pp), p ≈ 1.4×10⁻¹⁰: reject H₀ |

## Data

The data is the Kaggle dataset "IPL Ball-by-Ball 2008–2026: All 19 Seasons" (https://www.kaggle.com/datasets/abirambharathisa/ipl-ball-by-ball-2008-2026-all-19-seasons), which is built from Cricsheet (https://cricsheet.org). Two files are used:

- `ipl_ball_by_ball_2008_2026.csv`: one row per delivery, 295,732 rows × 38 columns.
- `ipl_matches_2008_2026.csv`: one row per match, 1,243 rows × 32 columns.

The CSVs are not included in this repository. Download them from Kaggle and place them next to the notebook.

## How to run

```bash
pip install pandas numpy scipy statsmodels matplotlib jupyter
jupyter nbconvert --to notebook --execute ipl_hypothesis_tests.ipynb
```

If the CSVs are stored somewhere else, set the `IPL_BALLS` and `IPL_MATCHES` environment variables to their paths. A full run takes under a minute.

To rebuild the report, run `cd report && xelatex ipl_report.tex`. This needs the Carlito font, the same font the course template uses.

## Files

| File | Contents |
|---|---|
| `ipl_hypothesis_tests.ipynb` | Main notebook, already executed: preparation, EDA and the five tests with the alternative tests compared |
| `ipl_hypothesis_tests.py` | The same code as a plain script (jupytext percent format) |
| `figures/` | All figures, as PDF and PNG |
| `outputs/results.json` | Every number quoted in the report |
| `outputs/prep_table.csv`, `outputs/before_after.csv` | Tables for report sections 2 and 3 |
| `report/ipl_report.tex`, `report/ipl_report.pdf` | The report in the course template layout (7 pages, 10 figures) |

## Method in brief

**Preparation**
- Wides and no-balls are re-bowled with the same ball number, so (match, innings, over, ball) repeats for 11,075 deliveries. These are kept, and `delivery_in_over` is used as the key instead.
- The 175 super-over balls are removed.
- Empty extra and wicket fields mean "none" (no extra, no wicket), not "unknown", and are filled that way.
- The DRS review columns are 99.7% empty and are dropped.
- "Retired hurt" is not counted as a wicket.
- Sixes are counted with the `is_six` flag, not by runs = 6.
- 60 matches are left out of the score and chase tests: 9 no-results, 16 ties, 23 D/L results and other shortened games.
- Totals from the ball file match the match file exactly.

**Encoding**
- After encoding, the ball table is 295,557 × 114 = 33.7M cells.

**Unit of analysis**
- Deliveries within an innings are not independent, so no test is run on single balls.
- H1, H2 and H5 use one first innings per match. H3 and H4 use one row per match.

**Robustness checks**

The five hypotheses, their sides and their thresholds were written down before any outcome in the data was examined. That is the safeguard against choosing hypotheses to fit the data. On top of the main tests, each one is challenged by its strongest alternative explanation:

- **H1: rule or trend?**
  - Segmented regression with a season trend: +18.1 runs beyond the trend (11.9 to 24.3).
  - Standard errors clustered by season.
  - Venue fixed effects: +19.5.
  - Placebo rule years 2011–2021: no placebo jump exceeds 7.7 runs.
  - India-only matches.
- **H2:** the six rate per 120 balls, and a negative-binomial regression with an offset. The rate ratio is 1.33, and 1.18 net of the trend.
- **H3:**
  - TOST equivalence test: the toss effect lies within ±5 pp, p = 0.008.
  - Choosing to field first versus bat first: +9.4 pp, odds ratio 1.48.
- **H4:** logistic regression on target and era. At the same target, Impact-era chasers have about twice the odds of winning (OR 1.97), which offsets the higher targets.
- **H5:**
  - Logistic regression adjusted for venue and trend: OR 0.61 per wicket.
  - Wickets × era interaction: OR 1.07, p = 0.62. Deeper batting does not soften the cost of a collapse.
- **Across all five tests:** a Holm correction leaves every decision unchanged.
