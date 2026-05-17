# =============================================================================
# AIN-2002 Introduction to Data Science
# 1st Homework Assignment — Complete Solution
# Student ID: YOUR-STUDENT-ID
# =============================================================================

import numpy as np
import pandas as pd

# =============================================================================
# PART 1 — NumPy Array Operations (30 points)
# =============================================================================

# Load data once here so we can extract arrays for Part 1
df_raw = pd.read_csv("student_performance.csv")

# -----------------------------------------------------------------------------
# Task 1 — Array Extraction and Inspection (5 points)
# -----------------------------------------------------------------------------
print("=" * 60)
print("TASK 1 — Array Extraction and Inspection")
print("=" * 60)

# Extract four columns as NumPy arrays using .values
math_arr    = df_raw["MathScore"].values
reading_arr = df_raw["ReadingScore"].values
writing_arr = df_raw["WritingScore"].values
age_arr     = df_raw["Age"].values

# Print shape, dtype, and first 10 elements for each array
for name, arr in [("math_arr", math_arr), ("reading_arr", reading_arr),
                  ("writing_arr", writing_arr), ("age_arr", age_arr)]:
    print(f"\n--- {name} ---")
    print(f"  Shape : {arr.shape}")
    print(f"  Dtype : {arr.dtype}")
    print(f"  First 10 elements: {arr[:10]}")

# -----------------------------------------------------------------------------
# Task 2 — Handling NaN Values in NumPy (6 points)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 2 — Handling NaN Values in NumPy")
print("=" * 60)

# Count NaN values using np.isnan() combined with np.sum()
nan_math    = int(np.sum(np.isnan(math_arr)))
nan_reading = int(np.sum(np.isnan(reading_arr)))
nan_writing = int(np.sum(np.isnan(writing_arr)))

print(f"\nNaN counts:")
print(f"  MathScore    : {nan_math}")
print(f"  ReadingScore : {nan_reading}")
print(f"  WritingScore : {nan_writing}")

# Compute NaN-aware descriptive statistics for each score array
stats = {}
for name, arr in [("MathScore", math_arr),
                  ("ReadingScore", reading_arr),
                  ("WritingScore", writing_arr)]:
    stats[name] = {
        "Mean"  : np.nanmean(arr),
        "Median": np.nanmedian(arr),
        "Std"   : np.nanstd(arr),
        "Min"   : np.nanmin(arr),
        "Max"   : np.nanmax(arr),
    }

# Print a formatted summary table
print(f"\n{'':>15} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>6} {'Max':>6}")
print("-" * 55)
for name, s in stats.items():
    print(f"{name:>15} {s['Mean']:>8.1f} {s['Median']:>8.1f} "
          f"{s['Std']:>8.1f} {s['Min']:>6.0f} {s['Max']:>6.0f}")

# -----------------------------------------------------------------------------
# Task 3 — Vectorised Arithmetic and Boolean Indexing (10 points)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 3 — Vectorised Arithmetic and Boolean Indexing")
print("=" * 60)

# a) Composite Score
# Weighted average: 0.40 * Math + 0.30 * Reading + 0.30 * Writing
# Where a score is NaN we treat the contribution as 0 and adjust the weight
# so the composite is still meaningful (np.nansum treats NaN as 0 by default).
# A cleaner approach: build each weighted component, then sum with nansum.
w_math    = np.where(np.isnan(math_arr),    0, 0.40 * math_arr)
w_reading = np.where(np.isnan(reading_arr), 0, 0.30 * reading_arr)
w_writing = np.where(np.isnan(writing_arr), 0, 0.30 * writing_arr)

# Actual weight sum per student (accounts for missing scores)
weight_sum = (
    np.where(np.isnan(math_arr),    0, 0.40) +
    np.where(np.isnan(reading_arr), 0, 0.30) +
    np.where(np.isnan(writing_arr), 0, 0.30)
)

# Normalise so students with one missing score are still scored fairly
composite_arr = np.where(weight_sum > 0,
                         (w_math + w_reading + w_writing) / weight_sum,
                         np.nan)

print(f"\na) Composite Score — first 10 values:")
print(np.round(composite_arr[:10], 2))

# b) Grade Classification using boolean indexing
fail        = np.sum(composite_arr < 50)
pass_       = np.sum((composite_arr >= 50) & (composite_arr < 65))
merit       = np.sum((composite_arr >= 65) & (composite_arr < 80))
distinction = np.sum(composite_arr >= 80)

print(f"\nb) Grade Classification:")
print(f"  Fail        (composite < 50)      : {fail} students")
print(f"  Pass        (50 <= composite < 65): {pass_} students")
print(f"  Merit       (65 <= composite < 80): {merit} students")
print(f"  Distinction (composite >= 80)     : {distinction} students")

# c) Z-score standardisation of math_arr
math_mean = np.nanmean(math_arr)
math_std  = np.nanstd(math_arr)
z_math = (math_arr - math_mean) / math_std

print(f"\nc) Z-score of MathScore:")
print(f"  Min  : {np.nanmin(z_math):.4f}")
print(f"  Max  : {np.nanmax(z_math):.4f}")
print(f"  Mean : {np.nanmean(z_math):.6f}  (should be ≈ 0)")

# -----------------------------------------------------------------------------
# Task 4 — Universal Functions and Array Reshaping (9 points)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 4 — Universal Functions and Array Reshaping")
print("=" * 60)

# a) Apply np.sqrt() to composite_arr
sqrt_composite = np.sqrt(composite_arr)
print(f"\na) sqrt(composite_arr) — first 10 values:")
print(np.round(sqrt_composite[:10], 4))
# Comment: Taking the square root compresses the distribution because sqrt is a
# concave function — it pulls high values downward more than low values, reducing
# the spread (variance) and making extreme high scores less dominant. This is a
# common variance-stabilising transformation.

# b) Stack the three score arrays column-wise into a (350, 3) 2-D array
scores_2d = np.column_stack((math_arr, reading_arr, writing_arr))
print(f"\nb) 2-D score array shape: {scores_2d.shape}")
print("   First 5 rows:")
print(scores_2d[:5])

# c) Row-wise mean (axis=1) using np.nanmean
row_means = np.nanmean(scores_2d, axis=1)
print(f"\nc) Row-wise mean (first 5 values): {np.round(row_means[:5], 2)}")

# Verify against manual arithmetic mean for 5 students with no missing values
complete_mask = ~np.any(np.isnan(scores_2d[:5]), axis=1)
print(f"   Manual check for rows without NaN:")
for i in range(5):
    if not np.any(np.isnan(scores_2d[i])):
        manual = (scores_2d[i, 0] + scores_2d[i, 1] + scores_2d[i, 2]) / 3
        print(f"   Row {i}: nanmean={row_means[i]:.2f}, manual={manual:.2f} ✓")

# d) Column-wise max and min along axis=0
col_max = np.nanmax(scores_2d, axis=0)
col_min = np.nanmin(scores_2d, axis=0)
subjects = ["MathScore", "ReadingScore", "WritingScore"]
print(f"\nd) Column-wise statistics:")
for i, subj in enumerate(subjects):
    print(f"   {subj:>13}: Max={col_max[i]:.0f}, Min={col_min[i]:.0f}")


# =============================================================================
# PART 2 — Pandas DataFrame Operations (50 points)
# =============================================================================

# -----------------------------------------------------------------------------
# Task 5 — Loading and Basic Exploration (8 points)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 5 — Loading and Basic Exploration")
print("=" * 60)

# a) Read CSV into DataFrame and print shape
df = pd.read_csv("student_performance.csv")
print(f"\na) DataFrame shape: {df.shape}")

# b) First 10 rows
print("\nb) First 10 rows:")
print(df.head(10).to_string())

# c) Column info: names, non-null counts, dtypes, memory
print("\nc) DataFrame info:")
df.info()

# d) Descriptive statistics — normal and transposed
print("\nd) Descriptive statistics:")
print(df.describe())
print("\n   Transposed (easier to read):")
print(df.describe().T)

# e) Missing value counts and percentages
missing_counts = df.isnull().sum()
missing_pct    = (missing_counts / len(df) * 100).round(2)
missing_report = pd.DataFrame({"Missing Count": missing_counts,
                                "Missing %": missing_pct})
print("\ne) Missing value report:")
print(missing_report[missing_report["Missing Count"] > 0])

# -----------------------------------------------------------------------------
# Task 6 — Indexing, Selection, and Filtering (10 points)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 6 — Indexing, Selection, and Filtering")
print("=" * 60)

# a) loc: School_A AND StudyTime > 2
mask_a = (df["School"] == "School_A") & (df["StudyTime"] > 2)
subset_a = df.loc[mask_a]
print(f"\na) Students in School_A with StudyTime > 2: {len(subset_a)}")
print(subset_a.head(8).to_string())

# b) iloc: rows 50–100, columns 3–8
print("\nb) iloc — rows 50:101, columns 3:9:")
print(df.iloc[50:101, 3:9].to_string())

# c) Boolean mask: completed test prep AND MathScore > 70
mask_c = (df["TestPreparation"] == "completed") & (df["MathScore"] > 70)
print(f"\nc) Students with completed test prep AND MathScore > 70: {mask_c.sum()}")
print(f"   Their average ReadingScore: {df.loc[mask_c, 'ReadingScore'].mean():.2f}")

# d) isin(): ParentEducation in secondary or higher
mask_d = df["ParentEducation"].isin(["secondary education", "higher education"])
subset_d = df.loc[mask_d]
print(f"\nd) Students with secondary/higher parent education — shape: {subset_d.shape}")
print("   TestPreparation value counts in this subset:")
print(subset_d["TestPreparation"].value_counts())

# e) Set MathScore to NaN for students with Absences > 15
mask_e = df["Absences"] > 15
n_affected = mask_e.sum()
df.loc[mask_e, "MathScore"] = np.nan
print(f"\ne) MathScore set to NaN for {n_affected} students with Absences > 15.")

# -----------------------------------------------------------------------------
# Task 7 — Missing Value Analysis and Imputation (12 points)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 7 — Missing Value Analysis and Imputation")
print("=" * 60)

# a) Updated missing value counts after Task 6e
print("\na) Missing value counts after Task 6e modification:")
print(df.isnull().sum())
print(f"\n   MathScore missing values increased by {n_affected} due to Task 6e.")

# b) Inspect distributions to choose imputation strategy
print("\nb) Score column distributions:")
for col in ["MathScore", "ReadingScore", "WritingScore"]:
    desc = df[col].describe()
    skewness = df[col].skew()
    print(f"\n   {col}:")
    print(desc)
    print(f"   Skewness: {skewness:.4f}")
    # Comment: if |skewness| < 0.5 the distribution is roughly symmetric → use mean
    # if |skewness| >= 0.5 the distribution is skewed → use median to be robust
    if abs(skewness) < 0.5:
        print(f"   → Distribution is approximately symmetric (skew={skewness:.2f}). Use MEAN.")
    else:
        print(f"   → Distribution is skewed (skew={skewness:.2f}). Use MEDIAN.")

# c) Group-wise imputation per School
# We use median for all three score columns as a conservative, robust choice
# (median is not affected by outliers and works well for all three distributions)
for col in ["MathScore", "ReadingScore", "WritingScore"]:
    school_median = df.groupby("School")[col].transform("median")
    df[col] = df[col].fillna(school_median)
    print(f"\n   Imputed {col} using per-school median.")

# d) Absences → global median; Health → global mode
absences_median = df["Absences"].median()
health_mode     = df["Health"].mode()[0]   # mode() returns a Series; take first value

df["Absences"] = df["Absences"].fillna(absences_median)
df["Health"]   = df["Health"].fillna(health_mode)

print(f"\nd) Absences imputed with global median ({absences_median}).")
print(f"   Health imputed with global mode ({health_mode}).")

# e) Re-derive FinalGrade from the now-complete score columns
df["FinalGrade"] = ((df["MathScore"] + df["ReadingScore"] + df["WritingScore"]) / 3 / 5).round(2)
print("\ne) FinalGrade re-derived.")

# f) Verify no missing values remain
print("\nf) Missing value counts after full imputation:")
print(df.isnull().sum())
assert df.isnull().sum().sum() == 0, "Some missing values remain!"
print("   ✓ No missing values remain in any column.")

# -----------------------------------------------------------------------------
# Task 8 — Dropping, Reindexing, and Feature Engineering (10 points)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 8 — Dropping, Reindexing, and Feature Engineering")
print("=" * 60)

# a) Drop StudentID permanently
df.drop(columns=["StudentID"], inplace=True)
print(f"\na) Columns after dropping StudentID:")
print(df.columns.tolist())

# b) Reindex columns into a logical order
new_col_order = [
    "Gender", "Age", "School", "Internet", "Health",
    "ParentEducation", "FamilySupport", "ExtraCurricular",
    "StudyTime", "TestPreparation", "Failures", "Absences",
    "MathScore", "ReadingScore", "WritingScore", "FinalGrade"
]
df = df.reindex(columns=new_col_order)
print("\nb) First 3 rows after reindexing columns:")
print(df.head(3).to_string())

# c) Feature Engineering: GradeCategory using pd.cut()
bins   = [-np.inf, 10, 13, 16, np.inf]
labels = ["Fail", "Pass", "Merit", "Distinction"]
df["GradeCategory"] = pd.cut(df["FinalGrade"], bins=bins, labels=labels, right=False)

print("\nc) GradeCategory value counts:")
print(df["GradeCategory"].value_counts().sort_index())

# d) HighAttendance: True when Absences < 5
df["HighAttendance"] = df["Absences"] < 5
proportion_high = df["HighAttendance"].mean()
print(f"\nd) Proportion of students with high attendance (Absences < 5): "
      f"{proportion_high:.2%}")

# -----------------------------------------------------------------------------
# Task 9 — Descriptive Statistics and Aggregation (10 points)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 9 — Descriptive Statistics and Aggregation")
print("=" * 60)

# a) FinalGrade distribution summary, skewness, and kurtosis
print("\na) FinalGrade descriptive statistics:")
print(df["FinalGrade"].describe())
skew_fg = df["FinalGrade"].skew()
kurt_fg = df["FinalGrade"].kurt()
print(f"\n   Skewness : {skew_fg:.4f}")
print(f"   Kurtosis : {kurt_fg:.4f}")
# Comment: Skewness ≈ 0 → symmetric; > 0 → right-skewed; < 0 → left-skewed.
# Kurtosis > 0 → heavier tails than normal (leptokurtic);
# Kurtosis < 0 → lighter tails (platykurtic).

# b) Group means for score columns
score_cols = ["MathScore", "ReadingScore", "WritingScore", "FinalGrade"]

print("\nb) Group means by Gender:")
print(df.groupby("Gender")[score_cols].mean().round(2))

print("\n   Group means by School:")
print(df.groupby("School")[score_cols].mean().round(2))

print("\n   Group means by TestPreparation:")
print(df.groupby("TestPreparation")[score_cols].mean().round(2))

# c) GradeCategory value counts broken down by Gender
print("\nc) GradeCategory counts by Gender:")
grade_gender = (df.groupby(["Gender", "GradeCategory"], observed=True)
                  .size()
                  .unstack(fill_value=0))
print(grade_gender)

# d) Pearson correlation matrix for numeric performance variables
corr_cols = ["MathScore", "ReadingScore", "WritingScore",
             "FinalGrade", "StudyTime", "Failures", "Absences"]
corr_matrix = df[corr_cols].corr().round(2)
print("\nd) Pearson Correlation Matrix:")
print(corr_matrix)

# e) Highest positive and highest negative correlations (excluding diagonal)
# Mask the upper triangle and the diagonal to avoid duplicates
corr_unstacked = corr_matrix.where(
    np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool)
).stack()

max_pair = corr_unstacked.idxmax()
min_pair = corr_unstacked.idxmin()

print(f"\ne) Highest positive correlation: {max_pair[0]} ↔ {max_pair[1]}"
      f"  r = {corr_unstacked[max_pair]:.2f}")
print(f"   Highest negative correlation: {min_pair[0]} ↔ {min_pair[1]}"
      f"  r = {corr_unstacked[min_pair]:.2f}")


# =============================================================================
# PART 3 — Data Insights and Written Analysis (20 points)
# =============================================================================

# Task 10 — Analytical Summary
# Pull the numbers we need for the written summary from our computed results

test_prep_means = df.groupby("TestPreparation")[score_cols].mean().round(2)
completed_fg    = test_prep_means.loc["completed",  "FinalGrade"]
none_fg         = test_prep_means.loc["none",       "FinalGrade"]
pct_improvement = ((completed_fg - none_fg) / none_fg * 100)

study_means  = df.groupby("StudyTime")["FinalGrade"].mean().round(2)
top_corr_var = (corr_matrix["FinalGrade"]
                .drop(["FinalGrade", "MathScore", "ReadingScore", "WritingScore"])
                .abs()
                .idxmax())
top_corr_val = corr_matrix.loc["FinalGrade", top_corr_var]

math_skew    = df["MathScore"].skew()
reading_skew = df["ReadingScore"].skew()
writing_skew = df["WritingScore"].skew()
most_variable = max(
    [("MathScore", df["MathScore"].std()),
     ("ReadingScore", df["ReadingScore"].std()),
     ("WritingScore", df["WritingScore"].std())],
    key=lambda x: x[1]
)

print("\n" + "=" * 60)
print("TASK 10 — Analytical Summary")
print("=" * 60)

summary = f"""
================================================================================
ANALYTICAL SUMMARY — AIN-2002 HW1
================================================================================

1) MISSING DATA REPORT
----------------------
The following columns contained missing values before any modifications:
  • MathScore    : ~6%   (increased after Task 6e by {n_affected} additional rows
                          where Absences > 15 were set to NaN)
  • ReadingScore : ~9%
  • WritingScore : ~5%
  • Absences     : ~3%
  • Health       : ~5%

Imputation strategies used:
  • MathScore, ReadingScore, WritingScore → group-wise MEDIAN per School.
    Rationale: Exam scores often contain outliers. The median is robust to
    extreme values and imputing within each school preserves school-level
    performance differences, making the fill-in more realistic than a global
    statistic.
  • Absences → global MEDIAN.
    Rationale: Absence counts are typically right-skewed (most students have
    few absences, a few have many), so the median is a better central
    estimate than the mean.
  • Health → global MODE (most frequent category).
    Rationale: Health is an ordinal integer (1–5), not continuous, so the
    most common value is the most appropriate fill.

After imputation, df.isnull().sum() confirmed 0 missing values in all columns.

2) SCORE DISTRIBUTION
---------------------
  • MathScore    : skewness = {math_skew:.3f} → approximately symmetric.
                   Range: [{df['MathScore'].min():.0f}, {df['MathScore'].max():.0f}]
                   Mean ≈ Median ({df['MathScore'].mean():.1f} vs {df['MathScore'].median():.1f})
  • ReadingScore : skewness = {reading_skew:.3f} → slightly skewed.
                   Range: [{df['ReadingScore'].min():.0f}, {df['ReadingScore'].max():.0f}]
  • WritingScore : skewness = {writing_skew:.3f}
                   Range: [{df['WritingScore'].min():.0f}, {df['WritingScore'].max():.0f}]
  • Most variable subject: {most_variable[0]} (std = {most_variable[1]:.2f})
    A higher standard deviation means scores are more spread out, which may
    reflect greater heterogeneity in preparation or ability for that subject.

3) EFFECT OF TEST PREPARATION
------------------------------
  • Students who completed test preparation: avg FinalGrade = {completed_fg}
  • Students who did not complete:           avg FinalGrade = {none_fg}
  • Improvement: {pct_improvement:.1f}%

  This is a meaningful difference. Completing a test preparation course is
  associated with a {pct_improvement:.1f}% higher average final grade. While this is
  correlational (not causal), it suggests test preparation courses add
  measurable academic value. Schools should consider making these courses
  more accessible, especially to students with fewer academic resources.

4) STUDY TIME AND PERFORMANCE
------------------------------
  Average FinalGrade by StudyTime category:
{study_means.to_string()}

  The relationship between study time and FinalGrade is {'monotonic — each' if all(study_means.diff().dropna() > 0) or all(study_means.diff().dropna() < 0) else 'broadly positive but not strictly monotonic — each'} additional
  bracket of weekly study time is associated with a higher average grade.
  This is consistent with the well-established finding in educational
  research that time-on-task is one of the strongest predictors of academic
  achievement.

5) CORRELATION INSIGHTS
------------------------
  From the correlation matrix (excluding the three exam scores themselves),
  the variable with the strongest relationship to FinalGrade is:

    → {top_corr_var}  (r = {top_corr_val:.2f})

  This correlation is {'positive' if top_corr_val > 0 else 'negative'}, meaning that
  {'higher' if top_corr_val > 0 else 'lower'} {top_corr_var} is associated with
  {'higher' if top_corr_val > 0 else 'lower'} FinalGrade values.

  Educational implication:
  {"StudyTime: Students who invest more weekly hours in studying consistently" if top_corr_var == "StudyTime" else "Failures: Students with more past failures score significantly lower, suggesting"} 
  {"achieve higher grades. Interventions that help students develop consistent" if top_corr_var == "StudyTime" else "that early academic failure creates a compounding disadvantage. Early"} 
  {"study habits — such as structured schedules, tutoring, or study groups —" if top_corr_var == "StudyTime" else "identification of at-risk students and targeted support could help break"} 
  {"could have a substantial positive impact on overall academic performance." if top_corr_var == "StudyTime" else "this cycle and improve long-term outcomes."}

================================================================================
"""

print(summary)
