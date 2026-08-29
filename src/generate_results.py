import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(
    "results/student_risk_results.csv"
)


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

import os

os.makedirs(
    "results/graphs",
    exist_ok=True
)


# ============================================================
# GRAPH 1: RISK LEVEL DISTRIBUTION
# ============================================================

risk_counts = df["risk_level"].value_counts()

risk_order = ["LOW", "MEDIUM", "HIGH"]

risk_counts = risk_counts.reindex(
    risk_order,
    fill_value=0
)

plt.figure(figsize=(8, 5))

risk_counts.plot(
    kind="bar"
)

plt.title(
    "Student Risk Level Distribution"
)

plt.xlabel(
    "Risk Level"
)

plt.ylabel(
    "Number of Students"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "results/graphs/risk_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 2: ATTENDANCE VS RISK SCORE
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["attendance"],
    df["risk_score"],
    alpha=0.7
)

plt.title(
    "Attendance vs Risk Score"
)

plt.xlabel(
    "Attendance (%)"
)

plt.ylabel(
    "Risk Score"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "results/graphs/attendance_vs_risk.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 3: INTERNAL MARKS VS RISK SCORE
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["internal_marks"],
    df["risk_score"],
    alpha=0.7
)

plt.title(
    "Internal Marks vs Risk Score"
)

plt.xlabel(
    "Internal Marks"
)

plt.ylabel(
    "Risk Score"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "results/graphs/marks_vs_risk.png",
    dpi=300
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("Graphs generated successfully!")

print("\nGenerated files:")

print(
    "1. results/graphs/risk_distribution.png"
)

print(
    "2. results/graphs/attendance_vs_risk.png"
)

print(
    "3. results/graphs/marks_vs_risk.png"
)