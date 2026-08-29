import pandas as pd
from fuzzy_engine import predict_risk


df = pd.read_csv("data/student_data.csv")

risk_scores = []
risk_levels = []

for index, student in df.iterrows():

    try:
        score, level = predict_risk(
            attendance_value=student["attendance"],
            internal_marks_value=student["internal_marks"],
            assignment_value=student["assignment_performance"],
            study_hours_value=student["study_hours"]
        )

        risk_scores.append(score)
        risk_levels.append(level)

    except Exception as e:
        print("\nERROR FOUND")
        print("Student:", student["student_id"])
        print("Attendance:", student["attendance"])
        print("Internal Marks:", student["internal_marks"])
        print("Assignment:", student["assignment_performance"])
        print("Study Hours:", student["study_hours"])
        print("Error:", e)

        raise


df["risk_score"] = risk_scores
df["risk_level"] = risk_levels

df.to_csv(
    "results/student_risk_results.csv",
    index=False
)

print("\nStudent risk prediction completed!")
print(f"Total students processed: {len(df)}")

print("\nRisk Level Distribution:")
print(df["risk_level"].value_counts())

print("\nFirst 10 Results:")
print(
    df[
        [
            "student_id",
            "attendance",
            "internal_marks",
            "assignment_performance",
            "study_hours",
            "risk_score",
            "risk_level"
        ]
    ].head(10)
)