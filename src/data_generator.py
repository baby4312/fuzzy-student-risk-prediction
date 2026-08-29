import numpy as np
import pandas as pd

# Make results reproducible
np.random.seed(42)

NUM_STUDENTS = 200

# ------------------------------------------------------------
# STUDENT PERFORMANCE PROFILES
# ------------------------------------------------------------
# 40% Good-performing students
# 40% Average-performing students
# 20% At-risk students

good_count = 80
average_count = 80
risk_count = 40


# ------------------------------------------------------------
# GOOD-PERFORMING STUDENTS
# ------------------------------------------------------------

good_attendance = np.random.normal(85, 7, good_count)
good_marks = np.random.normal(82, 8, good_count)
good_assignments = np.random.normal(84, 8, good_count)
good_study_hours = np.random.normal(6, 0.8, good_count)


# ------------------------------------------------------------
# AVERAGE-PERFORMING STUDENTS
# ------------------------------------------------------------

average_attendance = np.random.normal(68, 8, average_count)
average_marks = np.random.normal(62, 10, average_count)
average_assignments = np.random.normal(65, 10, average_count)
average_study_hours = np.random.normal(4, 0.8, average_count)


# ------------------------------------------------------------
# AT-RISK STUDENTS
# ------------------------------------------------------------

risk_attendance = np.random.normal(48, 8, risk_count)
risk_marks = np.random.normal(42, 10, risk_count)
risk_assignments = np.random.normal(45, 10, risk_count)
risk_study_hours = np.random.normal(2.2, 0.6, risk_count)


# ------------------------------------------------------------
# COMBINE ALL STUDENTS
# ------------------------------------------------------------

attendance = np.concatenate([
    good_attendance,
    average_attendance,
    risk_attendance
])

internal_marks = np.concatenate([
    good_marks,
    average_marks,
    risk_marks
])

assignment_performance = np.concatenate([
    good_assignments,
    average_assignments,
    risk_assignments
])

study_hours = np.concatenate([
    good_study_hours,
    average_study_hours,
    risk_study_hours
])


# ------------------------------------------------------------
# KEEP VALUES WITHIN VALID RANGES
# ------------------------------------------------------------

attendance = np.clip(attendance, 30, 100)
internal_marks = np.clip(internal_marks, 20, 100)
assignment_performance = np.clip(
    assignment_performance,
    20,
    100
)
study_hours = np.clip(study_hours, 0, 8)


# ------------------------------------------------------------
# SHUFFLE STUDENTS
# ------------------------------------------------------------

indices = np.random.permutation(NUM_STUDENTS)

attendance = attendance[indices]
internal_marks = internal_marks[indices]
assignment_performance = assignment_performance[indices]
study_hours = study_hours[indices]


# ------------------------------------------------------------
# CREATE STUDENT IDs
# ------------------------------------------------------------

student_ids = [
    f"STU{str(i).zfill(3)}"
    for i in range(1, NUM_STUDENTS + 1)
]


# ------------------------------------------------------------
# CREATE DATAFRAME
# ------------------------------------------------------------

df = pd.DataFrame({
    "student_id": student_ids,
    "attendance": np.round(attendance, 1),
    "internal_marks": np.round(internal_marks, 1),
    "assignment_performance": np.round(
        assignment_performance,
        1
    ),
    "study_hours": np.round(study_hours, 1)
})


# ------------------------------------------------------------
# SAVE DATASET
# ------------------------------------------------------------

df.to_csv(
    "data/student_data.csv",
    index=False
)


# ------------------------------------------------------------
# DISPLAY INFORMATION
# ------------------------------------------------------------

print("Dataset created successfully!")
print(f"Number of students: {len(df)}")

print("\nFirst 10 students:")
print(df.head(10))

print("\nDataset Statistics:")
print(df.describe())