from src.fuzzy_engine import predict_risk


# Test Student 1 - Expected to have relatively low risk
score, category = predict_risk(
    attendance_value=90,
    internal_marks_value=85,
    assignment_value=88,
    study_hours_value=6
)

print("Student 1")
print("Risk Score:", score)
print("Risk Level:", category)


# Test Student 2 - Expected to have relatively high risk
score, category = predict_risk(
    attendance_value=45,
    internal_marks_value=35,
    assignment_value=40,
    study_hours_value=1
)

print("\nStudent 2")
print("Risk Score:", score)
print("Risk Level:", category)


# Test Student 3 - Expected to have medium risk
score, category = predict_risk(
    attendance_value=70,
    internal_marks_value=60,
    assignment_value=65,
    study_hours_value=4
)

print("\nStudent 3")
print("Risk Score:", score)
print("Risk Level:", category)