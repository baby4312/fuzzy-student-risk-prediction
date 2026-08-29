import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# ============================================================
# FUZZY VARIABLES
# ============================================================

# Input variables
attendance = ctrl.Antecedent(
    np.arange(0, 101, 1),
    "attendance"
)

internal_marks = ctrl.Antecedent(
    np.arange(0, 101, 1),
    "internal_marks"
)

assignment_performance = ctrl.Antecedent(
    np.arange(0, 101, 1),
    "assignment_performance"
)

study_hours = ctrl.Antecedent(
    np.arange(0, 8.1, 0.1),
    "study_hours"
)

# Output variable
risk_score = ctrl.Consequent(
    np.arange(0, 101, 1),
    "risk_score"
)


# ============================================================
# MEMBERSHIP FUNCTIONS
# ============================================================

# Attendance
attendance["low"] = fuzz.trapmf(
    attendance.universe,
    [0, 0, 40, 60]
)

attendance["medium"] = fuzz.trimf(
    attendance.universe,
    [40, 60, 80]
)

attendance["high"] = fuzz.trapmf(
    attendance.universe,
    [60, 80, 100, 100]
)


# Internal Marks
internal_marks["poor"] = fuzz.trapmf(
    internal_marks.universe,
    [0, 0, 30, 50]
)

internal_marks["average"] = fuzz.trimf(
    internal_marks.universe,
    [30, 55, 75]
)

internal_marks["good"] = fuzz.trapmf(
    internal_marks.universe,
    [60, 80, 100, 100]
)


# Assignment Performance
assignment_performance["poor"] = fuzz.trapmf(
    assignment_performance.universe,
    [0, 0, 30, 50]
)

assignment_performance["average"] = fuzz.trimf(
    assignment_performance.universe,
    [30, 55, 75]
)

assignment_performance["good"] = fuzz.trapmf(
    assignment_performance.universe,
    [60, 80, 100, 100]
)


# Study Hours
study_hours["low"] = fuzz.trapmf(
    study_hours.universe,
    [0, 0, 1.5, 3]
)

study_hours["medium"] = fuzz.trimf(
    study_hours.universe,
    [2, 4, 5.5]
)

study_hours["high"] = fuzz.trapmf(
    study_hours.universe,
    [4.5, 6, 8, 8]
)


# Risk Score
risk_score["low"] = fuzz.trapmf(
    risk_score.universe,
    [0, 0, 20, 40]
)

risk_score["medium"] = fuzz.trimf(
    risk_score.universe,
    [30, 50, 70]
)

risk_score["high"] = fuzz.trapmf(
    risk_score.universe,
    [60, 80, 100, 100]
)


# ============================================================
# FUZZY RULES
# ============================================================

# HIGH RISK RULES

rule1 = ctrl.Rule(
    attendance["low"] & internal_marks["poor"],
    risk_score["high"]
)

rule2 = ctrl.Rule(
    attendance["low"] & assignment_performance["poor"],
    risk_score["high"]
)

rule3 = ctrl.Rule(
    attendance["low"] & study_hours["low"],
    risk_score["high"]
)

rule4 = ctrl.Rule(
    internal_marks["poor"] & assignment_performance["poor"],
    risk_score["high"]
)

rule5 = ctrl.Rule(
    internal_marks["poor"] & study_hours["low"],
    risk_score["high"]
)

rule6 = ctrl.Rule(
    attendance["medium"] & internal_marks["poor"],
    risk_score["high"]
)


# MEDIUM RISK RULES

rule7 = ctrl.Rule(
    attendance["low"] & internal_marks["average"],
    risk_score["medium"]
)

rule8 = ctrl.Rule(
    attendance["low"] & internal_marks["good"],
    risk_score["medium"]
)

rule9 = ctrl.Rule(
    attendance["medium"] & internal_marks["average"],
    risk_score["medium"]
)

rule10 = ctrl.Rule(
    attendance["medium"] & internal_marks["good"],
    risk_score["medium"]
)

rule11 = ctrl.Rule(
    attendance["high"] & internal_marks["average"],
    risk_score["medium"]
)

rule12 = ctrl.Rule(
    assignment_performance["average"] & study_hours["medium"],
    risk_score["medium"]
)

rule13 = ctrl.Rule(
    attendance["high"]
    & internal_marks["good"]
    & assignment_performance["average"],
    risk_score["medium"]
)


# LOW RISK RULES

rule14 = ctrl.Rule(
    attendance["high"] & internal_marks["good"],
    risk_score["low"]
)

rule15 = ctrl.Rule(
    attendance["high"] & assignment_performance["good"],
    risk_score["low"]
)

rule16 = ctrl.Rule(
    internal_marks["good"]
    & assignment_performance["good"]
    & study_hours["high"],
    risk_score["low"]
)

rule17 = ctrl.Rule(
    attendance["high"]
    & internal_marks["good"]
    & assignment_performance["good"],
    risk_score["low"]
)


# ============================================================
# CONTROL SYSTEM
# ============================================================

risk_control_system = ctrl.ControlSystem([
    rule1,
    rule2,
    rule3,
    rule4,
    rule5,
    rule6,
    rule7,
    rule8,
    rule9,
    rule10,
    rule11,
    rule12,
    rule13,
    rule14,
    rule15,
    rule16,
    rule17
])


# ============================================================
# RISK PREDICTION FUNCTION
# ============================================================

def predict_risk(
    attendance_value,
    internal_marks_value,
    assignment_value,
    study_hours_value
):
    """
    Predict student academic risk using fuzzy logic.
    """

    # Create fuzzy simulation
    simulation = ctrl.ControlSystemSimulation(
        risk_control_system
    )

    # Provide input values
    simulation.input["attendance"] = attendance_value
    simulation.input["internal_marks"] = internal_marks_value
    simulation.input["assignment_performance"] = assignment_value
    simulation.input["study_hours"] = study_hours_value

    # Perform fuzzy inference
    simulation.compute()

    # Get crisp risk score after defuzzification
    score = simulation.output["risk_score"]

    # Calculate membership degrees
    low_membership = fuzz.interp_membership(
        risk_score.universe,
        risk_score["low"].mf,
        score
    )

    medium_membership = fuzz.interp_membership(
        risk_score.universe,
        risk_score["medium"].mf,
        score
    )

    high_membership = fuzz.interp_membership(
        risk_score.universe,
        risk_score["high"].mf,
        score
    )

    # Store membership values
    memberships = {
        "LOW": low_membership,
        "MEDIUM": medium_membership,
        "HIGH": high_membership
    }

    # Select category with highest membership
    category = max(
        memberships,
        key=memberships.get
    )

    return round(score, 2), category