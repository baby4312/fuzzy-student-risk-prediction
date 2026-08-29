import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.fuzzy_engine import (
    predict_risk,
    attendance,
    internal_marks,
    assignment_performance,
    study_hours,
    risk_score
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Risk Predictor",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🎓 Fuzzy-Based Student Performance & Risk Prediction")

st.markdown(
    """
    **Soft Computing Techniques Project**

    A Fuzzy Logic based decision-support system that evaluates
    student performance and identifies students who may require
    additional academic attention.
    """
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

try:
    df = pd.read_csv("results/student_risk_results.csv")
except FileNotFoundError:
    df = None


# ============================================================
# NAVIGATION
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "🎯 Risk Prediction",
    "📊 Dataset Analysis",
    "🧠 Fuzzy Logic"
])


# ============================================================
# TAB 1 — RISK PREDICTION
# ============================================================

with tab1:

    st.header("Individual Student Risk Prediction")

    st.write(
        "Enter the student's academic information below. "
        "The fuzzy inference system will calculate a risk "
        "score and risk category."
    )

    col1, col2 = st.columns(2)

    with col1:

        attendance_value = st.slider(
            "Attendance (%)",
            0,
            100,
            75
        )

        internal_marks_value = st.slider(
            "Internal Marks",
            0,
            100,
            70
        )

    with col2:

        assignment_value = st.slider(
            "Assignment Performance (%)",
            0,
            100,
            70
        )

        study_hours_value = st.slider(
            "Study Hours per Day",
            0.0,
            8.0,
            4.0,
            0.1
        )

    st.divider()

    if st.button(
        "🔍 Predict Academic Risk",
        use_container_width=True
    ):

        score, category = predict_risk(
            attendance_value=attendance_value,
            internal_marks_value=internal_marks_value,
            assignment_value=assignment_value,
            study_hours_value=study_hours_value
        )

        st.subheader("Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric(
                "Risk Score",
                f"{score}/100"
            )

        with result_col2:
            st.metric(
                "Risk Level",
                category
            )

        with result_col3:

            if category == "LOW":
                status = "Good"

            elif category == "MEDIUM":
                status = "Needs Attention"

            else:
                status = "Immediate Attention"

            st.metric(
                "Status",
                status
            )

        st.progress(
            min(score / 100, 1.0)
        )

        if category == "LOW":

            st.success(
                "🟢 LOW RISK — The student is currently "
                "performing well."
            )

            st.write(
                "**Recommendation:** Maintain current "
                "attendance, academic performance and "
                "study habits."
            )

        elif category == "MEDIUM":

            st.warning(
                "🟡 MEDIUM RISK — The student may need "
                "additional academic attention."
            )

            st.write(
                "**Recommendation:** Improve consistency "
                "in attendance, assignments and study habits."
            )

        else:

            st.error(
                "🔴 HIGH RISK — The student may require "
                "immediate academic support."
            )

            st.write(
                "**Recommendation:** Focus on improving "
                "attendance, academic performance, assignments "
                "and daily study habits."
            )


# ============================================================
# TAB 2 — DATASET ANALYSIS
# ============================================================

with tab2:

    st.header("📊 Student Dataset Analysis")

    if df is not None:

        # ----------------------------------------------------
        # SUMMARY METRICS
        # ----------------------------------------------------

        total_students = len(df)

        low_count = len(
            df[df["risk_level"] == "LOW"]
        )

        medium_count = len(
            df[df["risk_level"] == "MEDIUM"]
        )

        high_count = len(
            df[df["risk_level"] == "HIGH"]
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Students",
                total_students
            )

        with col2:
            st.metric(
                "🟢 Low Risk",
                low_count
            )

        with col3:
            st.metric(
                "🟡 Medium Risk",
                medium_count
            )

        with col4:
            st.metric(
                "🔴 High Risk",
                high_count
            )

        st.divider()

        # ----------------------------------------------------
        # RISK DISTRIBUTION
        # ----------------------------------------------------

        st.subheader("Risk Level Distribution")

        risk_counts = (
            df["risk_level"]
            .value_counts()
            .reindex(
                ["LOW", "MEDIUM", "HIGH"],
                fill_value=0
            )
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        risk_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Student Risk Level Distribution"
        )

        ax.set_xlabel(
            "Risk Level"
        )

        ax.set_ylabel(
            "Number of Students"
        )

        ax.tick_params(
            axis="x",
            rotation=0
        )

        st.pyplot(fig)

        # ----------------------------------------------------
        # PERFORMANCE VS RISK
        # ----------------------------------------------------

        st.subheader(
            "Performance Factors vs Risk Score"
        )

        col1, col2 = st.columns(2)

        with col1:

            fig1, ax1 = plt.subplots(
                figsize=(7, 5)
            )

            ax1.scatter(
                df["attendance"],
                df["risk_score"],
                alpha=0.7
            )

            ax1.set_title(
                "Attendance vs Risk Score"
            )

            ax1.set_xlabel(
                "Attendance (%)"
            )

            ax1.set_ylabel(
                "Risk Score"
            )

            ax1.grid(
                True,
                alpha=0.3
            )

            st.pyplot(fig1)

        with col2:

            fig2, ax2 = plt.subplots(
                figsize=(7, 5)
            )

            ax2.scatter(
                df["internal_marks"],
                df["risk_score"],
                alpha=0.7
            )

            ax2.set_title(
                "Internal Marks vs Risk Score"
            )

            ax2.set_xlabel(
                "Internal Marks"
            )

            ax2.set_ylabel(
                "Risk Score"
            )

            ax2.grid(
                True,
                alpha=0.3
            )

            st.pyplot(fig2)

        # ----------------------------------------------------
        # STUDENT TABLE
        # ----------------------------------------------------

        st.subheader("Student Risk Results")

        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        csv_data = df.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Results CSV",
            csv_data,
            "student_risk_results.csv",
            "text/csv",
            use_container_width=True
        )

    else:

        st.warning(
            "Results dataset not found. "
            "Run the student processing script first."
        )


# ============================================================
# TAB 3 — FUZZY LOGIC
# ============================================================

with tab3:

    st.header("🧠 Fuzzy Logic Model")

    st.write(
        "The system converts numerical student information "
        "into linguistic fuzzy variables and applies "
        "IF–THEN rules to determine academic risk."
    )

    st.subheader("Input Variables")

    st.markdown(
        """
        **Attendance**
        - Low
        - Medium
        - High

        **Internal Marks**
        - Poor
        - Average
        - Good

        **Assignment Performance**
        - Poor
        - Average
        - Good

        **Study Hours**
        - Low
        - Medium
        - High
        """
    )

    st.subheader("Output Variable")

    st.markdown(
        """
        **Risk Score**
        - Low
        - Medium
        - High
        """
    )

    st.divider()

    st.subheader(
        "Example Fuzzy Rules"
    )

    st.code(
        """
IF Attendance is LOW
AND Internal Marks are POOR
THEN Risk is HIGH

IF Attendance is MEDIUM
AND Internal Marks are AVERAGE
THEN Risk is MEDIUM

IF Attendance is HIGH
AND Internal Marks are GOOD
THEN Risk is LOW

IF Internal Marks are GOOD
AND Assignment Performance is GOOD
AND Study Hours are HIGH
THEN Risk is LOW
        """,
        language="text"
    )

    st.divider()

    st.subheader(
        "Risk Score Membership Functions"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        risk_score.universe,
        risk_score["low"].mf,
        label="Low Risk"
    )

    ax.plot(
        risk_score.universe,
        risk_score["medium"].mf,
        label="Medium Risk"
    )

    ax.plot(
        risk_score.universe,
        risk_score["high"].mf,
        label="High Risk"
    )

    ax.set_xlabel(
        "Risk Score"
    )

    ax.set_ylabel(
        "Membership Degree"
    )

    ax.set_title(
        "Risk Score Fuzzy Membership Functions"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3
    )

    st.pyplot(fig)

    st.divider()

    st.subheader(
        "System Architecture"
    )

    st.code(
        """
Student Data
     │
     ▼
Fuzzification
     │
     ▼
Fuzzy Rule Base
     │
     ▼
Fuzzy Inference
     │
     ▼
Defuzzification
     │
     ▼
Risk Score
     │
     ▼
LOW / MEDIUM / HIGH
        """,
        language="text"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Soft Computing Techniques | "
    "Fuzzy-Based Student Performance & Early Risk Prediction"
)