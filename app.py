import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set page configuration and styling
st.set_page_config(page_title="UN HDR Intelligence Dashboard", layout="wide")
sns.set_theme(style="whitegrid", palette="muted")

@st.cache_data
def load_data():
    try:
        with open("extracted_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

app_data = load_data()

st.title("🌍 UN Development Intelligence Dashboard")
st.markdown("Automated insights and visual analytics derived from local LLM extraction pipelines.")

if not app_data:
    st.error("Data source missing. Please ensure 'extracted_data.json' exists or run the pipeline.")
else:
    data = app_data["data"]
    evaluation = app_data["evaluation"]
    
    st.header(f"Country Profile: {data.get('country', 'Unknown')}")
    
    # --- TASK 3: Interactive Dashboard ---
    
    # 1. Core Indicators (Top Metrics)
    st.subheader("Core Development Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("HDI Value", data.get('HDI_value', 'N/A'))
    col2.metric("HDI Rank", data.get('HDI_rank', 'N/A'))
    col3.metric("Life Expectancy", f"{data.get('life_expectancy_years', 'N/A')}y")
    col4.metric("Schooling (Expected)", f"{data.get('expected_years_of_schooling', 'N/A')}y")
    col5.metric("GNI per Capita", f"${data.get('GNI_per_capita', 'N/A')}")
    
    st.markdown("---")
    
    # --- Data Visualizations ---
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Plot 1: Thematic Distribution (Seaborn Barplot)
        st.subheader("Thematic Extraction Distribution")
        themes = data.get('themes', {})
        if themes:
            theme_df = pd.DataFrame(list(themes.items()), columns=['Theme', 'Frequency'])
            theme_df = theme_df.sort_values(by='Frequency', ascending=False)
            
            fig_themes, ax_themes = plt.subplots(figsize=(6, 4))
            sns.barplot(data=theme_df, x='Frequency', y='Theme', ax=ax_themes, palette="viridis")
            ax_themes.set_title("Frequency of Themes in Report Narrative", pad=15)
            ax_themes.set_xlabel("Mentions / Focus Score")
            ax_themes.set_ylabel("")
            st.pyplot(fig_themes)

    with chart_col2:
        # Plot 2: Demographic / HDI Trend (Seaborn Lineplot)
        st.subheader("Historical HDI Trend")
        trend_data = data.get('hdi_trend', {})
        if trend_data:
            trend_df = pd.DataFrame(list(trend_data.items()), columns=['Year', 'HDI'])
            trend_df['Year'] = pd.to_numeric(trend_df['Year']) # Ensure proper numerical sorting
            trend_df = trend_df.sort_values('Year')
            
            fig_trend, ax_trend = plt.subplots(figsize=(6, 4))
            sns.lineplot(data=trend_df, x='Year', y='HDI', marker='o', linewidth=2, color='coral', ax=ax_trend)
            ax_trend.set_title("Human Development Index Over Time", pad=15)
            ax_trend.set_ylim(0, 1) # HDI is always between 0 and 1
            st.pyplot(fig_trend)

    st.markdown("---")

    # --- Qualitative Extraction ---
    st.subheader("Qualitative Extraction Insights")
    qual_col1, qual_col2 = st.columns(2)
    
    with qual_col1:
        st.info("**Key Strengths Identified:**")
        for strength in data.get('key_strengths', []):
            st.markdown(f"- {strength}")
            
    with qual_col2:
        st.warning("**Key Challenges Identified:**")
        for challenge in data.get('key_challenges', []):
            st.markdown(f"- {challenge}")

    st.markdown("---")
    
    # --- Extension Task: Model Comparison ---
    st.subheader("Cross-LLM Evaluation (LLM-as-a-Judge)")
    with st.expander("View Evaluator Output", expanded=True):
        st.write(evaluation)