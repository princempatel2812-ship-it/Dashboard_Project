import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set layout configurations and browser tab settings
st.set_page_config(
    page_title="Zambia Development Analysis", 
    page_icon="🇿🇲", 
    layout="wide"
)
sns.set_theme(style="whitegrid", palette="muted")

@st.cache_data
def load_data():
    try:
        with open("extracted_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

app_data = load_data()

st.title("🇿🇲 Zambia Human Development Report 2016")
st.markdown("Exploratory data analysis dashboard visualizing key development trends, national strengths, and systemic challenges.")

if not app_data:
    st.error("Data source missing. Please ensure 'extracted_data.json' exists in your root folder.")
else:
    data = app_data["data"]
    evaluation = app_data["evaluation"]
    
    # --- "Country Profile: Zambia" HEADING HAS BEEN REMOVED FROM HERE ---
    
    # Core Indicators Section (Top KPI Metrics)
    st.subheader("Core Development Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("HDI Value", data.get('HDI_value', 'N/A'))
    col2.metric("HDI Rank", data.get('HDI_rank', 'N/A'))
    col3.metric("Life Expectancy", f"{data.get('life_expectancy_years', 'N/A')} yrs")
    col4.metric("Schooling (Expected)", f"{data.get('expected_years_of_schooling', 'N/A')} yrs")
    col5.metric("GNI per Capita", f"${data.get('GNI_per_capita', 'N/A')}")
    
    st.markdown("---")
    
    # --- ROW 1: THEMATIC DISTRIBUTION & MODEL COMPARISON ---
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        # PLOT 1: Thematic Focus Distribution
        st.subheader("1. Thematic Extraction Distribution")
        themes = data.get('themes', {})
        if themes:
            theme_df = pd.DataFrame(list(themes.items()), columns=['Theme', 'Mentions']).sort_values(by='Mentions', ascending=False)
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.barplot(data=theme_df, x='Mentions', y='Theme', ax=ax1, palette="viridis")
            ax1.set_xlabel("Frequency Focus Score")
            ax1.set_ylabel("")
            st.pyplot(fig1)
            
    with row1_col2:
        # PLOT 2: Cross-LLM Accuracy Comparison
        st.subheader("2. Model Comparison of Numerical Indicators")
        comp_data = data.get('model_comparison', {})
        if comp_data:
            comp_df = pd.DataFrame(comp_data)
            melted_df = comp_df.melt(id_vars="Indicators", var_name="Model/Source", value_name="Value")
            
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.barplot(data=melted_df, x="Indicators", y="Value", hue="Model/Source", ax=ax2, palette="Set2")
            ax2.set_yscale("log")
            ax2.set_ylabel("Values (Log Scale)")
            ax2.set_xlabel("")
            plt.xticks(rotation=15)
            st.pyplot(fig2)

    st.markdown("---")
    
    # --- ROW 2: HISTORICAL TRENDS & DEMOGRAPHIC CHANGES ---
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        # PLOT 3: Historical HDI Time-Series
        st.subheader("3. Historical HDI Trend")
        hdi_trend = data.get('hdi_trend', {})
        if hdi_trend:
            hdi_df = pd.DataFrame(list(hdi_trend.items()), columns=['Year', 'HDI'])
            hdi_df['Year'] = pd.to_numeric(hdi_df['Year'])
            hdi_df = hdi_df.sort_values('Year')
            
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.lineplot(data=hdi_df, x='Year', y='HDI', marker='o', linewidth=2.5, color='royalblue', ax=ax3)
            ax3.set_ylim(0, 1)
            ax3.set_ylabel("Human Development Index Score")
            st.pyplot(fig3)
            
    with row2_col2:
        # PLOT 4: Demographic / Population Dynamic over Time
        st.subheader("4. Demographic Trend (Population Growth)")
        pop_trend = data.get('population_trend', {})
        if pop_trend:
            pop_df = pd.DataFrame(list(pop_trend.items()), columns=['Year', 'Population (Millions)'])
            pop_df['Year'] = pd.to_numeric(pop_df['Year'])
            pop_df = pop_df.sort_values('Year')
            
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            sns.lineplot(data=pop_df, x='Year', y='Population (Millions)', marker='s', linewidth=2.5, color='forestgreen', ax=ax4)
            ax4.fill_between(pop_df['Year'], pop_df['Population (Millions)'], color='forestgreen', alpha=0.1)
            ax4.set_ylabel("Total Population (Millions)")
            st.pyplot(fig4)

    st.markdown("---")

    # Qualitative Framework Outputs
    st.subheader("Qualitative Narrative Insights")
    qual_col1, qual_col2 = st.columns(2)
    
    with qual_col1:
        st.info("**Key Strategic Strengths:**")
        for strength in data.get('key_strengths', []):
            st.markdown(f"- {strength}")
            
    with qual_col2:
        st.warning("**Key Strategic Challenges:**")
        for challenge in data.get('key_challenges', []):
            st.markdown(f"- {challenge}")

    st.markdown("---")
    
    # Cross-LLM Behavior Analysis Documentation
    st.subheader("Evaluation & Cross-LLM Behavior Analysis")
    st.write(evaluation)
