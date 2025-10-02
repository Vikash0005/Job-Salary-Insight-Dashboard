import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ----------------- Load Dataset ----------------- #
df = pd.read_csv("data/ds_salaries.csv")

# ----------------- Streamlit Page Setup ----------------- #
st.set_page_config(page_title="Job Salary Insight Dashboard", layout="wide")
st.title("💼 Job Salary Insight Dashboard")

# ----------------- Session State for Filters ----------------- #
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = df.copy()

# ----------------- Dataset Preview ----------------- #
st.subheader("Dataset Preview")
st.dataframe(df.head(10))

# ----------------- Sidebar Filters ----------------- #
st.sidebar.header("🔍 Search Filters")

# Filter by Job Title
job_input = st.sidebar.multiselect(
    "Search by Job Title",
    options=sorted(df['job_title'].unique())
)

# Filter by Company Location
location_input = st.sidebar.multiselect(
    "Search by Company Location",
    options=sorted(df['company_location'].unique())
)

# Apply filters
filtered_df = df.copy()
if job_input:
    filtered_df = filtered_df[filtered_df['job_title'].isin(job_input)]
if location_input:
    filtered_df = filtered_df[filtered_df['company_location'].isin(location_input)]

st.session_state.filtered_df = filtered_df

# ----------------- Summary Statistics ----------------- #
st.subheader(" Summary Statistics")
if not filtered_df.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Jobs", len(filtered_df))
    col2.metric("Avg Salary (USD)", round(filtered_df['salary_in_usd'].mean(), 2))
    col3.metric("Highest Salary", filtered_df['salary_in_usd'].max())

    # Salary Distribution
    st.subheader("Salary Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(filtered_df['salary_in_usd'], kde=True, ax=ax1, color='green')
    st.pyplot(fig1)

    # Top 10 Job Titles vs Salary #
    st.subheader("Top 10 Job Titles vs Avg Salary")
    top_jobs = (filtered_df.groupby('job_title')['salary_in_usd']
                .mean()
                .sort_values(ascending=False)
                .head(10))
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(x=top_jobs.index, y=top_jobs.values, ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig2)
else:
    st.warning("No data found for selected filters.")

# ----------------- ML Model Training ----------------- #
le_job = LabelEncoder()
le_type = LabelEncoder()
le_exp = LabelEncoder()
le_loc = LabelEncoder()

df['job_title_enc'] = le_job.fit_transform(df['job_title'])
df['employment_type_enc'] = le_type.fit_transform(df['employment_type'])
df['experience_level_enc'] = le_exp.fit_transform(df['experience_level'])
df['company_location_enc'] = le_loc.fit_transform(df['company_location'])

X = df[['job_title_enc', 'employment_type_enc', 'experience_level_enc', 'company_location_enc']]
y = df['salary_in_usd']
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# ----------------- Salary Prediction ----------------- #
st.subheader("💰 Predict Job Salary")
with st.form("prediction_form"):
    selected_job = st.selectbox("Select Job Title", df['job_title'].unique())
    selected_emp_type = st.selectbox("Select Employment Type", df['employment_type'].unique())
    selected_exp = st.selectbox("Select Experience Level", df['experience_level'].unique())
    selected_loc = st.selectbox("Select Company Location", df['company_location'].unique())
    
    submitted = st.form_submit_button("Predict Salary")
    
    if submitted:
        input_data = pd.DataFrame({
            'job_title_enc': [le_job.transform([selected_job])[0]],
            'employment_type_enc': [le_type.transform([selected_emp_type])[0]],
            'experience_level_enc': [le_exp.transform([selected_exp])[0]],
            'company_location_enc': [le_loc.transform([selected_loc])[0]]
        })
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Salary: ${int(prediction):,}")
