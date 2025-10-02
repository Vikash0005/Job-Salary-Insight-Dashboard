# ------------------ Import Libraries ------------------ #
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ------------------ Load Dataset ------------------ #
df = pd.read_csv("data/ds_salaries.csv")

# ------------------ Streamlit Setup ------------------ #
st.set_page_config(page_title="Job Salary Insights", layout="wide")
st.title("Job Salary Insights")

# ------------------ Sidebar Filters ------------------ #
st.sidebar.header("Filters")

# Salary Range Slider
min_salary = int(df['salary_in_usd'].min())
max_salary = int(df['salary_in_usd'].max())
salary_range = st.sidebar.slider("Salary Range (USD)", min_salary, max_salary, (min_salary, max_salary))

# Job Title Filter
job_titles = st.sidebar.multiselect("Job Title", options=sorted(df['job_title'].unique()), default=sorted(df['job_title'].unique()))

# Company Filter
companies = st.sidebar.multiselect("Company", options=sorted(df['company'].unique()), default=sorted(df['company'].unique()) if 'company' in df.columns else [])

# Location Filter
locations = st.sidebar.multiselect("Location", options=sorted(df['company_location'].unique()), default=sorted(df['company_location'].unique()))

# ------------------ Apply Filters ------------------ #
filtered_df = df[
    (df['salary_in_usd'] >= salary_range[0]) &
    (df['salary_in_usd'] <= salary_range[1]) &
    (df['job_title'].isin(job_titles)) &
    (df['company_location'].isin(locations))
]

if 'company' in df.columns and companies:
    filtered_df = filtered_df[filtered_df['company'].isin(companies)]

# ------------------ Summary Statistics ------------------ #
st.subheader("Summary Statistics")
if not filtered_df.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Jobs", len(filtered_df))
    col2.metric("Average Salary (USD)", f"${filtered_df['salary_in_usd'].mean():,.0f}")
    col3.metric("Highest Salary", f"${filtered_df['salary_in_usd'].max():,}")

    # ------------------ Visualizations ------------------ #
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Salary by Job Title")
        job_salary = filtered_df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False)
        fig1, ax1 = plt.subplots()
        sns.barplot(x=job_salary.values, y=job_salary.index, ax=ax1, color='skyblue')
        ax1.set_xlabel("Salary")
        st.pyplot(fig1)

    with col2:
        st.subheader("Salary Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(filtered_df['salary_in_usd'], bins=20, kde=False, ax=ax2, color='dodgerblue')
        ax2.set_xlabel("Salary")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Salary by Location")
        loc_salary = filtered_df.groupby('company_location')['salary_in_usd'].mean()
        fig3, ax3 = plt.subplots()
        ax3.pie(loc_salary.values, labels=loc_salary.index, autopct='%1.1f%%', startangle=140)
        ax3.axis('equal')
        st.pyplot(fig3)

    with col4:
        st.subheader("Salary by Company")
        if 'company' in df.columns:
            comp_salary = filtered_df.groupby('company')['salary_in_usd'].mean().sort_values(ascending=False)
            fig4, ax4 = plt.subplots()
            sns.barplot(x=comp_salary.values, y=comp_salary.index, ax=ax4, color='steelblue')
            ax4.set_xlabel("Salary")
            st.pyplot(fig4)
        else:
            st.info("Company data not available in the dataset.")

else:
    st.warning("No data found for selected filters.")

# ------------------ Machine Learning Model ------------------ #
# Encode categorical variables
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

# ------------------ Salary Prediction Form ------------------ #
st.subheader("Predict Job Salary")
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
