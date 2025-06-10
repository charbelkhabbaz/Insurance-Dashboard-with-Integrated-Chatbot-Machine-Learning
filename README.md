# Insurance-Dashboard-with-Integrated-Chatbot-Machine-Learning
**Short Description**

An interactive Streamlit dashboard for exploring and analyzing insurance policy data,
featuring a natural language chatbot (SQL-based) and integrated machine learning models
for investment prediction, risk analysis, customer segmentation, anomaly detection,
and renewal prediction.

**Key Features**
- 📈 Interactive dashboard with filters and KPIs
- 💬 Natural language chatbot for querying database
- 🤖 Machine Learning models:
  -Investment Prediction (Regression)
  -Risk Classification
  -Customer Segmentation (KMeans)
  -Anomaly Detection (Isolation Forest)
  -Renewal Prediction (Logistic Regression)
- 🔗 MySQL backend for real data queries
- 📊 Plotly charts and user-driven insights

**Tech Stack**
- Python
- Streamlit
- scikit-learn
- Pandas, NumPy
- Plotly
- LangChain (for chatbot logic)
- MySQL

  **Installation instructions**
# 1. Clone the repo
clone the project
change directory to the project

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

- Create a MySQL database named `insurance`.
- Run the SQL schema and insert data:
 Provided in myDb.txt file
- Update database credentials in `config.toml` or `.env`:

[mysql]
host = "localhost"
user = "root"
password = "yourpassword"
database = "myDb"

---

**How to Run the App**

# From the root directory:
streamlit run Home.py

- The dashboard loads insurance data from a MySQL table (`insurance`).
- Users can explore the data using dropdown filters and see summaries by region, location, etc.
- The "ML Page" allows users to train and test models:
  - Users select ML task from a dropdown (investment prediction, risk analysis, etc.)
  - Models are trained using scikit-learn with visual feedback (feature importance, MSE, etc.)
- The chatbot uses a SQL generator (LangChain or custom parser) to convert natural language to SQL queries and return real results from the DB than convert back the response to a human friendly language.

 🧾 Dashboard Overview
This is the main page of the dashboard, where users can interact with the insurance dataset in real time.
A "View Dataset" section allows users to explore the full table stored in the MySQL database.
Users can filter the data dynamically by:
Region/Location/Construction type/Specific columns

🟡 Important: All visualizations and charts in the dashboard are automatically updated based on the filters applied — not just the table view.
⚡ Real-Time Data:
The dashboard fetches data live from the MySQL database using SQL queries. If any data in the database is updated, the dashboard reflects the changes.
![Capture1](https://github.com/user-attachments/assets/ef6883de-4b8f-4b96-b796-05c498aa6b9b)

![Capture2](https://github.com/user-attachments/assets/4def630d-bd12-4fbd-8628-8180f35a5db0)

![Capture3](https://github.com/user-attachments/assets/1b2009ac-1fee-44ee-a8a6-6bc7832bee6a)

The Progress page displays the target investment and shows how close the business’s total investment is to the specified target.
![Capture4](https://github.com/user-attachments/assets/fca3b576-c97e-40ea-bbd8-6ab65bcad078)

🤖 Machine Learning Page Overview
The Machine Learning page provides access to multiple predictive models that analyze the insurance data from different perspectives, including:

📈 Investment Prediction – Predicts investment amounts using either Linear Regression or Random Forest Regressor, selectable by the user
🎯 Risk Classification – Classifies policies based on calculated risk scores
👥 Customer Segmentation – Groups similar policyholders using KMeans clustering
🔍 Anomaly Detection – Detects unusual or outlier policies using Isolation Forest
🔄 Renewal Prediction – Estimates the likelihood of policy renewal using Logistic Regression
here is an example of the Anomaly Detection:
![Capture5](https://github.com/user-attachments/assets/bed3c8bc-e262-4c2e-acfb-6de8fd16d693)
![Capture6](https://github.com/user-attachments/assets/23ae48de-5914-476b-acdf-f5cb34d79e2c)

the chatbot
![Capture8](https://github.com/user-attachments/assets/8ba65574-f48f-4bb6-87a8-b25e924c674a)
![Capture9](https://github.com/user-attachments/assets/b2a46759-ce9e-4a62-8a10-754ba5c29f59)

![Capture10](https://github.com/user-attachments/assets/9e7cc470-2ffd-4d03-afca-c5e5af3670ac)





















 
