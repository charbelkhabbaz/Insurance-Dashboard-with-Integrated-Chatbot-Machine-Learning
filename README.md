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
  - Investment Prediction (Regression)
  - Risk Classification
  - Customer Segmentation (KMeans)
  - Anomaly Detection (Isolation Forest)
  - Renewal Prediction (Logistic Regression)
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

#### ✅ **How to Run the App**
```bash
# From the root directory:
streamlit run Home.py

- The dashboard loads insurance data from a MySQL table (`insurance`).
- Users can explore the data using dropdown filters and see summaries by region, location, etc.
- The "ML Page" allows users to train and test models:
  - Users select ML task from a dropdown (investment prediction, risk analysis, etc.)
  - Models are trained using scikit-learn with visual feedback (feature importance, MSE, etc.)
- The chatbot uses a SQL generator (LangChain or custom parser) to convert natural language to SQL queries and return real results from the DB than convert back the response to a human friendly language.
![Capture1](https://github.com/user-attachments/assets/ef6883de-4b8f-4b96-b796-05c498aa6b9b)














 
