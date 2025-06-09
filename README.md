# Insurance-Dashboard-with-Integrated-Chatbot-Machine-Learning
** Short Description**
An interactive Streamlit dashboard for exploring and analyzing insurance policy data,
featuring a natural language chatbot (SQL-based) and integrated machine learning models
for investment prediction, risk analysis, customer segmentation, anomaly detection,
and renewal prediction.

** Key Features**
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

** Tech Stack**
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

```toml
[mysql]
host = "localhost"
user = "root"
password = "yourpassword"
database = "myDb"

---

#### ✅ **How to Run the App**
```bash
# From the root directory:
streamlit run app/app.py

- The dashboard loads insurance data from a MySQL table (`insurance`).
- Users can explore the data using dropdown filters and see summaries by region, location, etc.
- The "ML Page" allows users to train and test models:
  - Users select ML task from a dropdown (investment prediction, risk analysis, etc.)
  - Models are trained using scikit-learn with visual feedback (feature importance, MSE, etc.)
- The chatbot uses a SQL generator (LangChain or custom parser) to convert natural language to SQL queries and return real results from the DB than convert back the response to a human friendly language.
![image](https://github.com/user-attachments/assets/f1690467-3219-4431-8009-4783e5052d95)
![image](https://github.com/user-attachments/assets/6292d036-78b0-47e4-8a57-a9191a5b11cc)
![image](https://github.com/user-attachments/assets/0e2ea162-439c-4346-a399-1c2602ec1b04)
![image](https://github.com/user-attachments/assets/2040f52e-d8aa-42ec-8ecf-c8356c3e4da8)
![image](https://github.com/user-attachments/assets/1ca0286e-573a-40fb-b396-e1737919fe5e)
![image](https://github.com/user-attachments/assets/ff590693-746a-4bb5-9494-d5c6f7537fd7)
![image](https://github.com/user-attachments/assets/4c60a606-f3e4-4b94-9a33-13908cdf05a5)
![image](https://github.com/user-attachments/assets/4a19d021-7d2b-4933-80f3-003faf511e51)












 
