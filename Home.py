import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from streamlit_option_menu import option_menu
from numerize.numerize import numerize
import time
from streamlit_extras.metric_cards import style_metric_cards
import matplotlib.pyplot as plt
from query import view_all_data
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    r2_score,
    accuracy_score, 
    precision_score,
    recall_score,
    f1_score,
    silhouette_score
)
import joblib
import requests
import re
import mysql.connector
from typing import List, Dict, Any

# OpenRouter API Configuration
OPENROUTER_API_KEY = "put_your_API"
OPENROUTER_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b:free"

# Security measures
BLOCKED_KEYWORDS = {
    'DROP', 'DELETE', 'TRUNCATE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'RENAME',
    'REPLACE', 'RESTORE', 'TRIGGER', 'GRANT', 'REVOKE', 'CALL', 'EXECUTE', 'MERGE'
}

def is_safe_query(query: str) -> bool:
    """Check if the SQL query is safe to execute."""
    upper_query = query.upper()
    return not any(keyword in upper_query for keyword in BLOCKED_KEYWORDS)

def execute_safe_query(query: str) -> List[Dict[str, Any]]:
    """Execute a safe SQL query and return results."""
    if not is_safe_query(query):
        raise ValueError("Unsafe query detected")
    
    # Use the existing connection from view_all_data
    result = view_all_data(custom_query=query)
    return result

def preprocess_query(text: str) -> str:
    """Preprocess the query to identify intent and key terms."""
    text = text.lower().strip()
    
    # Common variations and synonyms
    synonyms = {
        'show': ['display', 'list', 'give', 'tell', 'what are', 'what is'],
        'average': ['mean', 'typical', 'normal', 'avg'],
        'highest': ['top', 'best', 'maximum', 'max', 'most expensive'],
        'lowest': ['bottom', 'worst', 'minimum', 'min', 'least expensive'],
        'total': ['sum', 'overall', 'all', 'entire'],
        'count': ['how many', 'number of', 'quantity', 'amount'],
        'investment': ['money', 'cost', 'price', 'value', 'premium'],
        'rating': ['score', 'performance', 'evaluation', 'grade'],
        'policies': ['insurance', 'contracts', 'coverage', 'plans'],
        'risky': ['dangerous', 'high-risk', 'unsafe', 'hazardous']
    }
    
    # Replace synonyms with standard terms
    for standard, variations in synonyms.items():
        for variant in variations:
            if variant in text:
                text = text.replace(variant, standard)
    
    return text

def extract_intent(text: str) -> dict:
    """Extract the intent and parameters from the query."""
    text = preprocess_query(text)
    
    intents = {
        'greeting': ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'],
        'farewell': ['bye', 'goodbye', 'see you', 'thanks', 'thank you'],
        'help': ['help', 'what can you do', 'how do you work', 'what are your capabilities'],
        'analysis': ['analyze', 'study', 'examine', 'investigate', 'look into'],
        'comparison': ['compare', 'versus', 'vs', 'difference between', 'better'],
        'trend': ['trend', 'over time', 'pattern', 'historical', 'history'],
        'summary': ['summarize', 'overview', 'brief', 'summary'],
        'risk': ['risk', 'dangerous', 'unsafe', 'hazard', 'earthquake', 'flood']
    }
    
    # Detect primary intent
    detected_intent = 'query'  # default intent
    for intent, keywords in intents.items():
        if any(keyword in text for keyword in keywords):
            detected_intent = intent
            break
    
    # Extract parameters
    params = {
        'region': None,
        'business_type': None,
        'construction': None,
        'rating_threshold': None,
        'investment_threshold': None,
        'limit': None
    }
    
    # Number detection
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if numbers:
        if 'rating' in text:
            params['rating_threshold'] = float(numbers[0])
        elif 'investment' in text:
            params['investment_threshold'] = float(numbers[0])
        if 'top' in text or 'first' in text:
            params['limit'] = int(numbers[0]) if numbers else 5

    return {'intent': detected_intent, 'params': params, 'processed_text': text}

def get_chatbot_response(prompt: str, context: list = None) -> str:
    """Get enhanced response from OpenRouter API with better context understanding."""
    messages = []
    
    # Enhanced system message with specific table schema
    system_message = """You are an advanced insurance analytics assistant with expertise in SQL and insurance data analysis.
    
    You have access to ONE table named 'insurance' with the following schema:
    - Policy (int): Policy number
    - Expiry (varchar(9)): Expiry date in 'DD-MMM-YY' format
    - Location (varchar(5)): Urban/Rural
    - State (varchar(13)): State name
    - Region (varchar(9)): Geographic region
    - Investment (int): Investment amount
    - Construction (varchar(11)): Construction type
    - BusinessType (varchar(13)): Type of business
    - Earthquake (varchar(1)): 'Y' or 'N' for earthquake coverage
    - Flood (varchar(1)): 'Y' or 'N' for flood coverage
    - Rating (decimal(3,1)): Policy rating from 0.0 to 10.0
    - id (int): Auto-incrementing primary key

    When generating SQL queries:
    1. ALWAYS use the table name 'insurance'
    2. Be mindful of data types and use appropriate comparisons
    3. For Earthquake/Flood, use 'Y' or 'N' in comparisons
    4. For dates, respect the 'DD-MMM-YY' format
    5. For numeric comparisons on Rating, use decimal values

    Guidelines for your responses:
    - Be conversational and friendly, using emojis occasionally
    - Provide specific, data-driven insights
    - If a query is unclear, ask for clarification
    - Keep responses clear and well-formatted
    """
    messages.append({"role": "system", "content": system_message})
    
    # Add context from previous messages
    if context:
        for msg in context[-3:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Extract intent and enhance the prompt
    intent_data = extract_intent(prompt)
    enhanced_prompt = f"""
    User's original question: {prompt}
    Detected intent: {intent_data['intent']}
    Processed text: {intent_data['processed_text']}
    
    Please provide a response considering this context and intent.
    If generating SQL, ensure it uses the correct table name 'insurance' and appropriate data types.
    """
    
    messages.append({"role": "user", "content": enhanced_prompt})
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error connecting to AI service: {str(e)}"

def format_query_result(result: List[Dict[str, Any]]) -> str:
    """Format query results into a readable string."""
    if not result:
        return "No results found."
    
    df = pd.DataFrame(result)
    return df.to_string()

def is_database_query(text: str) -> bool:
    """Enhanced detection of database queries with better understanding of intent."""
    # Preprocess the query
    processed_text = preprocess_query(text)
    
    # Expanded database keywords for better detection
    database_keywords = {
        # Question starters
        'how many', 'show me', 'what is', 'list', 'count', 'average', 'tell me about',
        'find', 'search for', 'look up', 'display', 'give me', 'can you show',
        
        # Data elements
        'policies', 'investment', 'rating', 'region', 'state', 'location',
        'business type', 'construction', 'earthquake', 'flood', 'expiry',
        
        # Analysis terms
        'highest', 'lowest', 'total', 'percentage', 'average', 'mean',
        'maximum', 'minimum', 'greater than', 'less than', 'between',
        'top', 'bottom', 'best', 'worst', 'most', 'least',
        
        # Business terms
        'risk', 'coverage', 'premium', 'value', 'cost', 'price',
        'performance', 'score', 'grade', 'evaluation'
    }
    
    # Check for database intent
    text_lower = processed_text.lower()
    
    # Direct matches
    if any(keyword in text_lower for keyword in database_keywords):
        return True
    
    # Check for implicit queries
    implicit_patterns = [
        r'\d+(?:\.\d+)?(?:\s*(?:million|k|thousand))?',  # Numbers with units
        r'(?:in|at|for|from)\s+\w+',  # Location/time references
        r'(?:more|less)\s+than',  # Comparisons
        r'(?:high|low|medium)\s+(?:risk|rating|investment)',  # Quality descriptions
        r'recent|latest|oldest|expired?|active',  # Time-based queries
        r'(?:earth)?quake|flood|disaster|hazard',  # Risk-related terms
        r'business|commercial|residential|industrial'  # Business categories
    ]
    
    return any(re.search(pattern, text_lower) for pattern in implicit_patterns)

def clean_sql_query(query: str) -> str:
    """Clean the SQL query by removing any explanatory text and formatting."""
    # Find the actual SQL query by looking for common SQL keywords
    sql_keywords = ['SELECT', 'WITH', 'UPDATE', 'DELETE', 'INSERT']
    lines = query.split('\n')
    
    for line in lines:
        line = line.strip()
        if any(line.upper().startswith(keyword) for keyword in sql_keywords):
            return line
    
    # If no SQL keywords found, try to extract anything that looks like SQL
    for line in lines:
        line = line.strip()
        if 'SELECT' in line.upper():
            return line
    
    return query

def chatbot():
    st.markdown("### 🤖 Insurance Analytics Chatbot")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    prompt = st.chat_input("Ask me anything about insurance data or just chat with me...")
    
    if prompt:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process and display assistant response immediately
        with st.chat_message("assistant"):
            intent_data = extract_intent(prompt)
            
            if intent_data['intent'] in ['greeting', 'farewell', 'help']:
                with st.spinner("Thinking..."):
                    response = get_chatbot_response(prompt, context=st.session_state.messages)
                    st.markdown(response)
                    st.session_state.messages.extend([
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ])
            
            elif is_database_query(prompt):
                with st.spinner("Analyzing your question..."):
                    try:
                        # Get SQL query with enhanced context
                        query_prompt = f"""
                        Generate a SQL query for: {prompt}
                        
                        Rules:
                        1. Use ONLY the 'insurance' table
                        2. Return ONLY the raw SQL query, no explanations
                        3. Do not include words like 'SQL Query:' or 'sql'
                        4. Start directly with SELECT, WITH, or other SQL keyword
                        5. Use proper SQL syntax with no markdown or formatting
                        
                        Example format:
                        SELECT column FROM insurance WHERE condition;
                        
                        Table schema:
                        - Policy (int)
                        - Expiry (varchar(9))
                        - Location (varchar(5))
                        - State (varchar(13))
                        - Region (varchar(9))
                        - Investment (int)
                        - Construction (varchar(11))
                        - BusinessType (varchar(13))
                        - Earthquake (varchar(1))
                        - Flood (varchar(1))
                        - Rating (decimal(3,1))
                        - id (int)
                        """
                        
                        sql_query = get_chatbot_response(query_prompt, context=st.session_state.messages)
                        # Clean the SQL query
                        sql_query = clean_sql_query(sql_query)
                        
                        if sql_query and not any(keyword in sql_query.upper() for keyword in BLOCKED_KEYWORDS):
                            # Execute the query
                            result = execute_safe_query(sql_query)
                            
                            if result:
                                # Format the results
                                df = pd.DataFrame(result)
                                
                                # Get natural language explanation with enhanced context
                                explanation_prompt = f"""
                                Given:
                                1. Original question: {prompt}
                                2. Detected intent: {intent_data['intent']}
                                3. Results from insurance table: {df.to_string()}
                                
                                Provide a friendly and insightful analysis that:
                                - Directly answers the user's question
                                - Highlights key patterns or insights
                                - Suggests relevant follow-up questions
                                - Uses appropriate emojis and formatting
                                - Keeps the tone conversational but professional
                                """
                                
                                explanation = get_chatbot_response(explanation_prompt, context=st.session_state.messages)
                                
                                # Display results immediately
                                st.markdown("**Results:**")
                                st.dataframe(df)
                                st.markdown("**Analysis:**")
                                st.markdown(explanation)
                                
                                # Add to history after displaying
                                st.session_state.messages.extend([
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": f"**Results:**\n\n{df.to_string()}\n\n**Analysis:**\n{explanation}"}
                                ])
                            else:
                                response = "I couldn't find any matching data in the insurance table. Could you please rephrase your question? 🤔\n\nTry being more specific about what you're looking for!"
                                st.markdown(response)
                                st.session_state.messages.extend([
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": response}
                                ])
                        else:
                            response = "I apologize, but I cannot process that query for security reasons. Please try asking in a different way! 🔒"
                            st.markdown(response)
                            st.session_state.messages.extend([
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": response}
                            ])
                    
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        response = "I encountered an error while processing your request. Let me try to help you rephrase your question! 🙏\n\nCould you provide more details about what you're looking for?"
                        st.error(error_msg)
                        st.markdown(response)
                        st.session_state.messages.extend([
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response}
                        ])
            else:
                with st.spinner("Thinking..."):
                    response = get_chatbot_response(prompt, context=st.session_state.messages)
                    st.markdown(response)
                    st.session_state.messages.extend([
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ])
    
    # Add a clear chat button with confirmation
    if st.session_state.messages and st.button("Clear Chat History 🗑️"):
        st.session_state.messages = []
        st.experimental_rerun()

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.header("📊 Business Analytics Dashboard")

# Load CSS style
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Add custom CSS for loading animation
st.markdown("""
    <style>
        div.stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #1a73e8, #4285f4);
        }
        @keyframes gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        .gradient-text {
            background: linear-gradient(45deg, #1a73e8, #4285f4, #fbbc04);
            background-size: 200% 200%;
            animation: gradient 5s ease infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 900;
        }
    </style>
""", unsafe_allow_html=True)

# Load data from MySQL
# @st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data():
    result = view_all_data()
    df = pd.DataFrame(result, columns=["Policy", "Expiry", "Location", "State", "Region", "Investment", "Construction", "BusinessType", "Earthquake", "Flood", "Rating", "id"])
    
    # Convert Expiry to datetime using the specific format
    df['Expiry'] = pd.to_datetime(df['Expiry'], format='%d-%b-%y')
    
    # Convert Y/N to 1/0 for Earthquake and Flood
    df['Earthquake'] = (df['Earthquake'] == 'Y').astype(float)
    df['Flood'] = (df['Flood'] == 'Y').astype(float)
    
    # Ensure Rating and Investment are numeric
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df['Investment'] = pd.to_numeric(df['Investment'], errors='coerce')
    
    return df

df = load_data()

# Sidebar filters with better UI
with st.sidebar:
    # Add logo above filters
    st.image(
        "data/logo1.png",
        caption="",
        use_container_width=True
    )
    
    st.markdown("### Filters")
    st.markdown("---")
    
    region = st.multiselect(
        "🌍 Select Region",
        options=df["Region"].unique(),
        default=df["Region"].unique()
    )
    
    location = st.multiselect(
        "📍 Select Location",
        options=df["Location"].unique(),
        default=df["Location"].unique()
    )
    
    construction = st.multiselect(
        "🏗️ Select Construction",
        options=df["Construction"].unique(),
        default=df["Construction"].unique()
    )

df_selection = df.query("Region == @region & Location == @location & Construction == @construction")

def Home():
    # Data viewer with enhanced UI
    with st.expander("📋 View Dataset"):
        col1, col2 = st.columns([3, 1])
        with col1:
            showData = st.multiselect(
                'Select Columns to Display:',
                df_selection.columns,
                default=["Policy", "Expiry", "Location", "State", "Region", "Investment", "Construction", "BusinessType"]
            )
        with col2:
            st.markdown("##### Download Data")
            st.download_button(
                "📥 Export CSV",
                df_selection[showData].to_csv(index=False).encode('utf-8'),
                "filtered_data.csv",
                "text/csv",
                key='download-csv'
            )
        st.dataframe(
            df_selection[showData],
            use_container_width=True,
            height=300
        )

    # Enhanced metrics calculation
    investment_values = pd.Series(df_selection['Investment'])
    rating_values = pd.Series(df_selection['Rating'])

    total_investment = float(investment_values.sum()) if not investment_values.empty else 0.0
    investment_mode = float(investment_values.mode().iloc[0]) if not investment_values.mode().empty else 0.0
    investment_mean = float(investment_values.mean()) if not investment_values.empty else 0.0
    investment_median = float(investment_values.median()) if not investment_values.empty else 0.0
    rating = float(rating_values.sum()) if not rating_values.empty else 0.0

    # Metrics with better visualization
    st.markdown("### Key Metrics")
    total1, total2, total3, total4, total5 = st.columns(5, gap='small')

    with total1:
        st.info('Total Investment', icon="💰")
        st.metric(
            label="Sum TZS",
            value=f"{total_investment:,.0f}",
            delta=f"{(total_investment - investment_mean):,.0f} vs mean"
        )

    with total2:
        st.info('Most Common Investment', icon="📊")
        st.metric(
            label="Mode TZS",
            value=f"{investment_mode:,.0f}"
        )

    with total3:
        st.info('Average Investment', icon="📈")
        st.metric(
            label="Mean TZS",
            value=f"{investment_mean:,.0f}"
        )

    with total4:
        st.info('Median Investment', icon="⚖️")
        st.metric(
            label="Median TZS",
            value=f"{investment_median:,.0f}"
        )

    with total5:
        st.info('Overall Rating', icon="⭐")
        st.metric(
            label="Rating",
            value=numerize(rating),
            help=f"Total Rating: {rating:,.2f}"
        )

    style_metric_cards(
        background_color="#FFFFFF",
        border_left_color="#1a73e8",
        border_color="#dadce0",
        box_shadow="0 2px 6px rgba(60, 64, 67, 0.1)"
    )

   

def graphs():
    # Business Type Analysis
    investment_by_business_type = df_selection.groupby("BusinessType").count()[["Investment"]].sort_values(by="Investment")
    fig_investment = px.bar(
        investment_by_business_type,
        x="Investment",
        y=investment_by_business_type.index,
        orientation="h",
        title="<b>Investment Distribution by Business Type</b>",
        color_discrete_sequence=["#1a73e8"] * len(investment_by_business_type),
        template="plotly_white",
    )
    fig_investment.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#202124"),
        yaxis=dict(showgrid=True, gridcolor='#dadce0'),
        xaxis=dict(showgrid=True, gridcolor='#dadce0'),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(t=30, l=10, r=10, b=10)
    )

    # State Analysis
    investment_state = df_selection.groupby("State").count()[["Investment"]]
    fig_state = px.line(
        investment_state,
        x=investment_state.index,
        y="Investment",
        markers=True,
        title="<b>Investment Trends by State</b>",
        color_discrete_sequence=["#1a73e8"],
        template="plotly_white",
    )
    fig_state.update_layout(
        xaxis=dict(tickmode="linear", gridcolor='#dadce0'),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor='#dadce0'),
        font=dict(color="#202124"),
        margin=dict(t=30, l=10, r=10, b=10)
    )

    # Layout for charts
    st.markdown("### 📈 Investment Analysis")
    left, right, center = st.columns(3)
    
    with left:
        st.plotly_chart(fig_state, use_container_width=True)
    
    with right:
        st.plotly_chart(fig_investment, use_container_width=True)
    
    with center:
        fig = px.pie(
            df_selection,
            values='Rating',
            names='State',
            title='<b>Rating Distribution by State</b>',
            color_discrete_sequence=px.colors.sequential.Blues_r,
            hole=0.3
        )
        fig.update_layout(
            font=dict(color="#202124"),
            legend_title="States",
            legend_y=0.9,
            margin=dict(t=30, l=10, r=10, b=10)
        )
        fig.update_traces(
            textinfo='percent+label',
            textposition='inside',
            textfont=dict(color="white")
        )
        st.plotly_chart(fig, use_container_width=True)

def Progressbar():
    st.markdown("### Target Progress")
    
    # Add target range slider
    target = st.slider(
        "Set Target Investment (TZS)",
        min_value=1000000000,  # 1 billion
        max_value=10000000000,  # 10 billion
        value=3000000000,  # Default 3 billion
        step=500000000,  # 500 million steps
        format="%d"
    )
    
    current = df_selection["Investment"].sum()
    percent = round((current / target * 100)) if target else 0
    
    # Create a more informative progress display
    col1, col2 = st.columns([3, 1])
    
    with col1:
        mybar = st.progress(0)
        if percent > 100:
            st.success("🎉 Target Achieved!")
        else:
            st.write(f"Progress: {percent}% of {format(target, ',d')} TZS")
            for percent_complete in range(min(percent, 100)):  # Cap at 100%
                time.sleep(0.01)  # Faster animation
                mybar.progress(percent_complete + 1)
    
    with col2:
        gap = target - current
        delta_color = "normal" if gap > 0 else "inverse"
        st.metric(
            "Gap to Target",
            value=f"{format(abs(gap), ',d')} TZS",
            delta=f"{percent}% Complete",
            delta_color=delta_color
        )
    
    # Add some context about the target and current investment
    st.markdown("---")
    st.markdown("### 📊 Investment Summary")
    col3, col4 = st.columns(2)
    
    with col3:
        st.metric(
            "Current Total Investment",
            value=f"{format(current, ',d')} TZS"
        )
    
    with col4:
        remaining = max(0, target - current)
        st.metric(
            "Required Additional Investment",
            value=f"{format(remaining, ',d')} TZS"
        )

def MLPage():
    st.markdown("### 🤖 Machine Learning Analytics")
    
    # Initialize session state for models if not exists
    if 'investment_model' not in st.session_state:
        st.session_state.investment_model = None
    if 'label_encoders' not in st.session_state:
        st.session_state.label_encoders = {}
    if 'feature_cols' not in st.session_state:
        st.session_state.feature_cols = None
    
    # Create tabs for different ML tasks
    ml_task = st.selectbox(
        "Select ML Task",
        ["Investment Prediction", "Risk Analysis", "Customer Segmentation", "Anomaly Detection", "Renewal Prediction"]
    )
    
    if ml_task == "Investment Prediction":
        st.subheader("📈 Investment Prediction Model")
        
        # Feature engineering
        df_ml = df_selection.copy()
        
        # Define categorical columns first
        categorical_cols = ['Location', 'State', 'Region', 'Construction', 'BusinessType']
        
        # Define columns to exclude from features
        exclude_cols = ['Investment', 'id']  # Investment is target, id is identifier
        
        # Convert any boolean or string columns to numeric
        for col in df_ml.columns:
            if df_ml[col].dtype == 'bool':
                df_ml[col] = df_ml[col].astype(int)
            elif df_ml[col].dtype == 'object' and col not in categorical_cols:
                try:
                    df_ml[col] = pd.to_numeric(df_ml[col])
                except:
                    st.warning(f"Column {col} contains non-numeric data and will be excluded from the analysis.")
        
        # Label encode categorical columns
        label_encoders = {}
        
        for col in categorical_cols:
            if col in df_ml.columns:
                label_encoders[col] = LabelEncoder()
                df_ml[col] = label_encoders[col].fit_transform(df_ml[col])
        
        # Select only numeric columns for features, excluding specified columns
        numeric_cols = df_ml.select_dtypes(include=['int64', 'float64']).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Create features and target
        X = df_ml[feature_cols]
        y = df_ml['Investment']
        
        # Create feature information table
        feature_info = pd.DataFrame({
            'Feature Name': feature_cols,
            'Data Type': [str(df_ml[col].dtype) for col in feature_cols],
            'Non-Null Values': [df_ml[col].count() for col in feature_cols],
            'Unique Values': [df_ml[col].nunique() for col in feature_cols],
            'Mean/Mode': [df_ml[col].mean() if df_ml[col].dtype in ['int64', 'float64'] 
                         else df_ml[col].mode().iloc[0] for col in feature_cols]
        })
        
        # Show enhanced feature table
        st.markdown("### 📊 Selected Features Overview")
        st.markdown("""
        <style>
        .feature-table {
            font-size: 0.9rem;
            border-collapse: collapse;
            margin: 1rem 0;
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        .feature-table th {
            background-color: #1a73e8;
            color: white;
            font-weight: 500;
            text-align: left;
            padding: 12px 15px;
        }
        .feature-table td {
            padding: 10px 15px;
            border-top: 1px solid #eee;
        }
        .feature-table tr:hover {
            background-color: #f8f9fa;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Format the table values
        feature_info['Mean/Mode'] = feature_info.apply(
            lambda x: f"{x['Mean/Mode']:,.2f}" if isinstance(x['Mean/Mode'], (int, float)) 
            else x['Mean/Mode'], axis=1
        )
        feature_info['Non-Null Values'] = feature_info['Non-Null Values'].map('{:,}'.format)
        feature_info['Unique Values'] = feature_info['Unique Values'].map('{:,}'.format)
        
        # Add feature descriptions
        feature_descriptions = {
            'Rating': 'Customer rating score (1-10)',
            'Earthquake': 'Binary indicator for earthquake risk (0/1)',
            'Flood': 'Binary indicator for flood risk (0/1)',
            'Location': 'Geographic location category',
            'State': 'State/region name',
            'Region': 'Broader geographic region',
            'Construction': 'Type of construction',
            'BusinessType': 'Category of business',
            'Days_To_Expiry': 'Number of days until policy expiration'
        }
        
        feature_info['Description'] = [
            feature_descriptions.get(feature, 'Feature description not available') 
            for feature in feature_info['Feature Name']
        ]
        
        # Display the table with custom styling
        st.markdown(
            feature_info.to_html(
                index=False,
                classes=['feature-table'],
                escape=False
            ),
            unsafe_allow_html=True
        )
        
        # Add summary statistics
        st.markdown("### 📈 Feature Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Features",
                len(feature_cols),
                help="Number of features used in the model"
            )
        with col2:
            numeric_features = len([col for col in feature_cols 
                                 if df_ml[col].dtype in ['int64', 'float64']])
            st.metric(
                "Numeric Features",
                numeric_features,
                help="Number of numerical features"
            )
        with col3:
            categorical_features = len([col for col in feature_cols 
                                     if col in categorical_cols])
            st.metric(
                "Categorical Features",
                categorical_features,
                help="Number of categorical features (encoded)"
            )
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model selection
        model_type = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                if model_type == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate regression metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MSE", f"{mse:,.2f}")
                with col2:
                    st.metric("RMSE", f"{rmse:,.2f}")
                with col3:
                    st.metric("MAE", f"{mae:,.2f}")
                with col4:
                    st.metric("R² Score", f"{r2:.3f}")
                
                # Save model and features to session state
                st.session_state.investment_model = model
                st.session_state.label_encoders = label_encoders
                st.session_state.feature_cols = feature_cols
                
                # Feature importance plot
                if model_type == "Random Forest":
                    importance_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                else:
                    importance_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': abs(model.coef_)
                    }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance'
                )
                st.plotly_chart(fig)
        
        # Prediction interface
        st.subheader("Make Predictions")
        
        if st.session_state.investment_model is None:
            st.warning("Please train a model first before making predictions.")
        else:
            col1, col2 = st.columns(2)
            pred_data = {}
            
            # Create input fields for each feature
            for i, feature in enumerate(st.session_state.feature_cols):
                with col1 if i < len(st.session_state.feature_cols)/2 else col2:
                    if feature in categorical_cols:
                        # For categorical features, show original values
                        original_values = df_selection[feature].unique()
                        value = st.selectbox(f"{feature}", original_values)
                        # Convert to encoded value
                        pred_data[feature] = st.session_state.label_encoders[feature].transform([value])[0]
                    else:
                        # For numerical features
                        min_val = float(df_ml[feature].min())
                        max_val = float(df_ml[feature].max())
                        step = (max_val - min_val) / 100
                        value = st.number_input(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=float(df_ml[feature].mean()),
                            step=step
                        )
                        pred_data[feature] = value
            
            if st.button("Predict Investment"):
                # Create prediction dataframe
                pred_df = pd.DataFrame([pred_data])
                
                # Make prediction
                prediction = st.session_state.investment_model.predict(pred_df)[0]
                st.success(f"Predicted Investment: TZS {prediction:,.2f}")
    
    elif ml_task == "Risk Analysis":
        st.subheader("🎯 Risk Analysis")
        
        # Feature engineering for risk analysis
        df_ml = df_selection.copy()
        
        # Convert any boolean columns to numeric
        for col in df_ml.columns:
            if df_ml[col].dtype == 'bool':
                df_ml[col] = df_ml[col].astype(int)
            elif df_ml[col].dtype == 'object':
                try:
                    df_ml[col] = pd.to_numeric(df_ml[col])
                except:
                    continue
        
        # Calculate risk score using numeric columns only
        numeric_cols = df_ml.select_dtypes(include=['int64', 'float64']).columns
        risk_features = [col for col in ['Earthquake', 'Flood', 'Rating'] if col in numeric_cols]
        
        if len(risk_features) > 0:
            df_ml['Risk_Score'] = df_ml[risk_features].mean(axis=1)
            
            # Train risk classifier
            X = df_ml[['Investment', 'Risk_Score']]
            risk_threshold = df_ml['Risk_Score'].quantile(0.75)
            y = (df_ml['Risk_Score'] > risk_threshold).astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if st.button("Analyze Risks"):
                with st.spinner("Analyzing risks..."):
                    clf = RandomForestClassifier(n_estimators=100, random_state=42)
                    clf.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = clf.predict(X_test)
                    
                    # Calculate classification metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    
                    # Display metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.2%}")
                    with col2:
                        st.metric("Precision", f"{precision:.2%}")
                    with col3:
                        st.metric("Recall", f"{recall:.2%}")
                    with col4:
                        st.metric("F1 Score", f"{f1:.2%}")
                    
                    # Plot risk distribution
                    fig = px.scatter(
                        df_ml,
                        x='Investment',
                        y='Risk_Score',
                        color='Rating',
                        title='Risk Distribution',
                        labels={'Investment': 'Investment Amount', 'Risk_Score': 'Risk Score'}
                    )
                    st.plotly_chart(fig)
                    
                    # Show risk statistics
                    st.write("Risk Statistics:")
                    st.write(f"- High Risk Policies: {sum(y):,} ({(sum(y)/len(y)*100):.1f}%)")
        else:
            st.warning("Required risk features (Earthquake, Flood, Rating) are not available in numeric format.")
    
    elif ml_task == "Customer Segmentation":
        st.subheader("👥 Customer Segmentation")
        
        # Prepare data for clustering
        df_ml = df_selection.copy()
        
        # Convert any boolean columns to numeric
        for col in df_ml.columns:
            if df_ml[col].dtype == 'bool':
                df_ml[col] = df_ml[col].astype(int)
            elif df_ml[col].dtype == 'object':
                try:
                    df_ml[col] = pd.to_numeric(df_ml[col])
                except:
                    continue
        
        # Select numeric features for clustering
        numeric_cols = df_ml.select_dtypes(include=['int64', 'float64']).columns
        features = [col for col in ['Investment', 'Rating'] if col in numeric_cols]
        
        if len(features) > 0:
            X = df_ml[features]
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # K-means clustering
            n_clusters = st.slider("Number of Segments", 2, 6, 3)
            
            if st.button("Generate Segments"):
                with st.spinner("Generating customer segments..."):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    # Calculate clustering metrics
                    silhouette_avg = silhouette_score(X_scaled, clusters)
                    inertia = kmeans.inertia_
                    
                    # Display metrics in columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Silhouette Score", f"{silhouette_avg:.3f}", 
                                help="Measures how similar an object is to its own cluster compared to other clusters. Range: [-1, 1]")
                    with col2:
                        st.metric("Inertia", f"{inertia:,.0f}",
                                help="Sum of squared distances of samples to their closest cluster center")
                    
                    # Add cluster labels to dataframe
                    df_ml['Segment'] = clusters
                    
                    # Plot segments
                    fig = px.scatter(
                        df_ml,
                        x=features[0],
                        y=features[1] if len(features) > 1 else features[0],
                        color='Segment',
                        title='Customer Segments',
                        labels={features[0]: features[0], features[1] if len(features) > 1 else features[0]: features[1] if len(features) > 1 else features[0]}
                    )
                    st.plotly_chart(fig)
                    
                    # Show segment statistics
                    for i in range(n_clusters):
                        segment_data = df_ml[df_ml['Segment'] == i]
                        st.write(f"Segment {i+1}:")
                        st.write(f"- Count: {len(segment_data):,}")
                        for feature in features:
                            st.write(f"- Average {feature}: {segment_data[feature].mean():,.2f}")
        else:
            st.warning("Required features (Investment, Rating) are not available in numeric format.")
    
    elif ml_task == "Anomaly Detection":
        st.subheader("🔍 Anomaly Detection")
        
        # Prepare data for anomaly detection
        df_ml = df_selection.copy()
        
        # Convert any boolean columns to numeric
        for col in df_ml.columns:
            if df_ml[col].dtype == 'bool':
                df_ml[col] = df_ml[col].astype(int)
            elif df_ml[col].dtype == 'object':
                try:
                    df_ml[col] = pd.to_numeric(df_ml[col])
                except:
                    continue
        
        # Select numeric features for anomaly detection
        numeric_cols = df_ml.select_dtypes(include=['int64', 'float64']).columns
        features = [col for col in ['Investment', 'Rating', 'Earthquake', 'Flood'] if col in numeric_cols]
        
        if len(features) > 0:
            X = df_ml[features]
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            contamination = st.slider("Anomaly Threshold (%)", 1, 10, 5) / 100
            
            if st.button("Detect Anomalies"):
                with st.spinner("Detecting anomalies..."):
                    # Train isolation forest
                    iso_forest = IsolationForest(contamination=contamination, random_state=42)
                    anomalies = iso_forest.fit_predict(X_scaled)
                    
                    # Calculate anomaly detection metrics
                    n_anomalies = sum(anomalies == -1)
                    anomaly_ratio = n_anomalies / len(anomalies)
                    
                    # Display metrics in columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Number of Anomalies", f"{n_anomalies:,}")
                    with col2:
                        st.metric("Anomaly Ratio", f"{anomaly_ratio:.2%}")
                    
                    # Add anomaly labels to dataframe
                    df_ml['Is_Anomaly'] = anomalies == -1
                    
                    # Plot anomalies
                    fig = px.scatter(
                        df_ml,
                        x=features[0],
                        y=features[1] if len(features) > 1 else features[0],
                        color='Is_Anomaly',
                        title='Anomaly Detection',
                        labels={features[0]: features[0], features[1] if len(features) > 1 else features[0]: features[1] if len(features) > 1 else features[0]}
                    )
                    st.plotly_chart(fig)
                    
                    # Show anomaly statistics
                    anomaly_data = df_ml[df_ml['Is_Anomaly']]
                    for feature in features:
                        st.write(f"- Average {feature}: {anomaly_data[feature].mean():,.2f}")
        else:
            st.warning("Required features are not available in numeric format.")
    
    elif ml_task == "Renewal Prediction":
        st.subheader("🔄 Renewal Prediction")
        
        # Feature engineering for renewal prediction
        df_ml = df_selection.copy()
        
        # Convert Expiry to datetime if it's not already
        try:
            df_ml['Expiry'] = pd.to_datetime(df_ml['Expiry'])
            df_ml['Days_To_Expiry'] = (df_ml['Expiry'] - pd.Timestamp.now()).dt.days
            
            # Convert any boolean columns to numeric
            for col in df_ml.columns:
                if df_ml[col].dtype == 'bool':
                    df_ml[col] = df_ml[col].astype(int)
                elif df_ml[col].dtype == 'object':
                    try:
                        df_ml[col] = pd.to_numeric(df_ml[col])
                    except:
                        continue
            
            # Create synthetic renewal data for demonstration
            np.random.seed(42)
            df_ml['Will_Renew'] = (df_ml['Rating'] * np.random.random(len(df_ml)) > 2).astype(int)
            
            # Select numeric features
            numeric_cols = df_ml.select_dtypes(include=['int64', 'float64']).columns
            features = [col for col in ['Investment', 'Rating', 'Days_To_Expiry'] if col in numeric_cols]
            
            if len(features) > 0:
                X = df_ml[features]
                y = df_ml['Will_Renew']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if st.button("Train Renewal Model"):
                    with st.spinner("Training renewal prediction model..."):
                        # Train logistic regression
                        model = LogisticRegression(random_state=42)
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        y_prob = model.predict_proba(X_test)[:, 1]
                        
                        # Calculate classification metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        
                        # Display metrics in columns
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.2%}")
                        with col2:
                            st.metric("Precision", f"{precision:.2%}")
                        with col3:
                            st.metric("Recall", f"{recall:.2%}")
                        with col4:
                            st.metric("F1 Score", f"{f1:.2%}")
                        
                        # Show model performance
                        st.write("Renewal Statistics:")
                        st.write(f"- Likely to Renew: {sum(y):,} ({(sum(y)/len(y)*100):.1f}%)")
                        st.write(f"- Average Days to Expiry: {df_ml['Days_To_Expiry'].mean():.0f}")
            else:
                st.warning("Required features are not available in numeric format.")
        except Exception as e:
            st.error(f"Error processing Expiry dates: {str(e)}")

def distanalysis():
     # Enhanced distribution plots
     with st.expander("📊 Distribution Analysis"):
        st.markdown("### Frequency Distributions")
        fig, ax = plt.subplots(figsize=(16, 8))
        df_selection.select_dtypes("number").hist(
            ax=ax,
            color='#1a73e8',
            alpha=0.6,
            bins=30,
            edgecolor='white'
        )
        plt.tight_layout()
        st.pyplot(fig)
        # Add feature exploration section with enhanced styling
        st.markdown("### 📊 Distribution Analysis")
        st.markdown("Explore the distribution of numerical features across business types")

        col1, col2 = st.columns([3, 1])
        with col1:
            feature_y = st.selectbox(
                'Select Feature for Analysis',
                df_selection.select_dtypes("number").columns,
                help="Choose a numerical feature to analyze its distribution"
            )

        with col2:
            st.markdown("##### Chart Type")
            chart_type = st.radio(
                "",
                ["Box Plot", "Violin Plot"],
                horizontal=True
            )

        if chart_type == "Box Plot":
            fig2 = go.Figure(
                data=[go.Box(
                    x=df_selection['BusinessType'],
                    y=df_selection[feature_y],
                    marker_color="#1a73e8"
                )],
                layout=go.Layout(
                    title=f"Distribution of {feature_y} by Business Type",
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    xaxis=dict(showgrid=True, gridcolor='#dadce0'),
                    yaxis=dict(showgrid=True, gridcolor='#dadce0'),
                    font=dict(color='#202124'),
                    boxmode='group'
                )
            )
        else:
            fig2 = go.Figure(
                data=[go.Violin(
                    x=df_selection['BusinessType'],
                    y=df_selection[feature_y],
                    box_visible=True,
                    line_color="#1a73e8",
                    fillcolor="#1a73e8",
                    opacity=0.6
                )],
                layout=go.Layout(
                    title=f"Distribution of {feature_y} by Business Type",
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    xaxis=dict(showgrid=True, gridcolor='#dadce0'),
                    yaxis=dict(showgrid=True, gridcolor='#dadce0'),
                    font=dict(color='#202124')
                )
            )

        st.plotly_chart(fig2, use_container_width=True)

def sideBar():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Dashboard", "Progress", "ML Analytics", "Chatbot"],
            icons=["house-door-fill", "graph-up-arrow", "robot", "chat-dots-fill"],
            menu_icon="layout-text-sidebar-reverse",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#1a73e8", "font-size": "1rem"},
                "nav-link": {
                    "font-size": "0.9rem",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#1a73e8"},
            }
        )
        
    if selected == "Dashboard":
        Home()
        graphs()
        distanalysis()
    elif selected == "Progress":
        Progressbar()
    elif selected == "ML Analytics":
        MLPage()
    elif selected == "Chatbot":
        chatbot()

sideBar()


# Add footer with timestamp
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #666;'>"
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    "</div>",
    unsafe_allow_html=True
)

# Hide Streamlit elements
hide_st_style = """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
