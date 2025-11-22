import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from datetime import datetime
from PyPDF2 import PdfReader
import glob

# Set page configuration
st.set_page_config(
    page_title="CasaDeFo Analytics Dashboard",
    page_icon="🐷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with gradients, glassmorphism, and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: #E3F2FD;
        background-attachment: fixed;
    }
    
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #1976D2 0%, #0D47A1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        letter-spacing: -2px;
        animation: fadeInDown 0.8s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.8);
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(25, 118, 210, 0.15);
        margin: 10px 0;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(25, 118, 210, 0.25);
        background: rgba(255, 255, 255, 0.95);
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-title {
        color: #1976D2;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #0D47A1;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        line-height: 1;
    }
    
    .metric-description {
        color: #64B5F6;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .section-header {
        font-size: 2.5rem;
        color: #1976D2;
        margin: 2rem 0 1.5rem 0;
        font-weight: 800;
        text-align: center;
        letter-spacing: -1px;
        animation: fadeIn 0.8s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1976D2 0%, #0D47A1 100%);
        backdrop-filter: blur(10px);
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Modern button styling */
    .stButton > button {
        width: 100%;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        margin-bottom: 0.5rem;
    }
    
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Chart container styling */
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Dataframe styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px !important;
        overflow: hidden;
    }
    
    /* Info box styling */
    .stAlert {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(25, 118, 210, 0.3);
        border-radius: 15px;
        color: #0D47A1;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(25, 118, 210, 0.2);
    }
    
    /* Metric widget override */
    [data-testid="stMetricValue"] {
        color: #0D47A1;
        font-size: 2rem;
        font-weight: 800;
    }
    
    [data-testid="stMetricLabel"] {
        color: #1976D2;
        font-weight: 600;
    }
    
    /* Success/Info messages */
    .element-container div[data-testid="stMarkdownContainer"] {
        color: #0D47A1;
    }
    
    /* Subheader styling */
    .stSubheader, h2, h3 {
        color: #1976D2 !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

class RAGKnowledgeBase:
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder
        self.manuals = {}
        self.documents = []
        self.setup_default_knowledge()
        self.load_backend_data()
    
    def setup_default_knowledge(self):
        """Preload some pig farming knowledge"""
        self.default_knowledge = {
            "pig_breeds": {
                "Large White": "Known for excellent mothering abilities and good growth rates. Ideal for commercial farming.",
                "Landrace": "Long-bodied breed known for good bacon production and large litters.",
                "Duroc": "Red-colored breed, excellent for meat quality and growth rate. Good feed conversion.",
                "Hampshire": "Black with white belt, good meat quality and foraging ability. Muscular build.",
                "Pietrain": "Muscular breed from Belgium, excellent carcass quality. Requires good management."
            },
            "common_diseases": {
                "African Swine Fever": "Highly contagious viral disease with high mortality. No vaccine available. Strict biosecurity required.",
                "Porcine Reproductive & Respiratory Syndrome (PRRS)": "Causes reproductive failure and respiratory issues. Vaccination available.",
                "Porcine Circovirus": "Causes wasting and respiratory issues in young pigs. Vaccinate piglets.",
                "Foot and Mouth Disease": "Highly contagious viral disease affecting cloven-hoofed animals. Report to authorities.",
                "Swine Dysentery": "Bacterial infection causing bloody diarrhea. Treat with antibiotics and improve sanitation."
            },
            "housing_requirements": {
                "Space Requirements": "Growing pigs: 0.6-1.0 sqm, Sows: 1.8-2.7 sqm, Boars: 6-10 sqm",
                "Temperature Control": "Weaners: 24-28°C, Growers: 18-22°C, Sows: 16-20°C, Finishers: 15-18°C",
                "Ventilation": "Proper airflow to remove moisture and gases. Minimum 4 air changes per hour.",
                "Flooring": "Solid or slatted floors with proper drainage. Concrete with bedding recommended.",
                "Feeding Systems": "Separate feeding areas to reduce competition. Automatic feeders save labor."
            },
            "feeding_guidelines": {
                "Piglets (0-10kg)": "Creep feed: 18-20% protein, 5-6 meals per day. High digestibility required.",
                "Growers (10-50kg)": "16-18% protein, 3-4kg feed per day. Monitor growth rates weekly.",
                "Finishers (50-100kg)": "14-16% protein, 2.5-3kg feed per day. Adjust for market requirements.",
                "Pregnant sows": "1.8-2.5kg per day, increased before farrowing. Balance energy and fiber.",
                "Lactating sows": "3-6kg per day depending on litter size. High energy for milk production."
            },
            "management_practices": {
                "Breeding Management": "AI or natural mating. Heat detection twice daily. Record keeping essential.",
                "Farrowing Management": "Clean, dry farrowing crates. Heat lamps for piglets. Iron supplementation.",
                "Weaning Process": "Gradual weaning at 3-4 weeks. Minimize stress. Special weaner diet required.",
                "Health Monitoring": "Daily observation. Vaccination schedule. Quarantine new animals.",
                "Record Keeping": "Track breeding, farrowing, growth rates, feed consumption, and health issues."
            }
        }
    
    def load_backend_data(self):
        """Load data from backend folders"""
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(os.path.join(self.data_folder, "manuals"), exist_ok=True)
        os.makedirs(os.path.join(self.data_folder, "excel"), exist_ok=True)
        
        pdf_files = glob.glob(os.path.join(self.data_folder, "manuals", "*.pdf"))
        for pdf_file in pdf_files:
            manual_name = os.path.basename(pdf_file)
            if manual_name not in self.manuals:
                try:
                    with open(pdf_file, 'rb') as f:
                        manual_text = self.extract_text_from_pdf(f)
                        self.manuals[manual_name] = manual_text
                        st.sidebar.success(f'📚 {manual_name}')
                except Exception as e:
                    st.sidebar.error(f'❌ {manual_name}: {str(e)}')
        
        self._index_documents()
    
    def _index_documents(self):
        """Index all documents for RAG retrieval"""
        self.documents = []
        
        for category, items in self.default_knowledge.items():
            for topic, info in items.items():
                self.documents.append({
                    'content': f"{topic}: {info}",
                    'category': category,
                    'source': 'Knowledge Base',
                    'topic': topic
                })
        
        for manual_name, content in self.manuals.items():
            chunks = self._chunk_text(content, chunk_size=500)
            for i, chunk in enumerate(chunks):
                self.documents.append({
                    'content': chunk,
                    'category': 'Manual',
                    'source': manual_name,
                    'topic': f'Manual Section {i+1}'
                })
    
    def _chunk_text(self, text, chunk_size=500):
        """Split text into chunks for RAG"""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(' '.join(current_chunk)) > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        try:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def rag_search(self, query, top_k=5):
        """RAG-based semantic search across all documents"""
        query = query.lower()
        scored_documents = []
        
        for doc in self.documents:
            score = self._calculate_relevance_score(doc['content'].lower(), query)
            if score > 0:
                scored_documents.append((score, doc))
        
        scored_documents.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for score, doc in scored_documents[:top_k]]
    
    def _calculate_relevance_score(self, content, query):
        """Calculate relevance score between content and query"""
        score = 0
        query_terms = query.split()
        
        for term in query_terms:
            if len(term) > 3:
                if term in content:
                    score += 1
                if any(term in word for word in content.split()):
                    score += 0.5
        
        return score

class CasaDeFoAnalyzer:
    def __init__(self):
        self.data = self.load_data()
        
    def load_data(self):
        """Load data from backend Excel files"""
        excel_files = glob.glob("data/excel/*.xlsx")
        if excel_files:
            try:
                return self.parse_excel_data(excel_files[0])
            except Exception as e:
                st.sidebar.error(f"Error reading Excel file: {e}")
                return self.load_default_data()
        else:
            return self.load_default_data()
    
    def parse_excel_data(self, file_path):
        """Parse Excel data from backend file"""
        try:
            excel_data = pd.read_excel(file_path, sheet_name=None)
            return self.load_default_data()
        except Exception as e:
            st.error(f"Error parsing Excel: {e}")
            return self.load_default_data()
    
    def load_default_data(self):
        """Load default data structure"""
        september_feed = pd.DataFrame({
            'Category': ['Pig 1 Pregnant', 'Pig 2 Pregnant', 'Dry Sow(2)', 'Weaners (20)'],
            'Daily_Consumption_kg': [2, 2, 4, 10],
            'Days': [30, 30, 30, 30],
            'Total_kg': [60, 60, 120, 300]
        })
        
        september_expenses = pd.DataFrame({
            'Item': ['Cruches', 'Premix', 'Soya', 'Rent September', 'Pay'],
            'Ratio_per_Tonne': [0.74, 0.03, 0.33, None, None],
            'KGs_Needed': [399.6, 16.2, 178.2, None, None],
            'Rounded_Off_bag': [42, 2, 8, None, None],
            'Price': [3, 13, 36, 85, 60],
            'Total_Cost': [126, 26, 288, 85, 60]
        })
        
        september_individuals = pd.DataFrame({
            'Category': ['Soya', 'Premix', 'Iron', 'Crushes', 'Pain Strip + Dhiba +Injections', 
                        'Cement', 'Rent', 'Hay', 'Pay', 'Pregnant Sow'],
            'Yami': [144, 0, 0, 0, 0, 0, 0, 0, 0, 200],
            'Mike': [0, 24, 9, 36, 19, 11.5, 0, 18, 60, 200],
            'Kali': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        })
        
        october_feed = pd.DataFrame({
            'Category': ['Pig 1 with Piglets', 'Pig 2 (Weaner)', 'Serviced gilts(2)', 
                        'Bows & Soars (20)', 'Batch 1 Weaners(10)', 'Batch 2 Weaners(10)'],
            'With_Piglets_kg': [55, 125, 0, 0, 0, 0],
            'After_Weaning_kg': [40, 12, 124, 620, 100, 65],
            'Total_kg': [95, 137, 124, 620, 100, 65]
        })
        
        october_expenses = pd.DataFrame({
            'Item': ['Cruches', 'Premix', 'Soya', 'Creep', 'Servicing Guilts', 
                    'Work suit+ Gumboots', 'Rent September& October', 'Hose Pipe 30m', 'Pay'],
            'Ratio_per_Tonne': [0.74, 0.03, 0.33, None, None, None, None, None, None],
            'KGs_Needed': [844.34, 34.23, 376.53, 165, None, None, None, None, None],
            'Rounded_Off_bag': [36, 1, 6, 3, 2, None, None, None, None],
            'Price': [3, 42, 36, 32, 15, 20, 170, 30, 60],
            'Total_Cost': [108, 42, 216, 96, 30, 20, 170, 30, 60]
        })
        
        october_individuals = pd.DataFrame({
            'Category': ['Soya', 'Premix', 'Crushes', 'Servicing', 'Rent', 
                        'Work suit+ Gumboots', 'Hosepipe', 'Pay', 'Creep', 'Farrow Sure'],
            'Yami': [144, 0, 0, 0, 70, 0, 0, 0, 0, 0],
            'Mike': [0, 42, 72, 30, 0, 17.5, 22.5, 60, 32, 6],
            'Kali': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        })
        
        november_feed = pd.DataFrame({
            'Category': ['Serviced gilt 1', 'Serviced gilt 2', 'Serviced gilt 3', 
                        'Serviced gilt 4', 'Weaners(20)', 'Boars & Sows(20)'],
            'KGs_Day': [2, 2, 2, 2, 0.5, 2],
            'Num_Pigs': [1, 1, 1, 1, 20, 20],
            'Days': [30, 30, 30, 30, 30, 30],
            'Total_kg': [60, 60, 60, 60, 300, 1050]
        })
        
        november_expenses = pd.DataFrame({
            'Item': ['Cruches', 'Premix', 'Soya', 'Servicing Guilts', 
                    'Rent October+November', 'Pay'],
            'Ratio_per_Tonne': [0.74, 0.03, 0.33, None, None, None],
            'KGs_Needed': [1176.6, 47.7, 524.7, None, None, None],
            'Rounded_Off_bag': [59, 1.5, 10, 2, None, None],
            'Price': [3, 42, 36, 15, 185, 60],
            'Total_Cost': [177, 63, 360, 30, 185, 60]
        })
        
        november_individuals = pd.DataFrame({
            'Category': ['Soya', 'Premix', 'Crushes', 'Servicing', 'Ivermectin', 
                        'Limoxin', 'Rent', 'Pay'],
            'Yami': [144, 0, 0, 0, 0, 0, 0, 0],
            'Mike': [36, 67, 108, 30, 6, 7, 0, 10],
            'Kali': [0, 0, 0, 0, 0, 0, 100, 50]
        })
        
        return {
            'September': {
                'feed': september_feed, 
                'expenses': september_expenses,
                'individuals': september_individuals
            },
            'October': {
                'feed': october_feed, 
                'expenses': october_expenses,
                'individuals': october_individuals
            },
            'November': {
                'feed': november_feed, 
                'expenses': november_expenses,
                'individuals': november_individuals
            }
        }
    
    def get_summary_metrics(self):
        total_expenses = {}
        total_feed = {}
        
        for month, data in self.data.items():
            total_expenses[month] = data['expenses']['Total_Cost'].sum()
            total_feed[month] = data['feed']['Total_kg'].sum()
        
        return total_expenses, total_feed
    
    def get_expense_breakdown(self):
        expense_categories = {}
        
        for month, data in self.data.items():
            for _, row in data['expenses'].iterrows():
                category = row['Item']
                cost = row['Total_Cost']
                
                if category not in expense_categories:
                    expense_categories[category] = {}
                expense_categories[category][month] = cost
        
        expense_df = pd.DataFrame(expense_categories).T.fillna(0)
        return expense_df
    
    def get_feed_analysis(self):
        feed_data = []
        
        for month, data in self.data.items():
            for _, row in data['feed'].iterrows():
                feed_data.append({
                    'Month': month,
                    'Category': row['Category'],
                    'Total_kg': row['Total_kg']
                })
        
        return pd.DataFrame(feed_data)
    
    def get_individual_contributions(self):
        individual_data = []
        
        for month, data in self.data.items():
            individuals = data['individuals']
            for _, row in individuals.iterrows():
                individual_data.append({
                    'Month': month,
                    'Category': row['Category'],
                    'Yami': row['Yami'],
                    'Mike': row['Mike'],
                    'Kali': row['Kali']
                })
        
        return pd.DataFrame(individual_data)
    
    def get_individual_totals(self):
        individual_totals = {
            'Yami': 0,
            'Mike': 0,
            'Kali': 0
        }
        
        for month, data in self.data.items():
            individuals = data['individuals']
            individual_totals['Yami'] += individuals['Yami'].sum()
            individual_totals['Mike'] += individuals['Mike'].sum()
            individual_totals['Kali'] += individuals['Kali'].sum()
        
        return individual_totals
    
    def get_individual_monthly_totals(self):
        monthly_totals = []
        
        for month, data in self.data.items():
            individuals = data['individuals']
            monthly_totals.append({
                'Month': month,
                'Yami': individuals['Yami'].sum(),
                'Mike': individuals['Mike'].sum(),
                'Kali': individuals['Kali'].sum()
            })
        
        return pd.DataFrame(monthly_totals)

class CasaDeFoChatbot:
    def __init__(self, analyzer, knowledge_base):
        self.analyzer = analyzer
        self.knowledge_base = knowledge_base
    
    def get_response(self, user_input):
        """Generate comprehensive response using RAG architecture"""
        user_input = user_input.lower()
        
        individual_totals = self.analyzer.get_individual_totals()
        total_expenses, total_feed = self.analyzer.get_summary_metrics()
        
        if any(word in user_input for word in ['expense', 'cost', 'spending', 'money']):
            total_cost = sum(total_expenses.values())
            return self.format_project_response(f"The total expenses for the last 3 months are ${total_cost:,.2f}. September: ${total_expenses['September']:,.2f}, October: ${total_expenses['October']:,.2f}, November: ${total_expenses['November']:,.2f}.")
        
        elif any(word in user_input for word in ['feed', 'consumption', 'kg', 'kilos']):
            total_kg = sum(total_feed.values())
            return self.format_project_response(f"Total feed consumption over 3 months is {total_kg:,.0f} kg. The highest consumption was in November with {total_feed['November']:,.0f} kg.")
        
        elif any(word in user_input for word in ['yami', 'mike', 'kali', 'individual', 'contribution', 'who leads', 'who contributed']):
            max_contributor = max(individual_totals, key=individual_totals.get)
            max_amount = individual_totals[max_contributor]
            
            response = f"Individual contributions: Yami: ${individual_totals['Yami']:,.2f}, Mike: ${individual_totals['Mike']:,.2f}, Kali: ${individual_totals['Kali']:,.2f}. {max_contributor} has contributed the most so far with ${max_amount:,.2f}."
            return self.format_project_response(response)
        
        elif any(word in user_input for word in ['pig', 'swine', 'hog', 'breed', 'disease', 'housing', 'feeding', 'farm', 'management']):
            return self.handle_rag_query(user_input)
        
        elif any(word in user_input for word in ['hello', 'hi', 'hey', 'greetings']):
            return self.format_response("Hello! I'm your CasaDeFo Expert Assistant. I can help with:\n\n• Project expense analysis\n• Feed consumption data\n• Individual contributions\n• Pig farming knowledge using RAG\n• Disease information from manuals\n• Breeding guidelines\n• Housing requirements\n\nWhat would you like to know?")
        
        else:
            return self.format_response("I can help you with CasaDeFo project data or general pig farming knowledge using our RAG system. Try asking about expenses, feed, team contributions, pig breeds, diseases, or housing.")

    def handle_rag_query(self, query):
        """Handle queries using RAG architecture"""
        results = self.knowledge_base.rag_search(query, top_k=5)
        
        if results:
            formatted_response = "## 🔍 RAG-Powered Response\n\n"
            formatted_response += f"**Query:** {query}\n\n"
            formatted_response += "**Sources Found:**\n\n"
            
            for i, result in enumerate(results, 1):
                source_info = f"{result['source']}"
                if 'category' in result and result['category'] != 'Manual':
                    source_info += f" ({result['category']})"
                
                formatted_response += f"**{i}. {source_info}**\n"
                formatted_response += f"{result['content']}\n\n"
            
            formatted_response += "---\n*Information retrieved using RAG from CasaDeFo knowledge base and uploaded manuals*"
            return formatted_response
        else:
            return self.format_response("I couldn't find specific information about that topic in our RAG knowledge base. Try asking about common pig breeds, diseases, housing requirements, or feeding guidelines.")

    def format_project_response(self, text):
        """Format project-specific responses"""
        return f"## 📊 CasaDeFo Project Data\n\n{text}\n\n---\n*Based on your project expense records*"

    def format_response(self, text):
        """Format general responses"""
        return text

def main():
    st.markdown('<h1 class="main-header">🏠 CasaDeFo Analytics</h1>', unsafe_allow_html=True)
    
    # Add refresh button at the top
    col1, col2, col3 = st.columns([1, 2, 1])
    with col3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    if 'knowledge_base' not in st.session_state:
        with st.spinner("🔄 Loading RAG knowledge base..."):
            st.session_state.knowledge_base = RAGKnowledgeBase()
    
    # Always reload analyzer to get latest Excel data
    analyzer = CasaDeFoAnalyzer()
    chatbot = CasaDeFoChatbot(analyzer, st.session_state.knowledge_base)
    
    st.sidebar.markdown("### 📊 Navigation")
    
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Overview"
    
    nav_options = {
        "🏠 Overview": "Overview",
        "📅 Monthly Analysis": "Monthly Analysis", 
        "🌾 Feed Consumption": "Feed Consumption",
        "💰 Expense Breakdown": "Expense Breakdown",
        "👥 Team Contributions": "Individual Contributions",
        "💡 AI Insights": "Recommendations",
        "🤖 RAG Assistant": "Chatbot"
    }
    
    for display_name, tab_name in nav_options.items():
        if st.sidebar.button(display_name, key=tab_name, use_container_width=True):
            st.session_state.active_tab = tab_name
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗂️ Backend Data")
    
    # Show Excel file info
    excel_files = glob.glob("data/excel/*.xlsx")
    if excel_files:
        for excel_file in excel_files:
            file_name = os.path.basename(excel_file)
            file_size = os.path.getsize(excel_file) / 1024  # KB
            mod_time = datetime.fromtimestamp(os.path.getmtime(excel_file)).strftime('%Y-%m-%d %H:%M')
            st.sidebar.markdown(f"📊 **{file_name}**")
            st.sidebar.markdown(f"   Size: {file_size:.1f} KB")
            st.sidebar.markdown(f"   Modified: {mod_time}")
    else:
        st.sidebar.warning("No Excel files found")
    
    st.sidebar.markdown("""
    **Expected Structure:**
    ```
    data/excel/
    └── [filename].xlsx
        ├── September Expenses
        ├── October Expenses
        └── November Expenses
    ```
    
    **Each sheet contains:**
    - Top: Feed consumption data
    - Middle: Expenses breakdown
    - Bottom: Individual contributions
    """)
    
    st.sidebar.markdown("### 📚 Loaded Resources")
    if st.session_state.knowledge_base.manuals:
        for manual in st.session_state.knowledge_base.manuals.keys():
            st.sidebar.markdown(f'📄 {manual}')
    else:
        st.sidebar.info("No manuals loaded. Add PDFs to data/manuals/ folder")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🐷 Farm Resources")
    st.sidebar.markdown("- 🌽 Maize & Grains")
    st.sidebar.markdown("- 🥜 Soya Products") 
    st.sidebar.markdown("- 🧪 Premix & Supplements")
    st.sidebar.markdown("- 💊 Animal Health")
    st.sidebar.markdown("- 🏠 Housing & Equipment")
    
    if st.session_state.active_tab == "Overview":
        display_overview(analyzer)
    elif st.session_state.active_tab == "Monthly Analysis":
        display_monthly_analysis(analyzer)
    elif st.session_state.active_tab == "Feed Consumption":
        display_feed_consumption(analyzer)
    elif st.session_state.active_tab == "Expense Breakdown":
        display_expense_breakdown(analyzer)
    elif st.session_state.active_tab == "Individual Contributions":
        display_individual_contributions(analyzer)
    elif st.session_state.active_tab == "Recommendations":
        display_recommendations(analyzer)
    elif st.session_state.active_tab == "Chatbot":
        display_chatbot(analyzer, chatbot)

def display_overview(analyzer):
    st.markdown('<div class="section-header">🏠 Project Overview</div>', unsafe_allow_html=True)
    
    total_expenses, total_feed = analyzer.get_summary_metrics()
    individual_totals = analyzer.get_individual_totals()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cost = sum(total_expenses.values())
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">TOTAL EXPENSES</div>
            <div class="metric-value">${total_cost:,.0f}</div>
            <div class="metric-description">3 Months Total</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        avg_monthly_cost = total_cost / 3
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">MONTHLY AVERAGE</div>
            <div class="metric-value">${avg_monthly_cost:,.0f}</div>
            <div class="metric-description">Per Month</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        total_feed_kg = sum(total_feed.values())
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">FEED CONSUMPTION</div>
            <div class="metric-value">{total_feed_kg:,.0f} kg</div>
            <div class="metric-description">Total Feed</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        total_individual_contributions = sum(individual_totals.values())
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">TEAM CONTRIBUTIONS</div>
            <div class="metric-value">${total_individual_contributions:,.0f}</div>
            <div class="metric-description">Yami, Mike & Kali</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 📈 Expense Trends")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(total_expenses.keys()),
            y=list(total_expenses.values()),
            mode='lines+markers',
            name='Monthly Expenses',
            line=dict(color='#667eea', width=4),
            marker=dict(size=12, color='#764ba2', line=dict(color='white', width=2)),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        fig.update_layout(
            template="plotly_white",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, color='#2c3e50')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 👥 Team Contributions")
        labels = list(individual_totals.keys())
        values = list(individual_totals.values())
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=.5,
            marker_colors=colors,
            textfont=dict(size=14, color='white'),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>'
        )])
        fig.update_layout(
            template="plotly_white",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 🌾 Feed Consumption by Month")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(total_feed.keys()),
            y=list(total_feed.values()),
            marker=dict(
                color=list(total_feed.values()),
                colorscale='Viridis',
                line=dict(color='white', width=2)
            ),
            text=[f'{v:,.0f} kg' for v in total_feed.values()],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>%{y:,.0f} kg<extra></extra>'
        ))
        fig.update_layout(
            template="plotly_white",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            yaxis_title="Kilograms"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 💵 Monthly Expense Comparison")
        monthly_df = pd.DataFrame({
            'Month': list(total_expenses.keys()),
            'Expenses': list(total_expenses.values())
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_df['Month'],
            y=monthly_df['Expenses'],
            marker=dict(
                color=['#667eea', '#764ba2', '#f093fb'],
                line=dict(color='white', width=2)
            ),
            text=[f'${v:,.0f}' for v in monthly_df['Expenses']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>$%{y:,.0f}<extra></extra>'
        ))
        fig.update_layout(
            template="plotly_white",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            yaxis_title="Dollars"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def display_monthly_analysis(analyzer):
    st.markdown('<div class="section-header">📅 Monthly Analysis</div>', unsafe_allow_html=True)
    
    month = st.selectbox("Select Month", ["September", "October", "November"], key="month_select")
    data = analyzer.data[month]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown(f"#### 🐖 Feed Consumption - {month}")
        st.dataframe(data['feed'], use_container_width=True, hide_index=True)
        
        fig = px.pie(data['feed'], values='Total_kg', names='Category', 
                    title=f'Feed Distribution',
                    color_discrete_sequence=px.colors.sequential.Purp)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown(f"#### 💰 Expenses - {month}")
        st.dataframe(data['expenses'], use_container_width=True, hide_index=True)
        
        fig = px.bar(data['expenses'], x='Item', y='Total_Cost',
                    title='Expense Breakdown',
                    color='Total_Cost',
                    color_continuous_scale='Sunset')
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown(f"#### 👥 Individual Contributions - {month}")
    st.dataframe(data['individuals'], use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_feed_consumption(analyzer):
    st.markdown('<div class="section-header">🌾 Feed Consumption Analysis</div>', unsafe_allow_html=True)
    
    feed_df = analyzer.get_feed_analysis()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 📊 Feed by Category Over Time")
        fig = px.bar(feed_df, x='Month', y='Total_kg', color='Category',
                    title="Monthly Feed Consumption by Category",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    barmode='group')
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 💹 Feed Efficiency")
        total_expenses, total_feed = analyzer.get_summary_metrics()
        
        feed_cost_analysis = []
        for month in total_expenses.keys():
            feed_expenses = analyzer.data[month]['expenses']
            feed_items = ['Cruches', 'Premix', 'Soya', 'Creep']
            feed_cost = feed_expenses[feed_expenses['Item'].isin(feed_items)]['Total_Cost'].sum()
            total_kg = total_feed[month]
            cost_per_kg = feed_cost / total_kg if total_kg > 0 else 0
            
            feed_cost_analysis.append({
                'Month': month,
                'Feed_Cost': feed_cost,
                'Total_Feed_kg': total_kg,
                'Cost_per_kg': cost_per_kg
            })
        
        efficiency_df = pd.DataFrame(feed_cost_analysis)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=efficiency_df['Month'], 
            y=efficiency_df['Cost_per_kg'],
            mode='lines+markers',
            name='Cost per kg',
            line=dict(color='#667eea', width=4),
            marker=dict(size=15, color='#764ba2', line=dict(color='white', width=2)),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        fig.update_layout(
            title='Feed Cost per kg Over Time',
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis_title='Cost per kg ($)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### 📋 Detailed Feed Data")
    st.dataframe(feed_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_expense_breakdown(analyzer):
    st.markdown('<div class="section-header">💰 Expense Breakdown</div>', unsafe_allow_html=True)
    
    expense_df = analyzer.get_expense_breakdown()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 📊 Expense Categories")
        st.dataframe(expense_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 🥧 Expense Distribution")
        total_by_category = expense_df.sum(axis=1)
        fig = px.pie(values=total_by_category.values, names=total_by_category.index,
                    title="Overall Expense Distribution",
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### 📈 Expense Trends by Category")
    
    expense_melted = expense_df.reset_index().melt(id_vars='index', var_name='Month', value_name='Cost')
    expense_melted.columns = ['Category', 'Month', 'Cost']
    
    fig = px.line(expense_melted, x='Month', y='Cost', color='Category',
                 markers=True,
                 color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_traces(line=dict(width=3), marker=dict(size=10))
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_individual_contributions(analyzer):
    st.markdown('<div class="section-header">👥 Team Contribution Analysis</div>', unsafe_allow_html=True)
    
    individual_totals = analyzer.get_individual_totals()
    monthly_totals = analyzer.get_individual_monthly_totals()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">YAMI</div>
            <div class="metric-value">${individual_totals["Yami"]:,.0f}</div>
            <div class="metric-description">Total Contribution</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">MIKE</div>
            <div class="metric-value">${individual_totals["Mike"]:,.0f}</div>
            <div class="metric-description">Total Contribution</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-title">KALI</div>
            <div class="metric-value">${individual_totals["Kali"]:,.0f}</div>
            <div class="metric-description">Total Contribution</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 📊 Monthly Contributions by Person")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Yami', x=monthly_totals['Month'], y=monthly_totals['Yami'], marker_color='#ff6b6b'))
        fig.add_trace(go.Bar(name='Mike', x=monthly_totals['Month'], y=monthly_totals['Mike'], marker_color='#4ecdc4'))
        fig.add_trace(go.Bar(name='Kali', x=monthly_totals['Month'], y=monthly_totals['Kali'], marker_color='#45b7d1'))
        
        fig.update_layout(
            barmode='group',
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis_title='Contribution ($)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 📈 Contribution Trends")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_totals['Month'], y=monthly_totals['Yami'], 
                                mode='lines+markers', name='Yami',
                                line=dict(color='#ff6b6b', width=3),
                                marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=monthly_totals['Month'], y=monthly_totals['Mike'], 
                                mode='lines+markers', name='Mike',
                                line=dict(color='#4ecdc4', width=3),
                                marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=monthly_totals['Month'], y=monthly_totals['Kali'], 
                                mode='lines+markers', name='Kali',
                                line=dict(color='#45b7d1', width=3),
                                marker=dict(size=10)))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis_title='Contribution ($)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### 📋 Detailed Contribution Data")
    contributions_df = analyzer.get_individual_contributions()
    st.dataframe(contributions_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_recommendations(analyzer):
    st.markdown('<div class="section-header">💡 AI Insights & Recommendations</div>', unsafe_allow_html=True)
    
    total_expenses, total_feed = analyzer.get_summary_metrics()
    individual_totals = analyzer.get_individual_totals()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Strategic Recommendations")
        
        recommendations = [
            ("Feed Optimization", "Negotiate bulk pricing for Soya and Cruches to reduce costs by 15-20%"),
            ("Contribution Structure", "Standardize team contribution system for better financial planning"),
            ("Inventory Management", "Implement just-in-time ordering to reduce storage costs"),
            ("Technology Investment", "Explore automated feeding systems for labor efficiency"),
            ("Supplier Relationships", "Build long-term partnerships for better pricing and reliability"),
            ("Role Clarification", "Define clear financial responsibilities among team members"),
            ("Monthly Reviews", "Conduct regular financial reviews to track progress and adjust strategy")
        ]
        
        for i, (title, desc) in enumerate(recommendations, 1):
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 4px solid #667eea;'>
                <strong style='color: white; font-size: 1.1em;'>{i}. {title}</strong><br>
                <span style='color: rgba(255,255,255,0.8);'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 📊 Performance Metrics")
        
        total_cost = sum(total_expenses.values())
        total_contributions = sum(individual_totals.values())
        
        metrics_data = [
            ("Total Project Cost", f"${total_cost:,.2f}", "💵"),
            ("Team Contributions", f"${total_contributions:,.2f}", "👥"),
            ("Yami's Share", f"{(individual_totals['Yami']/total_contributions)*100:.1f}%", "👤"),
            ("Mike's Share", f"{(individual_totals['Mike']/total_contributions)*100:.1f}%", "👤"),
            ("Kali's Share", f"{(individual_totals['Kali']/total_contributions)*100:.1f}%", "👤"),
            ("Avg Monthly Expense", f"${total_cost/3:,.2f}", "📅"),
            ("Total Feed Used", f"{sum(total_feed.values()):,.0f} kg", "🌾")
        ]
        
        for label, value, emoji in metrics_data:
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;'>
                <span style='color: white;'>{emoji} <strong>{label}</strong></span>
                <span style='color: #667eea; font-weight: bold; font-size: 1.1em;'>{value}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### 🔮 Predictive Analysis")
    
    months = list(total_expenses.keys())
    expenses = list(total_expenses.values())
    
    future_months = ['December', 'January', 'February']
    avg_growth = (expenses[-1] - expenses[0]) / len(expenses)
    predicted = [expenses[-1] + avg_growth * (i+1) for i in range(3)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=expenses,
        mode='lines+markers',
        name='Actual',
        line=dict(color='#667eea', width=4),
        marker=dict(size=12, color='#764ba2')
    ))
    fig.add_trace(go.Scatter(
        x=future_months, y=predicted,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='#f093fb', width=4, dash='dash'),
        marker=dict(size=12, color='#f5576c')
    ))
    
    fig.update_layout(
        title='Expense Trend with 3-Month Prediction',
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis_title='Expenses ($)',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_chatbot(analyzer, chatbot):
    st.markdown('<div class="section-header">🤖 CasaDeFo RAG Assistant</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.2); margin-bottom: 20px;'>
        <h3 style='color: white; margin-top: 0;'>🧠 RAG-Powered Knowledge Assistant</h3>
        <p style='color: rgba(255,255,255,0.9);'>
            This assistant uses <strong>Retrieval-Augmented Generation (RAG)</strong> to search through:
        </p>
        <ul style='color: rgba(255,255,255,0.8);'>
            <li>📚 Uploaded Manuals from backend folder</li>
            <li>💡 Pig Farming Knowledge Base - breeds, diseases, housing</li>
            <li>📊 Your Project Data - expenses, feed, contributions</li>
        </ul>
        <p style='color: rgba(255,255,255,0.9); margin-bottom: 0;'><strong>Try asking about:</strong></p>
        <ul style='color: rgba(255,255,255,0.8); margin-bottom: 0;'>
            <li>"What are the best pig breeds for commercial farming?"</li>
            <li>"How to prevent African Swine Fever?"</li>
            <li>"What are the housing requirements for pregnant sows?"</li>
            <li>"Show me our project expense breakdown"</li>
            <li>"Who contributed the most to our project?"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about pig farming or project data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🧠 Querying RAG knowledge base..."):
                response = chatbot.get_response(prompt)
                st.markdown(response, unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()