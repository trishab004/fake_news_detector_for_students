import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import re
import nltk
from datetime import datetime
import google.generativeai as palm
import json

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="Student Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Configure Google PaLM API
# -------------------------
palm.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# -------------------------
# Download NLTK data
# -------------------------
for resource in ["punkt"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        with st.spinner(f"Downloading {resource}..."):
            nltk.download(resource)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
    .main-header { font-size:2.5rem; color:#1f77b4; text-align:center; margin-bottom:1rem; }
    .result-box { padding:15px; border-radius:10px; margin:10px 0; border-left:5px solid; }
    .reliable { border-left-color:#28a745; background-color:#d4edda; }
    .fake { border-left-color:#dc3545; background-color:#f8d7da; }
    .borderline { border-left-color:#ffc107; background-color:#fff3cd; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# FakeNewsDetector Class
# -------------------------
class FakeNewsDetector:
    def __init__(self):
        self.loaded = True

    def analyze_text(self, text):
        """Analyze text for fake news indicators"""
        if len(text) < 20:
            return {'error': 'Text too short for analysis'}
        
        summary = self.generate_summary(text)
        analysis = self.model_based_analysis(text)
        features = self.extract_features(text)
        
        return {
            'summary': summary,
            'analysis': analysis,
            'features': features,
            'word_count': len(text.split()),
            'char_count': len(text)
        }

def model_based_analysis(self, text):
    """Use Google PaLM to classify Fake/Real"""
    try:
        prompt = f"""
Classify the following news article as either 'Fake' or 'Reliable'. 
Provide the label and confidence as JSON if possible. Article:
{text[:1000]}
"""
        response = palm.generate_text(model="chat-bison-001", prompt=prompt)
        result_text = response.result.strip()

        # Try to parse JSON
        try:
            result = json.loads(result_text)
            label = result.get("label", "").lower()
            score = float(result.get("confidence", 0))
        except:
            # Fallback: parse from text manually
            label = "fake" if "fake" in result_text.lower() else "reliable"
            score = 0.9 if label=="fake" else 0.95  # default confidences

        if label == "fake":
            verdict = "Fake News"
            color = "red"
        else:
            verdict = "Reliable"
            color = "green"

        return {
            "verdict": verdict,
            "confidence": score,
            "color": color,
            "scores": {
                "fake_score": score if verdict=="Fake News" else 1-score,
                "reliable_score": score if verdict=="Reliable" else 1-score
            }
        }
    except Exception as e:
        return {
            "verdict": "Error",
            "confidence": 0,
            "color": "orange",
            "scores": {"fake_score": 0, "reliable_score": 0},
            "error": str(e)
        }

    def extract_article_from_url(self, url):
        """Extract article content from URL"""
        try:
            headers = {'User-Agent':'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title = soup.find('title')
            title_text = title.get_text() if title else "No title found"
            
            content = ""
            article = soup.find('article')
            if article:
                content = article.get_text()
            else:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs])
            content = re.sub(r'\s+', ' ', content).strip()
            
            return {'title': title_text, 'content': content, 'success': len(content)>100}
        except Exception as e:
            return {'title': 'Error', 'content':'', 'success': False, 'error': str(e)}

    def generate_summary(self, text):
        """Extractive summary"""
        if len(text) < 100: return "Text too short for meaningful summary"
        try:
            sentences = nltk.sent_tokenize(text)
            return ' '.join(sentences[:3]) if len(sentences) > 3 else ' '.join(sentences)
        except:
            return text[:200] + "..."

    def extract_features(self, text):
        words = nltk.word_tokenize(text.lower())
        sentences = nltk.sent_tokenize(text)
        sensational_words = ['shocking','miracle','secret','breaking','urgent']
        sensational_count = sum(1 for word in words if any(sw in word for sw in sensational_words))
        reliable_indicators = ['according to','study shows','research indicates','experts say']
        reliable_count = sum(1 for word in words if any(rw in word for rw in reliable_indicators))
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words)/len(sentences) if sentences else 0,
            'exclamation_count': text.count('!'),
            'sensational_word_count': sensational_count,
            'reliable_indicator_count': reliable_count
        }

# -------------------------
# Streamlit pages
# -------------------------
def main():
    if 'detector' not in st.session_state:
        st.session_state.detector = FakeNewsDetector()
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

    with st.sidebar:
        st.title("🎓 Student Fake News Detector")
        st.markdown("---")
        page = st.radio("Choose a page:", ["🏠 Home", "🔍 Analyze Article", "📊 History", "🎓 Learn"])
        st.markdown("---")
        st.metric("Analyses", len(st.session_state.analysis_history))
    
    if page=="🏠 Home": render_home_page()
    elif page=="🔍 Analyze Article": render_analysis_page()
    elif page=="📊 History": render_history_page()
    elif page=="🎓 Learn": render_learn_page()

# -------------------------
# Other page render functions
# -------------------------
def render_home_page():
    st.markdown('<h1 class="main-header">🔍 Student Fake News Detector</h1>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
        ## 🛡️ Your Defense Against Misinformation
        This tool helps students:
        - **🔍 Analyze** news articles
        - **🎯 Assess** credibility
        - **📝 Summarize**
        """)
    with col2:
        st.markdown("<div style='text-align:center;font-size:120px;'>🔍</div>", unsafe_allow_html=True)

def render_analysis_page():
    st.title("🔍 Analyze Article")
    input_method = st.radio("Choose input:", ["📝 Paste Text", "🌐 Enter URL"], horizontal=True)
    if input_method=="📝 Paste Text":
        input_content = st.text_area("Paste text here", height=200)
        analyze_btn = st.button("🚀 Analyze Text")
    else:
        url = st.text_input("Enter URL")
        analyze_btn = st.button("🌐 Analyze URL")
        input_content = url
    if analyze_btn and input_content:
        with st.spinner("Analyzing..."):
            if input_method=="🌐 Enter URL":
                result = st.session_state.detector.extract_article_from_url(input_content)
                if not result['success']:
                    st.error("Failed to extract content")
                    return
                article_title = result['title']
                article_content = result['content']
            else:
                article_title = "Pasted Text Analysis"
                article_content = input_content
            analysis_result = st.session_state.detector.analyze_text(article_content)
            if 'error' in analysis_result:
                st.error(analysis_result['error'])
                return
            full_result = {'timestamp': datetime.now().isoformat(),'title':article_title,**analysis_result}
            st.session_state.analysis_history.append(full_result)
            display_results(full_result)

def display_results(result):
    st.success("✅ Analysis Complete!")
    col1,col2 = st.columns([2,1])
    with col1:
        st.subheader("📖 Article Info")
        st.write(f"**Title:** {result.get('title','N/A')}")
        st.write(f"**Words:** {result.get('word_count',0)}")
        st.subheader("📋 Summary")
        st.info(result['summary'])
    with col2:
        st.subheader("🎯 Credibility")
        analysis = result['analysis']
        color_class = {"red":"fake","green":"reliable","orange":"borderline"}.get(analysis['color'],'borderline')
        st.markdown(f"""<div class="result-box {color_class}"><h3>{analysis['verdict']}</h3><p>Confidence: {analysis['confidence']:.1%}</p></div>""", unsafe_allow_html=True)
        scores = analysis['scores']
        fig = px.bar(x=list(scores.keys()), y=list(scores.values()), color=list(scores.keys()), color_discrete_map={'fake_score':'red','reliable_score':'green'})
        st.plotly_chart(fig, use_container_width=True)

def render_history_page():
    st.title("📊 Analysis History")
    if not st.session_state.analysis_history:
        st.info("No analyses yet")
        return
    for i, analysis in enumerate(st.session_state.analysis_history):
        with st.expander(f"Analysis {i+1}: {analysis.get('title','Unknown')}"):
            st.write(f"Verdict: {analysis['analysis']['verdict']}")
            st.write(f"Confidence: {st.session_state.analysis_history[i]['analysis']['confidence']:.1%}")
            st.write(f"Date: {pd.Timestamp(analysis['timestamp']).strftime('%Y-%m-%d %H:%M')}")

def render_learn_page():
    st.title("🎓 Learn About Fake News")
    st.markdown("""
    ## 🔍 How to Spot Fake News
    **🚨 Red Flags:** Excessive capitalization!!! Emotional language, secret claims.
    **✅ Reliability Signs:** Clear sources, multiple citations.
    """)

if __name__=="__main__":
    main()
