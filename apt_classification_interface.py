#!/usr/bin/env python3
"""
APT Classification Interface - Version Améliorée
Interface élégante pour la classification des groupes APT
Focus sur les top 5 prédictions avec définitions complètes
"""

import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import re
from pathlib import Path
import io
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    """Structured classification result"""
    predicted_class: str
    confidence: float
    top5_probabilities: Dict[str, float]
    processing_time: float
    extracted_features: Dict[str, List[str]]
    attribution_factors: List[str]
    timestamp: str

class CySecBERTMaxPerformance(nn.Module):
    """CySecBERT optimized for maximum performance (95%+ F1)"""
    
    def __init__(
        self, 
        model_name: str = "markusbayer/CySecBERT",
        num_classes: int = 12,
        max_length: int = 384,
        dropout_rate: float = 0.15
    ):
        super(CySecBERTMaxPerformance, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        
        # CySecBERT specialized for cybersecurity
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # EXPANDED architecture for maximum capacity
        self.dropout = nn.Dropout(dropout_rate)
        self.intermediate1 = nn.Linear(self.config.hidden_size, 512)
        self.intermediate_dropout1 = nn.Dropout(dropout_rate * 0.6)
        self.intermediate2 = nn.Linear(512, 256)
        self.intermediate_dropout2 = nn.Dropout(dropout_rate * 0.7)
        
        # Batch normalization for stability
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)
        
        self.classifier = nn.Linear(256, num_classes)
        
        # Optimized activations
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # [CLS] token with minimal dropout
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)
        
        # First LARGE intermediate layer
        intermediate1 = self.gelu(self.intermediate1(cls_output))
        intermediate1 = self.intermediate_dropout1(intermediate1)
        
        if intermediate1.size(0) > 1:
            intermediate1 = self.batch_norm1(intermediate1)
        
        # Second intermediate layer
        intermediate2 = self.relu(self.intermediate2(intermediate1))
        intermediate2 = self.intermediate_dropout2(intermediate2)
        
        if intermediate2.size(0) > 1:
            intermediate2 = self.batch_norm2(intermediate2)
        
        # Final classification
        logits = self.classifier(intermediate2)
        
        return {
            'logits': logits,
            'probabilities': torch.softmax(logits, dim=-1)
        }

class APTClassifier:
    """Production-ready APT classifier with comprehensive profiles"""
    
    def __init__(self, model_path: str = "best_cysecbert_max_performance.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.label_encoder = None
        
        # Comprehensive APT group profiles with MITRE-style information
        self.apt_profiles = {
            'APT1': {
                'country': 'China',
                'flag': '🇨🇳',
                'aliases': ['Comment Crew', 'Comment Group', 'PLA Unit 61398', 'Shanghai Group'],
                'description': 'Chinese cyber espionage group attributed to the People\'s Liberation Army Unit 61398. Known for large-scale intellectual property theft and targeting of over 140 organizations across 20 industries.',
                'first_observed': '2006',
                'attribution_confidence': 'High',
                'sponsor': 'State-sponsored (PLA Unit 61398)',
                'malware': ['WEBC2', 'BACKDOOR.BARKIOFORK', 'AURIGA', 'BANGAT', 'BISCUIT'],
                'tools': ['HTRAN', 'GSECDUMP', 'GETMAIL', 'MAPIGET'],
                'targets': ['Intellectual property', 'Government agencies', 'Industrial companies', 'Legal services', 'IT companies'],
                'sectors': ['Information Technology', 'Energy', 'Financial Services', 'Government', 'Healthcare'],
                'regions': ['United States', 'Canada', 'United Kingdom', 'India'],
                'ttps': ['T1566.001', 'T1059.003', 'T1071.001', 'T1083', 'T1005'],
                'mitre_groups': ['G0006'],
                'notable_campaigns': ['Operation Aurora (2009)', 'RSA SecurID breach (2011)', 'Elderwood campaigns'],
                'motivations': ['Espionage', 'Intellectual property theft'],
                'sophistication': 'Medium to High'
            },
            'APT28': {
                'country': 'Russia',
                'flag': '🇷🇺',
                'aliases': ['Fancy Bear', 'Sofacy', 'Sednit', 'STRONTIUM', 'Pawn Storm', 'Swallowtail'],
                'description': 'Russian military intelligence cyber operations unit attributed to GRU Unit 26165. Highly sophisticated group known for targeting government, military, and security organizations worldwide.',
                'first_observed': '2007',
                'attribution_confidence': 'High',
                'sponsor': 'State-sponsored (GRU Unit 26165)',
                'malware': ['X-Agent', 'Sofacy', 'GAMEFISH', 'Zebrocy', 'CHOPSTICK', 'EVILTOSS'],
                'tools': ['Responder', 'Mimikatz', 'Compiled HTML Help', 'PowerShell Empire'],
                'targets': ['Government agencies', 'Military organizations', 'Defense contractors', 'Aerospace', 'Media'],
                'sectors': ['Government', 'Defense', 'Aerospace', 'Media', 'Think Tanks'],
                'regions': ['United States', 'Europe', 'Asia-Pacific', 'Middle East'],
                'ttps': ['T1566.001', 'T1059.001', 'T1055', 'T1027', 'T1083', 'T1203'],
                'mitre_groups': ['G0007'],
                'notable_campaigns': ['DNC hack (2016)', 'Olympic Destroyer (2018)', 'UEFI rootkit campaigns'],
                'motivations': ['Espionage', 'Political influence', 'Military intelligence'],
                'sophistication': 'Very High'
            },
            'APT29': {
                'country': 'Russia',
                'flag': '🇷🇺',
                'aliases': ['Cozy Bear', 'The Dukes', 'NOBELIUM', 'Midnight Blizzard', 'UNC2452'],
                'description': 'Russian foreign intelligence service (SVR) cyber unit. Extremely sophisticated group known for stealth, persistence, and advanced techniques in espionage operations.',
                'first_observed': '2008',
                'attribution_confidence': 'High',
                'sponsor': 'State-sponsored (SVR)',
                'malware': ['HAMMERTOSS', 'COZYCAR', 'SeaDuke', 'SUNBURST', 'TEARDROP', 'BEACON'],
                'tools': ['PowerShell', 'WMI', 'Cobalt Strike', 'AdFind', 'BloodHound'],
                'targets': ['Government agencies', 'Think tanks', 'Healthcare organizations', 'Technology companies'],
                'sectors': ['Government', 'Healthcare', 'Technology', 'Research', 'NGOs'],
                'regions': ['United States', 'Europe', 'Global'],
                'ttps': ['T1566.002', 'T1071.001', 'T1055', 'T1027', 'T1078', 'T1490'],
                'mitre_groups': ['G0016'],
                'notable_campaigns': ['SolarWinds supply chain attack (2020)', 'COVID-19 research targeting', 'Azure/M365 attacks'],
                'motivations': ['Espionage', 'Intelligence gathering', 'Political influence'],
                'sophistication': 'Very High'
            },
            'Lazarus': {
                'country': 'North Korea',
                'flag': '🇰🇵',
                'aliases': ['Lazarus Group', 'Hidden Cobra', 'ZINC', 'TEMP.Hermit', 'Labyrinth Chollima'],
                'description': 'North Korean state-sponsored hacking group known for financially motivated attacks, cryptocurrency theft, and destructive operations. Connected to RGB (Reconnaissance General Bureau).',
                'first_observed': '2009',
                'attribution_confidence': 'High',
                'sponsor': 'State-sponsored (RGB)',
                'malware': ['WannaCry', 'HOPLIGHT', 'TYPEFRAME', 'BADCALL', 'FALLCHILL', 'ELECTRICFISH'],
                'tools': ['PowerShell', 'Mimikatz', 'PsExec', 'Living-off-the-land binaries'],
                'targets': ['Financial institutions', 'Cryptocurrency exchanges', 'Entertainment companies', 'Defense contractors'],
                'sectors': ['Financial Services', 'Entertainment', 'Cryptocurrency', 'Defense', 'Healthcare'],
                'regions': ['Global', 'South Korea', 'United States', 'Europe'],
                'ttps': ['T1566.001', 'T1059.003', 'T1055', 'T1027', 'T1486', 'T1490'],
                'mitre_groups': ['G0032'],
                'notable_campaigns': ['Sony Pictures attack (2014)', 'WannaCry ransomware (2017)', 'SWIFT banking attacks'],
                'motivations': ['Financial gain', 'Espionage', 'Destruction', 'Sanctions evasion'],
                'sophistication': 'High'
            },
            'Equation': {
                'country': 'United States (suspected)',
                'flag': '🇺🇸',
                'aliases': ['Equation Group', 'EQGRP', 'Tilded Team'],
                'description': 'Highly sophisticated cyber espionage group suspected to be linked to the NSA. Known for advanced persistent threats, zero-day exploits, and firmware-level implants.',
                'first_observed': '2001',
                'attribution_confidence': 'Medium',
                'sponsor': 'State-sponsored (suspected NSA)',
                'malware': ['DOUBLEFANTASY', 'EQUATIONDRUG', 'GRAYFISH', 'FANNY', 'STUXNET'],
                'tools': ['EternalBlue', 'EternalRomance', 'DoublePulsar', 'FuzzBunch'],
                'targets': ['High-value targets', 'Government agencies', 'Telecommunications', 'Research institutions'],
                'sectors': ['Government', 'Telecommunications', 'Research', 'Technology', 'Energy'],
                'regions': ['Middle East', 'Asia', 'Europe', 'Global'],
                'ttps': ['T1055', 'T1027', 'T1083', 'T1068', 'T1542.009', 'T1014'],
                'mitre_groups': ['G0020'],
                'notable_campaigns': ['Operation Equation (2008-2015)', 'STUXNET collaboration', 'Flame malware'],
                'motivations': ['Espionage', 'Intelligence gathering', 'Sabotage'],
                'sophistication': 'Extremely High'
            },
            'Carbanak': {
                'country': 'International',
                'flag': '🌍',
                'aliases': ['FIN7', 'Carbanak Group', 'Anunak', 'Carbon Spider'],
                'description': 'Financially motivated cybercriminal organization responsible for stealing over $1 billion from financial institutions worldwide through ATM and point-of-sale attacks.',
                'first_observed': '2013',
                'attribution_confidence': 'High',
                'sponsor': 'Cybercriminal',
                'malware': ['Carbanak', 'CARBANAK', 'HALFBAKED', 'BABYMETAL', 'GRIFFON'],
                'tools': ['Cobalt Strike', 'Mimikatz', 'PowerShell Empire', 'Metasploit'],
                'targets': ['Financial institutions', 'Banks', 'Payment processors', 'Hospitality', 'Retail'],
                'sectors': ['Financial Services', 'Hospitality', 'Retail', 'Restaurant'],
                'regions': ['Global', 'United States', 'Europe', 'Asia'],
                'ttps': ['T1566.001', 'T1059.003', 'T1055', 'T1027', 'T1021.001', 'T1083'],
                'mitre_groups': ['G0008', 'G0046'],
                'notable_campaigns': ['Carbanak banking attacks', 'FIN7 point-of-sale attacks', 'Restaurant POS campaigns'],
                'motivations': ['Financial gain'],
                'sophistication': 'High'
            },
            'APT40': {
                'country': 'China',
                'flag': '🇨🇳',
                'aliases': ['Leviathan', 'TEMP.Periscope', 'TEMP.Jumper', 'Kryptonite Panda'],
                'description': 'Chinese state-sponsored cyber espionage group focused on maritime industries, engineering companies, and research organizations to support China\'s Belt and Road Initiative.',
                'first_observed': '2013',
                'attribution_confidence': 'High',
                'sponsor': 'State-sponsored (MSS Hainan)',
                'malware': ['BADFLICK', 'PHOTO', 'HOMEFRY', 'MURKYTOP', 'LUNCHMONEY'],
                'tools': ['China Chopper', 'Mimikatz', 'PowerShell', 'WMI'],
                'targets': ['Maritime industries', 'Engineering companies', 'Research organizations', 'Government agencies'],
                'sectors': ['Maritime', 'Engineering', 'Research', 'Government', 'Healthcare'],
                'regions': ['United States', 'Europe', 'Asia-Pacific'],
                'ttps': ['T1566.001', 'T1190', 'T1059.003', 'T1055', 'T1027'],
                'mitre_groups': ['G0065'],
                'notable_campaigns': ['Maritime industry targeting', 'COVID-19 research theft', 'Belt and Road surveillance'],
                'motivations': ['Espionage', 'Economic advantage', 'Strategic intelligence'],
                'sophistication': 'High'
            }
        }
        
        # Cybersecurity indicators for feature extraction
        self.security_indicators = {
            'malware': r'\b(trojan|virus|worm|ransomware|backdoor|rootkit|spyware|adware|botnet|rat|loader)\b',
            'techniques': r'\bT\d{4}(\.\d{3})?\b',
            'domains': r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            'ips': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'hashes': r'\b[a-fA-F0-9]{32,64}\b',
            'cve': r'\bCVE-\d{4}-\d{4,}\b',
            'tools': r'\b(cobalt strike|metasploit|mimikatz|powershell|psexec|wmi|bloodhound)\b'
        }
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            self.model = CySecBERTMaxPerformance(
                num_classes=checkpoint.get('num_classes', 12),
                dropout_rate=checkpoint.get('config', {}).get('dropout_rate', 0.15)
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.class_names = checkpoint.get('class_names', [])
            self.label_encoder = checkpoint.get('label_encoder')
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def extract_features(self, text: str) -> Dict[str, List[str]]:
        """Extract cybersecurity features from text"""
        features = {}
        text_lower = text.lower()
        
        for feature_type, pattern in self.security_indicators.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            features[feature_type] = list(set(matches))[:10]  # Limit to 10 items
        
        return features
    
    def get_attribution_factors(self, text: str, predicted_class: str) -> List[str]:
        """Get attribution factors for the prediction"""
        factors = []
        text_lower = text.lower()
        
        if predicted_class in self.apt_profiles:
            profile = self.apt_profiles[predicted_class]
            
            # Check for group mentions
            if predicted_class.lower() in text_lower:
                factors.append(f"Direct mention of {predicted_class}")
            
            # Check for aliases
            for alias in profile.get('aliases', []):
                if alias.lower() in text_lower:
                    factors.append(f"Alias detected: {alias}")
            
            # Check for known malware
            for malware in profile.get('malware', []):
                if malware.lower() in text_lower:
                    factors.append(f"Known malware: {malware}")
            
            # Check for tools
            for tool in profile.get('tools', []):
                if tool.lower() in text_lower:
                    factors.append(f"Known tool: {tool}")
            
            # Check for target sectors
            for target in profile.get('targets', []):
                if target.lower() in text_lower:
                    factors.append(f"Target sector match: {target}")
            
            # Check for TTPs
            for ttp in profile.get('ttps', []):
                if ttp in text:
                    factors.append(f"MITRE technique: {ttp}")
        
        return factors
    
    def classify(self, text: str, confidence_threshold: float = 0.5) -> ClassificationResult:
        """Classify a text input and return top 5 predictions"""
        start_time = time.time()
        
        # Tokenize
        encoding = self.model.tokenizer(
            text,
            max_length=self.model.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = outputs['probabilities'].cpu().numpy()[0]
        
        # Get top 5 predictions
        top5_indices = np.argsort(probabilities)[::-1][:5]
        predicted_class = self.class_names[top5_indices[0]]
        confidence = float(probabilities[top5_indices[0]])
        
        # Create top 5 probability distribution
        top5_probabilities = {
            self.class_names[idx]: float(probabilities[idx])
            for idx in top5_indices
        }
        
        # Extract features and attribution factors
        extracted_features = self.extract_features(text)
        attribution_factors = self.get_attribution_factors(text, predicted_class)
        
        processing_time = time.time() - start_time
        
        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence,
            top5_probabilities=top5_probabilities,
            processing_time=processing_time,
            extracted_features=extracted_features,
            attribution_factors=attribution_factors,
            timestamp=datetime.now().isoformat()
        )

def apply_custom_css():
    """Apply custom CSS for better UI"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        font-weight: 400;
        margin: 0.5rem 0 0 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border-left: 6px solid #667eea;
    }
    
    .prediction-title {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    
    .prediction-details {
        font-size: 1.1rem;
        color: #34495e;
        margin: 0.5rem 0;
    }
    
    .apt-profile {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .apt-profile h3 {
        margin: 0 0 1rem 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    .apt-profile-section {
        margin: 1rem 0;
        padding: 1rem;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    
    .apt-profile-item {
        margin: 0.5rem 0;
        font-size: 1rem;
        line-height: 1.4;
    }
    
    .feature-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .feature-item {
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 8px;
        border-left: 3px solid #667eea;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    
    .status-success {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
    }
    
    .top5-card {
        background: white;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .rank-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-right: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def create_main_interface():
    """Create the main Streamlit interface"""
    
    st.set_page_config(
        page_title="APT Classification System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🛡️ APT Classification System</h1>
        <p class="main-subtitle">Advanced Persistent Threat Group Classification using CySecBERT</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize classifier
    @st.cache_resource
    def load_classifier():
        return APTClassifier()
    
    try:
        classifier = load_classifier()
        st.markdown("""
        <div class="status-success">
            ✅ Model loaded successfully! CySecBERTMaxPerformance is ready for classification.
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        confidence_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Minimum confidence for predictions"
        )
        
        st.markdown("### 📊 Display Options")
        show_features = st.checkbox("🔍 Show Extracted Features", value=True)
        show_attribution = st.checkbox("🎯 Show Attribution Factors", value=True)
        show_full_profile = st.checkbox("📋 Show Complete APT Profile", value=True)
        
        st.markdown("### 📊 Model Information")
        st.info(f"""
        **Model**: CySecBERTMaxPerformance  
        **Classes**: {len(classifier.class_names)}  
        **Device**: {classifier.device}  
        **F1 Score**: 95.74%
        """)
    
    # Main tabs
    tab1, tab2 = st.tabs(["🔍 Text Analysis", "📄 File Upload"])
    
    with tab1:
        st.markdown("### 📝 Enter Incident Description")
        
        # Example selector
        example_texts = {
            "None": "",
            "APT28/Fancy Bear": """A sophisticated cyber espionage campaign attributed to APT28 (Fancy Bear) has been identified targeting government entities. The attackers utilized spear-phishing emails containing malicious attachments to deliver the X-Agent payload. The campaign demonstrated advanced persistence techniques including T1566.001 (Spearphishing Attachment) and T1055 (Process Injection). Network analysis revealed command and control communications to domains associated with previous APT28 operations. The group, also known as Sofacy, employed GAMEFISH malware and showed characteristics consistent with GRU Unit 26165 operations.""",
            "APT29/Cozy Bear": """Government agencies reported incidents involving sophisticated malware attributed to APT29 (Cozy Bear). The attackers employed advanced techniques including T1566.002 (spear-phishing via link) and utilized HAMMERTOSS for command and control. The campaign showed characteristics consistent with Russian foreign intelligence operations. NOBELIUM techniques were observed, including the use of SUNBURST and TEARDROP malware. The attacks targeted think tanks and healthcare organizations, consistent with SVR operational patterns.""",
            "Lazarus Group": """Financial institutions have been targeted by a campaign consistent with Lazarus Group operations. The attack chain involved watering hole attacks leading to the deployment of custom malware with cryptocurrency theft capabilities. TTPs observed include T1566.001 and T1059.003, consistent with North Korean threat actor methodologies. The campaign utilized HOPLIGHT and TYPEFRAME malware, with characteristics similar to previous Hidden Cobra operations attributed to RGB."""
        }
        
        selected_example = st.selectbox(
            "💡 Choose an example or write your own:",
            options=list(example_texts.keys())
        )
        
        text_input = st.text_area(
            "📄 Paste your cybersecurity incident description here:",
            value=example_texts[selected_example],
            height=200,
            placeholder="Describe the cybersecurity incident, including TTPs, malware, targets, and any attribution indicators..."
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🔍 Analyze Threat",
                type="primary",
                use_container_width=True
            )
        
        if analyze_button and text_input.strip():
            with st.spinner("🔍 Analyzing threat intelligence..."):
                result = classifier.classify(text_input, confidence_threshold)
            
            display_results(result, classifier, show_features, show_attribution, show_full_profile)
    
    with tab2:
        st.markdown("### 📄 Upload Threat Intelligence Report")
        
        uploaded_file = st.file_uploader(
            "Choose a file containing threat intelligence",
            type=['txt', 'pdf', 'docx'],
            help="Supported formats: TXT, PDF, DOCX"
        )
        
        if uploaded_file is not None:
            with st.spinner("📖 Reading file..."):
                file_content = process_uploaded_file(uploaded_file)
            
            if file_content:
                st.markdown("### 📖 File Content Preview")
                preview_length = 500
                preview_text = file_content[:preview_length]
                if len(file_content) > preview_length:
                    preview_text += "..."
                
                st.text_area(
                    "Content:",
                    value=preview_text,
                    height=150,
                    disabled=True
                )
                
                st.markdown(f"**File size:** {len(file_content):,} characters")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    analyze_file_button = st.button(
                        "🔍 Analyze File",
                        type="primary",
                        use_container_width=True
                    )
                
                if analyze_file_button:
                    with st.spinner("🔍 Analyzing file content..."):
                        result = classifier.classify(file_content, confidence_threshold)
                    
                    display_results(result, classifier, show_features, show_attribution, show_full_profile)

def display_results(result: ClassificationResult, classifier: APTClassifier, 
                   show_features: bool, show_attribution: bool, show_full_profile: bool):
    """Display classification results with top 5 predictions"""
    
    # Main prediction box
    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-title">
            🎯 Most Likely APT Group: <strong>{result.predicted_class}</strong>
        </div>
        <div class="prediction-details">
            <strong>📊 Confidence:</strong> {result.confidence:.1%}
        </div>
        <div class="prediction-details">
            <strong>⚡ Processing Time:</strong> {result.processing_time:.3f}s
        </div>
        <div class="prediction-details">
            <strong>🕒 Analysis Time:</strong> {datetime.fromisoformat(result.timestamp).strftime('%H:%M:%S UTC')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top 5 Predictions
    st.markdown("### 🏆 Top 5 Most Likely APT Groups")
    
    for i, (apt_group, probability) in enumerate(result.top5_probabilities.items(), 1):
        rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1]
        
        st.markdown(f"""
        <div class="top5-card">
            <span class="rank-badge">#{i}</span>
            <strong>{rank_emoji} {apt_group}</strong> - {probability:.1%}
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization of top 5
    fig = go.Figure(go.Bar(
        x=list(result.top5_probabilities.values()),
        y=list(result.top5_probabilities.keys()),
        orientation='h',
        marker=dict(
            color=list(result.top5_probabilities.values()),
            colorscale='Viridis',
            colorbar=dict(title="Probability")
        ),
        text=[f"{prob:.1%}" for prob in result.top5_probabilities.values()],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Top 5 APT Group Probabilities",
        xaxis_title="Probability",
        yaxis_title="APT Group",
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    

    # Complete APT Profile - VERSION STREAMLIT NATIVE
    if show_full_profile and result.predicted_class in classifier.apt_profiles:
        profile = classifier.apt_profiles[result.predicted_class]
        
        st.markdown(f"### {profile.get('flag', '🌍')} {result.predicted_class} - Complete Profile")
        
        # Créer des colonnes pour une meilleure organisation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Basic Information")
            st.write(f"**Origin:** {profile.get('country', 'Unknown')}")
            st.write(f"**First Observed:** {profile.get('first_observed', 'Unknown')}")
            st.write(f"**Attribution Confidence:** {profile.get('attribution_confidence', 'Unknown')}")
            st.write(f"**Sponsor:** {profile.get('sponsor', 'Unknown')}")
            st.write(f"**Sophistication Level:** {profile.get('sophistication', 'Unknown')}")
            
            st.subheader("🎭 Known Aliases")
            if profile.get('aliases'):
                for alias in profile.get('aliases', []):
                    st.write(f"• {alias}")
            else:
                st.write("No aliases available")
            
            st.subheader("🦠 Associated Malware")
            if profile.get('malware'):
                for malware in profile.get('malware', []):
                    st.code(malware)
            else:
                st.write("No malware information available")
            
            st.subheader("🛠️ Known Tools")
            if profile.get('tools'):
                for tool in profile.get('tools', []):
                    st.code(tool)
            else:
                st.write("No tools information available")
        
        with col2:
            st.subheader("🎯 Typical Targets")
            if profile.get('targets'):
                for target in profile.get('targets', []):
                    st.write(f"• {target}")
            else:
                st.write("No target information available")
            
            st.subheader("🏢 Target Sectors")
            if profile.get('sectors'):
                for sector in profile.get('sectors', []):
                    st.write(f"• {sector}")
            else:
                st.write("No sector information available")
            
            st.subheader("🌍 Geographic Focus")
            if profile.get('regions'):
                for region in profile.get('regions', []):
                    st.write(f"• {region}")
            else:
                st.write("No geographic information available")
            
            st.subheader("⚙️ MITRE ATT&CK TTPs")
            if profile.get('ttps'):
                ttp_cols = st.columns(3)
                for i, ttp in enumerate(profile.get('ttps', [])):
                    with ttp_cols[i % 3]:
                        st.code(ttp)
            else:
                st.write("No TTP information available")
        
        # Section pleine largeur pour la description
        st.subheader("📝 Description")
        st.write(profile.get('description', 'No description available'))
        
        # Autres informations en pleine largeur
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🔗 MITRE Groups")
            if profile.get('mitre_groups'):
                for group in profile.get('mitre_groups', []):
                    st.code(group)
            else:
                st.write("No MITRE group information available")
            
            st.subheader("💡 Primary Motivations")
            if profile.get('motivations'):
                for motivation in profile.get('motivations', []):
                    st.write(f"• {motivation}")
            else:
                st.write("No motivation information available")
        
        with col4:
            st.subheader("🚨 Notable Campaigns")
            if profile.get('notable_campaigns'):
                for campaign in profile.get('notable_campaigns', []):
                    st.write(f"• {campaign}")
            else:
                st.write("No campaign information available")
    
    # Create columns for features and attribution
    col1, col2 = st.columns(2)
    
    # Attribution factors
    if show_attribution and result.attribution_factors:
        with col1:
            st.markdown("""
            <div class="feature-box">
                <div class="feature-title">🎯 Attribution Factors</div>
            </div>
            """, unsafe_allow_html=True)
            
            for factor in result.attribution_factors:
                st.markdown(f"""
                <div class="feature-item">
                    {factor}
                </div>
                """, unsafe_allow_html=True)
    
    # Extracted features
    if show_features and any(result.extracted_features.values()):
        with col2:
            st.markdown("""
            <div class="feature-box">
                <div class="feature-title">🔍 Extracted Features</div>
            </div>
            """, unsafe_allow_html=True)
            
            for feature_type, features in result.extracted_features.items():
                if features:
                    icon_map = {
                        'malware': '🦠',
                        'techniques': '⚙️',
                        'domains': '🌐',
                        'ips': '🔢',
                        'hashes': '#️⃣',
                        'cve': '🚨',
                        'tools': '🛠️'
                    }
                    icon = icon_map.get(feature_type, '📌')
                    
                    st.markdown(f"**{icon} {feature_type.title()}:**")
                    for feature in features[:5]:
                        st.markdown(f"""
                        <div class="feature-item">
                            {feature}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if len(features) > 5:
                        st.markdown(f"*... and {len(features) - 5} more*")
    
    # Download results
    st.markdown("### 💾 Export Results")
    
    result_data = {
        'analysis_summary': {
            'predicted_apt_group': result.predicted_class,
            'confidence': result.confidence,
            'processing_time': result.processing_time,
            'timestamp': result.timestamp
        },
        'top_5_predictions': result.top5_probabilities,
        'extracted_features': result.extracted_features,
        'attribution_factors': result.attribution_factors,
        'apt_profile': classifier.apt_profiles.get(result.predicted_class, {})
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📄 Download JSON",
            data=json.dumps(result_data, indent=2),
            file_name=f"apt_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Create detailed CSV
        csv_lines = [
            f"APT Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "PREDICTION SUMMARY",
            f"Predicted APT Group,{result.predicted_class}",
            f"Confidence,{result.confidence:.1%}",
            f"Processing Time,{result.processing_time:.3f}s",
            "",
            "TOP 5 PREDICTIONS"
        ]
        
        for i, (apt, prob) in enumerate(result.top5_probabilities.items(), 1):
            csv_lines.append(f"{i}. {apt},{prob:.1%}")
        
        if result.attribution_factors:
            csv_lines.extend(["", "ATTRIBUTION FACTORS"])
            for factor in result.attribution_factors:
                csv_lines.append(f"- {factor}")
        
        st.download_button(
            label="📊 Download CSV",
            data="\n".join(csv_lines),
            file_name=f"apt_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

def process_uploaded_file(uploaded_file) -> str:
    """Process uploaded file and extract text content"""
    
    file_type = uploaded_file.type
    
    try:
        if file_type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        
        elif file_type == "application/pdf":
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
            except ImportError:
                st.error("📚 PyPDF2 not installed. Install with: `pip install PyPDF2`")
                return ""
        
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                from docx import Document
                doc = Document(uploaded_file)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except ImportError:
                st.error("📄 python-docx not installed. Install with: `pip install python-docx`")
                return ""
        
        else:
            st.error(f"❌ Unsupported file type: {file_type}")
            return ""
            
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        return ""

if __name__ == "__main__":
    create_main_interface()