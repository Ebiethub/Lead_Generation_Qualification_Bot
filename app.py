import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
import pandas as pd
import datetime
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lead_data" not in st.session_state:
    st.session_state.lead_data = {}
if "conversation_stage" not in st.session_state:
    st.session_state.conversation_stage = "greeting"
if "analytics" not in st.session_state:
    st.session_state.analytics = {
        "interactions": [],
        "conversion_rate": 0,
        "lead_scores": []
    }

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def init_groq():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-specdec",
        temperature=0.3
    )

# Qualification Criteria
QUALIFICATION_CRITERIA = {
    "budget": {"threshold": 10000, "weight": 0.4},
    "timeline": {"threshold": 30, "weight": 0.3},
    "interest_level": {"threshold": 7, "weight": 0.3}
}

def qualify_lead(lead_data):
    """Score and categorize leads"""
    score = 0
    try:
        budget = float(lead_data.get("budget", 0))
        timeline = int(lead_data.get("timeline", 999))
        interest_level = int(lead_data.get("interest_level", 0))
        
        if budget >= QUALIFICATION_CRITERIA["budget"]["threshold"]:
            score += QUALIFICATION_CRITERIA["budget"]["weight"] * 100
        
        if timeline <= QUALIFICATION_CRITERIA["timeline"]["threshold"]:
            score += QUALIFICATION_CRITERIA["timeline"]["weight"] * 100
        
        if interest_level >= QUALIFICATION_CRITERIA["interest_level"]["threshold"]:
            score += QUALIFICATION_CRITERIA["interest_level"]["weight"] * 100
        
        if score >= 80:
            return "Hot"
        elif score >= 50:
            return "Warm"
        return "Cold"
    except:
        return "Cold"

def get_next_question():
    """Dynamic question flow with progress tracking"""
    stages = {
        "greeting": {
            "question": "Hi there! 👋 Welcome to Opolla. How can we help you today?",
            "type": "greeting"
        },
        "qualify": [
            {"question": "What's your estimated budget for this project? (USD)", "field": "budget", "type": "number"},
            {"question": "What's your ideal timeline for implementation (in days)?", "field": "timeline", "type": "number"},
            {"question": "On a scale of 1-10, how urgent is this need?", "field": "interest_level", "type": "scale"}
        ],
        "contact": [
            {"question": "What's your full name?", "field": "name", "type": "text"},
            {"question": "What's your email address?", "field": "email", "type": "email"},
            {"question": "What's the best phone number to reach you?", "field": "phone", "type": "phone"}
        ],
        "schedule": {
            "question": "Would you like to schedule a call with our expert?",
            "type": "schedule"
        }
    }

    current_stage = st.session_state.conversation_stage
    
    try:
        if current_stage == "greeting":
            return stages["greeting"]
            
        if current_stage == "qualify":
            for q in stages["qualify"]:
                if q["field"] not in st.session_state.lead_data:
                    return q
            st.session_state.conversation_stage = "contact"
            return get_next_question()
            
        if current_stage == "contact":
            for q in stages["contact"]:
                if q["field"] not in st.session_state.lead_data:
                    return q
            st.session_state.conversation_stage = "schedule"
            return stages["schedule"]
            
        if current_stage == "schedule":
            return stages["schedule"]
            
    except Exception as e:
        st.error(f"Error in conversation flow: {str(e)}")
    
    return {"question": "How can I help you today?", "type": "fallback"}

def validate_response(response, field_type):
    """Enhanced validation with contextual messages"""
    validation_rules = {
        "email": {
            "pattern": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
            "message": "Please enter a valid email address (e.g., name@company.com)"
        },
        "phone": {
            "pattern": r"^\+?[1-9]\d{1,14}$",
            "message": "Please enter a valid phone number with country code (e.g., +1234567890)"
        },
        "number": {
            "pattern": r"^\d+$",
            "message": "Please enter a valid number"
        },
        "scale": {
            "pattern": r"^([1-9]|10)$",
            "message": "Please enter a number between 1-10"
        }
    }
    
    if field_type in validation_rules:
        if not re.match(validation_rules[field_type]["pattern"], response):
            st.error(validation_rules[field_type]["message"])
            return False
    return True

def track_analytics(event_type, metadata=None):
    """Track user interactions in session state"""
    st.session_state.analytics["interactions"].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "event_type": event_type,
        "metadata": metadata or {}
    })

def show_progress():
    """Visual progress indicator"""
    progress_stages = {
        "greeting": 0,
        "qualify": 25,
        "contact": 50,
        "schedule": 75,
        "completed": 100
    }
    
    current_progress = progress_stages.get(st.session_state.conversation_stage, 0)
    
    st.markdown(f"""
    <div style="margin: 20px 0; padding: 10px; border-radius: 5px; background: #f0f2f6;">
        <div style="width: {current_progress}%; 
                  height: 10px; 
                  background: #2196F3;
                  border-radius: 5px; 
                  transition: width 0.5s ease;">
        </div>
        <p style="text-align: center; margin: 5px 0;">Progress: {current_progress}%</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Lead Gen Bot", page_icon="🤖", layout="wide")
    st.title("AI-Powered Lead Generation Assistant")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        show_progress()
        
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Main conversation handler
        if st.session_state.conversation_stage != "completed":
            next_q = get_next_question()
            
            if isinstance(next_q, dict) and "type" in next_q:
                with st.chat_message("assistant"):
                    if next_q["type"] == "greeting":
                        st.markdown(next_q["question"])
                        st.session_state.conversation_stage = "qualify"
                        st.rerun()
                        
                    elif next_q["type"] == "schedule":
                        st.markdown(next_q["question"])
                        col1, col2 = st.columns([1,2])
                        with col1:
                            if st.button("✅ Yes, Schedule Now"):
                                st.session_state.lead_data["scheduled"] = True
                                track_analytics("meeting_scheduled")
                                st.rerun()
                        with col2:
                            if st.button("❌ No, Later"):
                                st.session_state.lead_data["scheduled"] = False
                                track_analytics("meeting_declined")
                                st.rerun()
                    
                    elif "field" in next_q:
                        input_key = f"input_{next_q['field']}"
                        user_input = st.text_input(
                            label=next_q["question"],
                            key=input_key,
                            help="Press Enter to submit"
                        )
                        if user_input:
                            if validate_response(user_input, next_q["type"]):
                                st.session_state.lead_data[next_q["field"]] = user_input
                                track_analytics("question_answered", {
                                    "field": next_q["field"],
                                    "value": user_input
                                })
                                st.rerun()

        # Final conversion handling
        required_fields = ["name", "email", "phone", "budget", "timeline", "interest_level"]
        if all(field in st.session_state.lead_data for field in required_fields):
            st.session_state.conversation_stage = "completed"
            category = qualify_lead(st.session_state.lead_data)
            st.session_state.analytics["lead_scores"].append({
                **st.session_state.lead_data,
                "category": category,
                "timestamp": datetime.datetime.now()
            })
            track_analytics("conversion")
            
            with st.chat_message("assistant"):
                st.success(f"""
                    **Thank you {st.session_state.lead_data['name']}!** 🎉
                    - We've sent details to {st.session_state.lead_data['email']}
                    - Our team will call {st.session_state.lead_data['phone']}
                    - [Schedule follow-up](https://calendly.com)
                """)
    
    with col2:
        # Analytics and controls
        if st.button("🔄 Reset Conversation"):
            st.session_state.clear()
            st.rerun()
            
        st.download_button(
            "💾 Export Leads CSV",
            pd.DataFrame(st.session_state.analytics["lead_scores"]).to_csv(),
            "leads.csv",
            "text/csv",
            disabled=not st.session_state.analytics["lead_scores"]
        )
        
        # Analytics display
        st.subheader("Real-time Analytics")
        if st.session_state.analytics["lead_scores"]:
            df = pd.DataFrame(st.session_state.analytics["lead_scores"])
            st.metric("Total Leads", len(df))
            st.metric("Hot Leads", len(df[df["category"] == "Hot"]))
            st.write("Lead Distribution:")
            st.bar_chart(df["category"].value_counts())
        else:
            st.info("No leads collected yet")

if __name__ == "__main__":
    main()