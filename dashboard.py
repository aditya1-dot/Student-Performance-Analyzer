import streamlit as st
import os
import requests
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import google.generativeai as genai

class StreamlitQuizPerformanceApp:
    def __init__(self):
        # Streamlit page configuration
        st.set_page_config(page_title="Quiz Performance Dashboard", layout="wide")
        
        # API Endpoints
        self.API_ENDPOINTS = {
            'historical_data': 'https://api.jsonserve.com/XgAgFJ'
        }
        
        # Gemini API initialization
        self.gemini_client = None
        
    def init_gemini_client(self, api_key):
        """Initialize Gemini client if API key is provided"""
        try:
            genai.configure(api_key=api_key)
            self.gemini_client = genai.GenerativeModel('gemini-pro')
            return True
        except Exception as e:
            st.error(f"Error initializing Gemini client: {e}")
            return False
    
    def load_performance_data(self, user_id):
        """Load and process performance data for a specific user"""
        try:
            # Fetch historical data
            historical_response = requests.get(self.API_ENDPOINTS['historical_data'])
            historical_data = historical_response.json()
            
            # Filter data for specific user
            user_historical_data = [
                quiz for quiz in historical_data 
                if quiz.get('user_id') == user_id
            ]
            
            # Check if any data found
            if not user_historical_data:
                st.warning(f"No performance data found for User ID: {user_id}")
                return None
            
            # Prepare performance data
            performance_data = {
                'historical_data': user_historical_data,
                'average_score': np.mean([quiz.get('score', 0) for quiz in user_historical_data]),
                'average_accuracy': np.mean([
                    float(quiz.get('accuracy', '0%').rstrip('%')) 
                    for quiz in user_historical_data
                ]),
                'topic_performance': {}
            }
            
            # Analyze topic performance
            for quiz in user_historical_data:
                topic = quiz.get('quiz', {}).get('topic', 'Unknown')
                score = quiz.get('score', 0)
                accuracy = float(quiz.get('accuracy', '0%').rstrip('%'))
                
                if topic not in performance_data['topic_performance']:
                    performance_data['topic_performance'][topic] = {
                        'scores': [],
                        'accuracies': []
                    }
                
                performance_data['topic_performance'][topic]['scores'].append(score)
                performance_data['topic_performance'][topic]['accuracies'].append(accuracy)
            
            # Calculate average metrics for each topic
            for topic, data in performance_data['topic_performance'].items():
                data['average_score'] = np.mean(data['scores'])
                data['average_accuracy'] = np.mean(data['accuracies'])
            
            return performance_data
        
        except Exception as e:
            st.error(f"Error loading performance data: {e}")
            return None
    
    def plot_topic_performance(self, performance_data):
        """Create interactive topic performance plot"""
        topics = list(performance_data['topic_performance'].keys())
        avg_scores = [
            performance_data['topic_performance'][topic]['average_score'] 
            for topic in topics
        ]
        
        fig = px.bar(
            x=topics, 
            y=avg_scores, 
            title='Average Quiz Scores by Topic',
            labels={'x':'Topics', 'y':'Average Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_accuracy_trend(self, performance_data):
        """Visualize accuracy trend over time"""
        accuracy_data = [
            float(quiz.get('accuracy', '0%').rstrip('%'))
            for quiz in performance_data.get('historical_data', [])
        ]
        
        fig = px.line(
            y=accuracy_data, 
            title='Accuracy Trend Over Time',
            labels={'index':'Quiz Attempt', 'y':'Accuracy (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def generate_topic_radar_chart(self, performance_data):
        """Create radar chart for topic performance"""
        topics = list(performance_data['topic_performance'].keys())
        accuracies = [
            performance_data['topic_performance'][topic]['average_accuracy'] 
            for topic in topics
        ]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=accuracies,
            theta=topics,
            fill='toself'
        ))
        fig.update_layout(
            title='Topic Performance Radar Chart',
            polar=dict(radialaxis=dict(visible=True, range=[0, 100]))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def generate_ai_insights(self, performance_data):
        """Generate AI-powered insights using Gemini"""
        if not self.gemini_client:
            st.warning("Gemini client not initialized. AI insights unavailable.")
            return None
        
        # Prepare insights prompt
        insights_prompt = f"""
        Analyze the following quiz performance data and provide comprehensive insights:
        
        Overall Average Score: {performance_data.get('average_score', 'N/A')}
        Overall Average Accuracy: {performance_data.get('average_accuracy', 'N/A')}
        
        Topic Performance Details:
        {str(performance_data['topic_performance'])}
        
        Please provide:
        1. Detailed analysis of strengths and weaknesses
        2. Personalized learning recommendations
        3. Strategies for improvement
        4. A student persona based on the performance data
        
        Format the response in clear, actionable language suitable for a student.
        """
        
        try:
            response = self.gemini_client.generate_content(insights_prompt)
            return response.text
        except Exception as e:
            st.error(f"Error generating AI insights: {e}")
            return None
    
    def run(self):
        """Main Streamlit app"""
        st.title("Quiz Performance Dashboard")
        
        # Sidebar for configuration
        st.sidebar.header("Dashboard Configuration")
        
        # Gemini API Key input
        gemini_api_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password")
        
        # Initialize Gemini client if API key provided
        if gemini_api_key:
            self.init_gemini_client(gemini_api_key)
        
        # Load performance data
        if st.sidebar.button("Load Performance Data"):
            with st.spinner("Loading performance data..."):
                performance_data = self.load_performance_data("YcDFSO4ZukTJnnFMgRNVwZTE4j42")
                
                if performance_data:
                    # Performance Overview
                    st.header("Performance Overview")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Score", f"{performance_data['average_score']:.2f}")
                    with col2:
                        st.metric("Average Accuracy", f"{performance_data['average_accuracy']:.2f}%")
                    
                    # Visualization Tabs
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Topic Performance", 
                        "Accuracy Trend", 
                        "Topic Radar Chart", 
                        "Gemini AI Insights"
                    ])
                    
                    with tab1:
                        self.plot_topic_performance(performance_data)
                    
                    with tab2:
                        self.plot_accuracy_trend(performance_data)
                    
                    with tab3:
                        self.generate_topic_radar_chart(performance_data)
                    
                    with tab4:
                        st.header("AI-Powered Performance Insights")
                        if self.gemini_client:
                            with st.spinner("Generating AI insights..."):
                                ai_insights = self.generate_ai_insights(performance_data)
                                if ai_insights:
                                    st.write(ai_insights)
                        else:
                            st.info("Please provide a Gemini API key to generate AI insights.")

def main():
    app = StreamlitQuizPerformanceApp()
    app.run()

if __name__ == "__main__":
    main()