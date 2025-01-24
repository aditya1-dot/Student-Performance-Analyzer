import streamlit as st
import os
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from main import QuizPerformanceVisualizer
import requests

def main():
    st.set_page_config(page_title="Quiz Performance Dashboard", layout="wide")
    st.title("Quiz Performance Dashboard")

    # Configuration
    USER_ID = "YcDFSO4ZukTJnnFMgRNVwZTE4j42"
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')  # Get API key from environment
    
    API_ENDPOINTS = {
        'quiz_submission': 'https://api.jsonserve.com/rJvd7g',
        'quiz': 'https://www.jsonkeeper.com/b/LLQT',
        'historical_data': 'https://api.jsonserve.com/XgAgFJ'
    }

    # Sidebar for configuration
    st.sidebar.header("Dashboard Configuration")
    
    # Optional API Key inputs
    gemini_api_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password", value=GEMINI_API_KEY or "")

    # Load Performance Data Button
    if st.sidebar.button("Load Performance Data"):
        try:
            # Fetch quiz data, submission data, and historical data
            historical_response = requests.get(API_ENDPOINTS['historical_data'])
            historical_data = historical_response.json()
            
            quiz_submission_response = requests.get(API_ENDPOINTS['quiz_submission'])
            latest_submission = quiz_submission_response.json()
            
            quiz_response = requests.get(API_ENDPOINTS['quiz'])
            quiz_data = quiz_response.json()
            
            # Filter data for specific user
            user_historical_data = [
                quiz for quiz in historical_data 
                if quiz.get('user_id') == USER_ID
            ]
            
            # Prepare performance data for visualization
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

            # Initialize Visualizer
            visualizer = QuizPerformanceVisualizer(
                performance_data, 
                latest_submission=latest_submission, 
                quiz_data=quiz_data,
                gemini_api_key=gemini_api_key
            )

            # Performance Overview
            st.header("Performance Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Score", f"{performance_data['average_score']:.2f}")
            with col2:
                st.metric("Average Accuracy", f"{performance_data['average_accuracy']:.2f}%")

            # Create Tabs for Visualizations
            # Create Tabs for Visualizations
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Latest Submission", 
                "Topic Performance", 
                "Exam Mistake Report",  
                "Accuracy Trend", 
                "Performance Summary",
                "Overall Performance Summary"  # 
            ])

            with tab1:
                st.header("Latest Submission Performance")
                # visualizer.plot_latest_submission_performance()
                # st.image('latest_submission_performance.png')
                st.components.v1.html(open('latest_submission_performance_interactive.html').read(), height=500)

            with tab2:
                st.header("Topic Performance")
                # visualizer.plot_topic_performance()
                # st.image('topic_performance_bar.png')
                st.components.v1.html(open('topic_performance_interactive.html').read(), height=500)

            with tab3:
                st.header("Exam Mistake Report")
                # Wrong Answers Analysis
                wrong_ans = visualizer.find_wrong_answers()
                st.write("Detailed Wrong Answers Analysis:")
                with open('wrong_answers_detailed_insights.txt', 'r') as f:
                    st.write(f.read())

            with tab4:
                st.header("Accuracy Trend")
                visualizer.plot_accuracy_trend()
                st.image('accuracy_trend.png')

            with tab5:
                st.header("Performance Summary")
                comparison_df = visualizer.generate_performance_summary()
                st.dataframe(comparison_df)
                st.components.v1.html(open('topic_performance_comparison.html').read(), height=500)
                st.components.v1.html(open('topic_performance_comparison_bar.html').read(), height=500)

            

            with tab6:
                st.header("Overall Performance Summary")
                # Overall AI Insights
                ai_insights = visualizer.generate_ai_insights()
                st.write("Comprehensive Performance Insights:")
                with open('ai_performance_insights.txt', 'r') as f:
                    st.write(f.read())

        except Exception as e:
            st.error(f"Error processing data: {e}")

if __name__ == "__main__":
    main()