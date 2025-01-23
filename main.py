import os
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

# Add Gemini API support
import google.generativeai as genai

class QuizPerformanceVisualizer:
    def __init__(self, performance_data, gemini_api_key=None):
        self.data = performance_data
        self.gemini_client = None
        
        # Initialize Gemini client if API key is provided
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_client = genai.GenerativeModel('gemini-pro')
            except Exception as e:
                print(f"Error initializing Gemini client: {e}")
        
    def plot_topic_performance(self):
        """
        Create visualizations of topic performance
        """
        # Bar chart of average scores by topic
        topics = list(self.data['topic_performance'].keys())
        avg_scores = [
            self.data['topic_performance'][topic]['average_score'] 
            for topic in topics
        ]
        
        # Matplotlib static visualization
        plt.figure(figsize=(12, 6))
        plt.bar(topics, avg_scores)
        plt.title('Average Quiz Scores by Topic')
        plt.xlabel('Topics')
        plt.ylabel('Average Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('topic_performance_bar.png')
        plt.close()
        
        # Interactive Plotly visualization
        fig = px.bar(
            x=topics, 
            y=avg_scores, 
            title='Interactive Topic Performance',
            labels={'x':'Topics', 'y':'Average Score'}
        )
        fig.write_html('topic_performance_interactive.html')
        
    def plot_accuracy_trend(self):
        """
        Visualize accuracy trend over time
        """
        # Assuming historical data is available in performance metrics
        accuracy_data = [
            float(quiz.get('accuracy', '0%').rstrip('%'))
            for quiz in self.data.get('historical_data', [])
        ]
        
        plt.figure(figsize=(10, 5))
        plt.plot(accuracy_data, marker='o')
        plt.title('Accuracy Trend')
        plt.xlabel('Quiz Attempt')
        plt.ylabel('Accuracy (%)')
        plt.savefig('accuracy_trend.png')
        plt.close()
        
    def generate_performance_summary(self):
        """
        Create a comprehensive performance summary visualization
        """
        # Radar chart for topic strengths
        topics = list(self.data['topic_performance'].keys())
        accuracies = [
            self.data['topic_performance'][topic]['average_accuracy'] 
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
        fig.write_html('topic_radar_chart.html')
    
    def generate_ai_insights(self):
        """
        Generate AI-powered insights using Gemini
        """
        if not self.gemini_client:
            print("Gemini client not initialized. Skipping AI insights.")
            return None
        
        # Prepare insights prompt
        insights_prompt = f"""
        Analyze the following quiz performance data and provide comprehensive insights:
        
        Overall Average Score: {self.data.get('average_score', 'N/A')}
        Overall Average Accuracy: {self.data.get('average_accuracy', 'N/A')}
        
        Topic Performance Details:
        {json.dumps(self.data['topic_performance'], indent=2)}
        
        Please provide:
        1. Detailed analysis of strengths and weaknesses
        2. Personalized learning recommendations
        3. Strategies for improvement
        4. A student persona based on the performance data
        
        Format the response in clear, actionable language suitable for a student.
        """
        
        try:
            response = self.gemini_client.generate_content(insights_prompt)
            
            # Save insights to a file
            with open('ai_performance_insights.txt', 'w') as f:
                f.write(response.text)
            
            return response.text
        except Exception as e:
            print(f"Error generating AI insights: {e}")
            return None

def main():
    # Configuration
    USER_ID = "YcDFSO4ZukTJnnFMgRNVwZTE4j42"
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')  # Get API key from environment
    
    API_ENDPOINTS = {
        'quiz_submission': 'https://api.jsonserve.com/rJvd7g',
        'quiz': 'https://jsonkeeper.com/b/LLQT',
        'historical_data': 'https://api.jsonserve.com/XgAgFJ'
    }
    
    # Download and load historical data
    try:
        # Fetch historical data
        historical_response = requests.get(API_ENDPOINTS['historical_data'])
        historical_data = historical_response.json()
        
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
        
        # Initialize and run visualizer
        visualizer = QuizPerformanceVisualizer(performance_data, gemini_api_key=GEMINI_API_KEY)
        
        # Generate visualizations
        visualizer.plot_topic_performance()
        visualizer.plot_accuracy_trend()
        visualizer.generate_performance_summary()
        
        # Generate AI Insights (optional, requires Gemini API key)
        ai_insights = visualizer.generate_ai_insights()
        if ai_insights:
            print("AI Insights generated and saved to ai_performance_insights.txt")
    
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()