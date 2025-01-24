import os
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import csv
from groq import Groq
import google.generativeai as genai

class QuizPerformanceVisualizer:
    def __init__(self, performance_data, latest_submission,quiz_data,gemini_api_key=None,openai_api_key=None):
        self.data = performance_data
        self.gemini_client = None
        self.quiz_data = quiz_data
        self.latest_submission=latest_submission
        
        # Initialize Gemini client if API key is provided
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_client = genai.GenerativeModel('gemini-pro')
                # openai.api_key = os.getenv('OPENAI_API_KEY')
            except Exception as e:
                print(f"Error initializing Gemini client: {e}")
    
    def plot_latest_submission_performance(self):
        """
        Visualize the performance of the latest quiz submission
        """
        plt.figure(figsize=(10, 6))
        metrics = [
            'Score', 
            'Accuracy', 
            'Correct Answers', 
            'Total Questions', 
            'Speed'
        ]
        values = [
            self.latest_submission.get('score', 0),
            float(self.latest_submission.get('accuracy', '0%').rstrip('%')),
            self.latest_submission.get('correct_answers', 0),
            self.latest_submission.get('total_questions', 0),
            float(self.latest_submission.get('speed', '0'))
        ]
        
        plt.bar(metrics, values)
        plt.title('Latest Quiz Submission Performance')
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('latest_submission_performance.png')
        plt.close()
        
        # Interactive Plotly visualization
        fig = px.bar(
            x=metrics, 
            y=values, 
            title='Interactive Latest Submission Performance',
            labels={'x':'Metrics', 'y':'Value'}
        )
        fig.write_html('latest_submission_performance_interactive.html')
        
        # Print detailed latest submission information
        # print("\nLatest Quiz Submission Details:")
        # for metric, value in zip(metrics, values):
        #     print(f"{metric}: {value}")
    
    def plot_historical_performance(self):
        """
        Visualize historical performance trends
        """
        # Accuracy trend
        accuracy_data = [
            float(quiz.get('accuracy', '0%').rstrip('%'))
            for quiz in self.data.get('historical_data', [])
        ]
        
        # plt.figure(figsize=(10, 5))
        # plt.plot(accuracy_data, marker='o')
        # plt.title('Historical Accuracy Trend')
        # plt.xlabel('Quiz Attempt')
        # plt.ylabel('Accuracy (%)')
        # plt.tight_layout()
        # plt.savefig('historical_accuracy_trend.png')
        # plt.close()
        
        # Performance by difficulty
        difficulty_levels = {}
        for quiz in self.data.get('historical_data', []):
            difficulty = quiz.get('difficulty', 'Unknown')
            score = quiz.get('score', 0)
            
            if difficulty not in difficulty_levels:
                difficulty_levels[difficulty] = {'scores': [], 'attempts': 0}
            
            difficulty_levels[difficulty]['scores'].append(score)
            difficulty_levels[difficulty]['attempts'] += 1
        
        # Plotting difficulty performance
        difficulties = list(difficulty_levels.keys())
        avg_scores = [np.mean(difficulty_levels[diff]['scores']) for diff in difficulties]
        attempts = [difficulty_levels[diff]['attempts'] for diff in difficulties]
        
        fig = go.Figure(data=[
            go.Bar(name='Average Score', x=difficulties, y=avg_scores),
            go.Bar(name='Number of Attempts', x=difficulties, y=attempts, yaxis='y2')
        ])
        fig.update_layout(
            title='Historical Performance by Difficulty',
            xaxis_title='Difficulty Level',
            yaxis_title='Average Score',
            yaxis2=dict(
                title='Number of Attempts',
                overlaying='y',
                side='right'
            )
        )
        fig.write_html('historical_difficulty_performance.html')

        
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
    
        # Extract difficulty levels and 
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
        Create a comprehensive performance summary visualization with latest submission comparison
        """
        # Prepare topic data from historical performance
        topics = list(self.data['topic_performance'].keys())
        historical_accuracies = [
            self.data['topic_performance'][topic]['average_accuracy'] 
            for topic in topics
        ]
        
        # Extract latest submission topic and performance
        latest_topic = self.latest_submission.get('quiz', {}).get('topic', 'Unknown')
        latest_accuracy = float(self.latest_submission.get('accuracy', '0%').rstrip('%'))
        
        # Prepare comparison data
        comparison_accuracies = historical_accuracies.copy()
        
        # If latest topic is in historical topics, update its accuracy
        if latest_topic in topics:
            topic_index = topics.index(latest_topic)
            comparison_accuracies[topic_index] = (
                historical_accuracies[topic_index] + latest_accuracy
            ) / 2
        else:
            # Add latest topic if not in historical data
            topics.append(latest_topic)
            comparison_accuracies.append(latest_accuracy)
        
        # Radar chart for topic strengths with latest submission
        fig = go.Figure(data=go.Scatterpolar(
            r=comparison_accuracies,
            theta=topics,
            fill='toself'
        ))
        fig.update_layout(
            title='Topic Performance Radar Chart',
            polar=dict(
                radialaxis=dict(
                    visible=True, 
                    range=[0, 100]
                )
            )
        )
        fig.write_html('topic_performance_comparison.html')
        
        # Create a detailed comparison DataFrame
        comparison_df = pd.DataFrame({
            'Topic': topics,
            'Historical Avg Accuracy': historical_accuracies + [0] * (len(topics) - len(historical_accuracies)),
            'Latest Submission Accuracy': 
                [latest_accuracy if topic == latest_topic else 0 for topic in topics]
        })
        
        # Save comparison to CSV
        comparison_df.to_csv('topic_performance_comparison.csv', index=False)
        
        # Optional: Create a Plotly bar chart for comparison
        fig_bar = go.Figure(data=[
            go.Bar(name='Historical Average', x=topics, y=historical_accuracies),
            go.Bar(name='Latest Submission', x=topics, y=
                [latest_accuracy if topic == latest_topic else 0 for topic in topics]
            )
        ])
        fig_bar.update_layout(
            title='Topic Performance: Historical vs Latest Submission',
            xaxis_title='Topics',
            yaxis_title='Accuracy (%)'
        )
        fig_bar.write_html('topic_performance_comparison_bar.html')
        
        return comparison_df
    
    def generate_ai_insights(self):
        """
        Generate AI-powered insights using Gemini, comparing latest submission with historical data
        """
        if not self.gemini_client:
            print("Gemini client not initialized. Skipping AI insights.")
            return None
        
        # Prepare latest submission details
        latest_topic = self.latest_submission.get('quiz', {}).get('topic', 'Unknown')
        latest_score = self.latest_submission.get('score', 0)
        latest_accuracy = self.latest_submission.get('accuracy', 'N/A')
        
        # Historical performance for the latest topic
        historical_topic_data = self.data['topic_performance'].get(latest_topic, {})
        historical_avg_score = historical_topic_data.get('average_score', 'N/A')
        historical_avg_accuracy = historical_topic_data.get('average_accuracy', 'N/A')
        
        # Prepare insights prompt
        insights_prompt = f"""
        Comprehensive Performance Analysis:

        OVERALL PERFORMANCE:
        - Average Historical Score: {self.data.get('average_score', 'N/A')}
        - Average Historical Accuracy: {self.data.get('average_accuracy', 'N/A')}

        LATEST SUBMISSION FOCUS:
        Topic: {latest_topic}
        - Latest Submission Score: {latest_score}
        - Latest Submission Accuracy: {latest_accuracy}

        HISTORICAL TOPIC COMPARISON:
        - Historical Average Score for {latest_topic}: {historical_avg_score}
        - Historical Average Accuracy for {latest_topic}: {historical_avg_accuracy}

         
        Please provide all the belowe based on historical data:
        {json.dumps(self.data['topic_performance'], indent=2)}
        1. Detailed analysis of strengths and weaknesses
        2. Personalized learning recommendations
        3. Strategies for improvement
        4. A student persona based on the performance data


        Provide clear, actionable, student-friendly insights highlighting:
        - Specific strengths demonstrated
        - Areas needing improvement
        - Contextual performance evaluation
        - Precise learning path forward

        Maintain an encouraging, motivational tone that empowers learning.
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
    def find_wrong_answers(self):
        """
        Find wrong answers by comparing submitted responses with correct options in the quiz.
        Args:
            latest_submission (dict): The latest quiz submission data.
            quiz_data (dict): The quiz data containing questions and correct answers.
        Returns:
            list: A list of dictionaries containing details of incorrect answers.
        """
        wrong_answers = []
        
        # Extract the response map and questions
        response_map = self.latest_submission.get("response_map", {})
        questions = self.quiz_data.get("quiz", {}).get("questions", [])
        
        for question_id, submitted_option_id in response_map.items():
            # Find the question by ID
            question = next((q for q in questions if str(q["id"]) == str(question_id)), None)
            
            if question:
                # Find the correct option
                correct_option = next((opt for opt in question.get("options", []) if opt.get("is_correct")), None)
                
                # Find the submitted option
                submitted_option = next((opt for opt in question.get("options", []) if str(opt["id"]) == str(submitted_option_id)), None)
                
                # Compare and record if the submitted option is wrong
                if correct_option and submitted_option and correct_option["id"] != submitted_option["id"]:
                    wrong_answers.append({
                        "question_id": question_id,
                        "question_text": question.get("description", ""),
                        "submitted_option": submitted_option.get("description", ""),
                        "correct_option": correct_option.get("description", ""),
                        "topic": question.get("topic", "Unknown"),
                        "detailed_solution": question.get("detailed_solution", ""),
                    })
        self.save_wrong_answers_to_csv(wrong_answers,file_path="Wrong-Submission-Data.csv")
        # print(wrong_answers)
        return wrong_answers
    def save_wrong_answers_to_csv(self,wrong_answers,file_path):
        """
        Save the wrong answers to a CSV file.
        Args:
            wrong_answers (list): The list of dictionaries containing wrong answer details.
            file_path (str): The path to save the CSV file.
        """
        try:
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=["question_id", "question_text", "submitted_option", "correct_option", "topic", "detailed_solution"])
                writer.writeheader()
                writer.writerows(wrong_answers)
            print(f"Wrong answers saved to {file_path}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

    def generate_ai_latest(self, wrong_answers):
        """Generate AI insights for wrong answers using Groq API"""

        # Load Groq API key from environment variable inside the function
        groq_api_key = os.getenv("GROQ_API_KEY")

        

        # Initialize Groq client
        client = Groq(api_key="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")

        insights_prompt = f"""
        WRONG ANSWERS COMPREHENSIVE ANALYSIS

        OVERVIEW:
        - Total Wrong Answers: {len(wrong_answers)}
        - Quiz Topic: {self.latest_submission['quiz']['topic']}
        - Overall Accuracy: {self.latest_submission['accuracy']}

        DETAILED WRONG ANSWERS ANALYSIS:

        {json.dumps(wrong_answers, indent=2)}

        ANALYSIS FRAMEWORK FOR EACH WRONG ANSWER:

        1. CONCEPTUAL ERROR DIAGNOSIS
        - Specific knowledge gap identified
        - Root cause of misunderstanding
        - Precise explanation of correct concept

        2. LEARNING INTERVENTION
        - Targeted study recommendations
        - Recommended learning resources
        - Specific reading or practice suggestions

        3. KNOWLEDGE REINFORCEMENT
        - Key learning objectives
        - Practice questions or self-assessment strategies
        - Contextual understanding of the topic

        LEARNING APPROACH:
        - Provide clear, concise explanations
        - Focus on constructive, actionable insights
        - Emphasize understanding over correction
        - Suggest progressive learning pathways

        STUDENT GUIDANCE PRINCIPLES:
        - Motivational and encouraging tone
        - Highlight potential for improvement
        - Connect current errors to broader learning journey
        """

        try:
            # Requesting the Groq API with the new client-based syntax
            chat_completion = client.chat.completions.create(
                    #
                    # Required parameters
                    #
                    messages=[
                        # Set an optional system message. This sets the behavior of the
                        # assistant and can be used to provide specific instructions for
                        # how it should behave throughout the conversation.
                        {"role": "system", "content": "you are a helpful assistant."},
                        # Set a user message for the assistant to respond to.
                        {
                            "role": "user",
                            "content": insights_prompt,
                        },
                    ],
                    # The language model which will generate the completion.
                    model="mixtral-8x7b-32768",
                    #
                    # Optional parameters
                    #
                    # Controls randomness: lowering results in less random completions.
                    # As the temperature approaches zero, the model will become deterministic
                    # and repetitive.
                    temperature=0.5,
                    # The maximum number of tokens to generate. Requests can use up to
                    # 2048 tokens shared between prompt and completion.
                    max_tokens=1024,
                    # Controls diversity via nucleus sampling: 0.5 means half of all
                    # likelihood-weighted options are considered.
                    top_p=1,
                    # A stop sequence is a predefined or user-specified text string that
                    # signals an AI to stop generating content, ensuring its responses
                    # remain focused and concise. Examples include punctuation marks and
                    # markers like "[end]".
                    stop=None,
                    # If set, partial message deltas will be sent.
                    stream=False,
                )


            
            # Extract the response text
            ai_text = chat_completion.choices[0].message.content
            
            # Save to file
            with open('wrong_answers_detailed_insights.txt', 'w', encoding='utf-8') as f:
                f.write(ai_text)
            
            return ai_text
        except Exception as e:
            print(f"Error generating wrong answers insights: {e}")
            return "Can't perform the operation"
def main():
    # Configuration
    USER_ID = "YcDFSO4ZukTJnnFMgRNVwZTE4j42"
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')  # Get API key from environment
    
    API_ENDPOINTS = {
        'quiz_submission': 'https://api.jsonserve.com/rJvd7g',
        'quiz': 'https://www.jsonkeeper.com/b/LLQT',
        'historical_data': 'https://api.jsonserve.com/XgAgFJ'
    }
    
    # Download and load data
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
        

        
        # Initialize and run visualizer
        visualizer = QuizPerformanceVisualizer(
            performance_data, 
            latest_submission=latest_submission, 
            quiz_data=quiz_data,
            gemini_api_key=GEMINI_API_KEY
        )
        
        # Generate visualizations
        visualizer.plot_latest_submission_performance()
        visualizer.plot_historical_performance()
        visualizer.plot_topic_performance()
        visualizer.plot_accuracy_trend()
        visualizer.generate_performance_summary()
        
        # Generate AI Insights (optional, requires Gemini API key)
        wrong_ans=visualizer.find_wrong_answers()
        # Generate AI insights for wrong answers
        ai_wrong_answers_insights = visualizer.generate_ai_latest(wrong_answers=wrong_ans)
        if ai_wrong_answers_insights:
            print("Wrong Answers Insights generated and saved to wrong_answers_detailed_insights.txt")
    
        ai_insights = visualizer.generate_ai_insights()
        if ai_insights:
            print("AI Insights generated and saved to ai_performance_insights.txt")
        wrong_ans=visualizer.find_wrong_answers()
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()