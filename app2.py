import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sqlite3
import hashlib
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime, timedelta
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import requests
import json
from flask import Flask, request, jsonify
import threading
from queue import Queue as ThreadQueue
import uuid
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Smart Branch - Customer Service",
    page_icon="🎫",
    layout="wide"
)

# Load environment variables
def load_config():
    """Load configuration with A2A communication settings - FIXED for app2.py"""
    config = {
        'CAPACITY_MAX': int(os.getenv('CAPACITY_MAX', '200')),
        'QUEUE_SERVICE_RATE_SECONDS': int(os.getenv('QUEUE_SERVICE_RATE_SECONDS', '300')),
        'SMTP_HOST': os.getenv('SMTP_HOST', ''),
        'SMTP_PORT': int(os.getenv('SMTP_PORT', '465')),
        'SMTP_USER': os.getenv('SMTP_USER', ''),
        'SMTP_PASS': os.getenv('SMTP_PASS', ''),
        'FROM_EMAIL': os.getenv('FROM_EMAIL', ''),
        'TWILIO_ACCOUNT_SID': os.getenv('TWILIO_ACCOUNT_SID', ''),
        'TWILIO_AUTH_TOKEN': os.getenv('TWILIO_AUTH_TOKEN', ''),
        'TWILIO_PHONE_NUMBER': os.getenv('TWILIO_PHONE_NUMBER', ''),
        'ENABLE_NOTIFICATIONS': os.getenv('ENABLE_NOTIFICATIONS', 'true').lower() == 'true',
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY', ''),
        
        # FIXED A2A Communication settings - app2.py should connect to app1.py  
        'APP1_ENDPOINT': os.getenv('APP1_ENDPOINT', 'http://localhost:5001'),
        'CUSTOMER_AGENT_PORT': int(os.getenv('CUSTOMER_AGENT_PORT', '5002')),
        'ENABLE_A2A_COMMUNICATION': os.getenv('ENABLE_A2A_COMMUNICATION', 'true').lower() == 'true'
    }
    
    return config

# [Keep all the existing classes: SmartBranchRAG, EnhancedA2ACommunication, EnhancedCustomerServiceAgent - unchanged]
class SmartBranchRAG:
    def __init__(self, api_key):
        """Initialize the RAG system with Google Gemini AI"""
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # Use the current model name - try different options
                try:
                    self.model = genai.GenerativeModel('gemini-1.5-flash')
                except:
                    try:
                        self.model = genai.GenerativeModel('gemini-1.5-pro')
                    except:
                        try:
                            self.model = genai.GenerativeModel('gemini-pro')
                        except:
                            st.warning("Could not find a compatible Gemini model. Please check your API key and model availability.")
                            self.model = None
            except Exception as e:
                st.warning(f"Could not initialize AI model: {e}")
                self.model = None
        else:
            self.model = None
        
        self.knowledge_base = self.load_knowledge_base()
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.doc_vectors = None
        self.chunks = []
        
        if self.knowledge_base:
            self.prepare_knowledge_base()
    
    def load_knowledge_base(self):
        """Load the Smart Branch project documentation"""
        try:
            # Try to read from the uploaded file
            if os.path.exists("Smart Branch System - Complete Proj.txt"):
                with open("Smart Branch System - Complete Proj.txt", 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # Fallback to embedded knowledge
                return self.get_embedded_knowledge()
        except Exception as e:
            st.error(f"Error loading knowledge base: {e}")
            return self.get_embedded_knowledge()
    
    def get_embedded_knowledge(self):
        """Embedded knowledge about Smart Branch system"""
        return """
        Smart Branch System - Customer FAQ

        What is Smart Branch?
        Smart Branch is a comprehensive queue management and footfall analytics system designed for branch offices, banks, or service centers. It provides real-time queue management, customer analytics, and automated notifications.

        How do I join the queue?
        1. Login to your account using username/password or face recognition
        2. Go to the Queue Status tab
        3. Click "Join Queue" button
        4. You'll receive your position and estimated wait time
        5. You'll get notifications when it's almost your turn

        How does the queue system work?
        - Each customer gets a position in the queue based on arrival time
        - Wait time is estimated at 5 minutes per person ahead of you
        - You can track your position in real-time
        - Notifications are sent when you're next in line
        - You can leave the queue at any time

        What authentication methods are available?
        1. Username and Password: Traditional login method
        2. Face Recognition: Biometric authentication using your photo
        3. You can register with both methods for convenience

        How do notifications work?
        - Email notifications for queue updates
        - SMS notifications (if configured)
        - Real-time status updates in the app
        - Notifications when you're next in line
        - Service completion confirmations

        What is the feedback system?
        - Rate your experience from 1-5 stars
        - Provide written comments
        - View your previous feedback history
        - Receive email receipts for your visits
        - Download PDF receipts

        How accurate is the wait time estimation?
        Wait times are estimated based on:
        - Your position in queue
        - Average service time (5 minutes per person)
        - Current queue length
        - Real-time updates as people are served

        Can I leave the queue once I join?
        Yes, you can leave the queue at any time by clicking the "Leave Queue" button. Your status will be updated immediately and you won't receive further notifications.

        What happens if I miss my turn?
        If you don't respond when called, staff can mark you as "no show" and you'll need to rejoin the queue if you still need service.

        How do I update my profile information?
        Currently, profile updates need to be done by contacting branch staff. Future versions will include self-service profile management.

        Is my face data secure?
        Yes, face recognition uses a simple hash of facial features for identification. No actual face images are stored permanently in the system.

        What browsers are supported?
        The system works best with modern browsers that support camera access:
        - Chrome (recommended)
        - Firefox
        - Safari
        - Edge

        Technical Requirements:
        - Camera access for face recognition
        - Internet connection for real-time updates
        - Modern web browser
        - Email address for notifications (optional)

        Branch Hours and Capacity:
        - Maximum capacity varies by branch (typically 200 people)
        - Staff ratios are optimized based on occupancy
        - Real-time occupancy tracking
        - Peak hours analysis available

        For technical support or account issues, please contact branch staff.
        """
    
    def prepare_knowledge_base(self):
        """Prepare the knowledge base for vector search"""
        try:
            # Split knowledge base into chunks
            self.chunks = self.split_text(self.knowledge_base)
            
            # Create TF-IDF vectors only if we have chunks
            if self.chunks and len(self.chunks) > 0:
                self.doc_vectors = self.vectorizer.fit_transform(self.chunks)
            else:
                self.doc_vectors = None
        except Exception as e:
            st.error(f"Error preparing knowledge base: {e}")
            self.doc_vectors = None
            self.chunks = []
    
    def split_text(self, text, chunk_size=500):
        """Split text into manageable chunks"""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        
        for paragraph in paragraphs:
            if len(paragraph) < chunk_size:
                chunks.append(paragraph.strip())
            else:
                # Split long paragraphs by sentences
                sentences = paragraph.split('. ')
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) < chunk_size:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def find_relevant_chunks(self, query, top_k=3):
        """Find most relevant chunks using TF-IDF similarity"""
        # Fixed: Proper check for sparse matrix and chunks
        if self.doc_vectors is None or len(self.chunks) == 0:
            return []
        
        try:
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # Get top k indices
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # Return relevant chunks with scores
            relevant_chunks = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    relevant_chunks.append({
                        'text': self.chunks[idx],
                        'score': similarities[idx]
                    })
            
            return relevant_chunks
            
        except Exception as e:
            st.error(f"Error finding relevant chunks: {e}")
            return []
    
    def generate_answer(self, question, context_chunks):
        """Generate answer using Gemini AI with context"""
        if not self.model:
            return "AI service is not configured. Please check your API key."
        
        # Prepare context
        context = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        # Create prompt
        prompt = f"""
        You are a helpful customer service assistant for Smart Branch system. 
        Answer the customer's question based on the provided context about Smart Branch.
        
        Context:
        {context}
        
        Customer Question: {question}
        
        Instructions:
        - Provide a helpful, accurate answer based on the context
        - If the question is not fully covered in the context, say so and provide what information you can
        - Be friendly and professional
        - Keep the answer concise but complete
        - If the question is about technical issues, suggest contacting branch staff
        
        Answer:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I apologize, but I'm having trouble generating a response right now. Error: {str(e)}"
    
    def get_answer(self, question):
        """Main method to get answer for a question"""
        if not question.strip():
            return "Please ask a question about Smart Branch system."
        
        # Find relevant context
        relevant_chunks = self.find_relevant_chunks(question)
        
        if not relevant_chunks:
            return """I don't have specific information about that topic in my knowledge base. 

For general questions, you can ask about:
- How to join the queue
- Queue wait times and notifications
- Login and registration process
- Feedback system
- Technical requirements

For specific account issues or technical support, please contact branch staff."""
        
        # Generate answer using AI
        if self.model:
            return self.generate_answer(question, relevant_chunks)
        else:
            # Fallback: return relevant chunks directly
            answer = "Based on the available information:\n\n"
            for chunk in relevant_chunks[:2]:  # Limit to top 2 chunks
                answer += chunk['text'] + "\n\n"
            return answer.strip()

class EnhancedA2ACommunication:
    """Enhanced Agent-to-Agent Communication Manager for app2.py"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.app1_endpoint = config.get('APP1_ENDPOINT', 'http://localhost:5001')
        self.communication_port = config.get('CUSTOMER_AGENT_PORT', 5002)
        self.is_running = False
        self.message_queue = ThreadQueue()

        self.flask_app = None
        self.flask_thread = None
        self.last_heartbeat = None
        self.connection_status = "disconnected"
        self.retry_count = 0
        self.max_retries = 3
        self.logs = []
        
    def add_log(self, message: str):
        """Add timestamped log message"""
        log_entry = f"{datetime.now().strftime('%H:%M:%S')} - [A2A-{self.agent_id}] {message}"
        self.logs.append(log_entry)
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]
        logging.info(message)
    
    def start_communication_system(self) -> bool:
        """Start the enhanced communication system"""
        try:
            # Start Flask server for receiving messages
            if not self.initialize_flask_server():
                return False
            
            # Start message processing thread
            self.start_message_processor()
            
            # Start heartbeat monitor
            self.start_heartbeat_monitor()
            
            # Perform initial handshake
            if self.perform_handshake():
                self.is_running = True
                self.connection_status = "connected"
                self.add_log("A2A communication system started successfully")
                return True
            else:
                self.add_log("Initial handshake failed - running in degraded mode")
                self.is_running = True
                self.connection_status = "degraded"
                return True
                
        except Exception as e:
            self.add_log(f"Failed to start communication system: {str(e)}")
            return False
    
    def initialize_flask_server(self) -> bool:
        """Initialize Flask server with enhanced endpoints"""
        try:
            self.flask_app = Flask(f'customer_agent_{self.agent_id}')
            
            @self.flask_app.route('/health', methods=['GET'])
            def health_check():
                """Health check endpoint"""
                return jsonify({
                    'status': 'healthy',
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'connection_status': self.connection_status
                })
            
            @self.flask_app.route('/handshake', methods=['POST'])
            def handshake():
                """Handle handshake from app1.py agents"""
                try:
                    data = request.get_json()
                    remote_agent_id = data.get('agent_id')
                    
                    self.add_log(f"Handshake received from agent: {remote_agent_id}")
                    
                    return jsonify({
                        'status': 'success',
                        'agent_id': self.agent_id,
                        'message': 'Handshake acknowledged',
                        'timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    return jsonify({'status': 'error', 'message': str(e)}), 500
            
            @self.flask_app.route('/receive_message', methods=['POST'])
            def receive_message():
                """Enhanced message receiving endpoint - matches app1.py expectations"""
                try:
                    data = request.get_json()
                    
                    # Validate message format
                    if not self.validate_message(data):
                        return jsonify({'status': 'error', 'message': 'Invalid message format'}), 400
                    
                    # Add to message queue for processing
                    self.message_queue.put(data)
                    
                    message_type = data.get('message', {}).get('type', 'unknown')
                    from_agent = data.get('from_agent', 'unknown')
                    self.add_log(f"Message received from {from_agent}: {message_type}")
                    
                    return jsonify({
                        'status': 'success',
                        'message_id': data.get('message_id', 'unknown'),
                        'agent_id': self.agent_id,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    self.add_log(f"Error receiving message: {str(e)}")
                    return jsonify({'status': 'error', 'message': str(e)}), 500
            
            @self.flask_app.route('/heartbeat', methods=['POST'])
            def heartbeat():
                """Handle heartbeat from app1.py agents"""
                try:
                    data = request.get_json()
                    self.last_heartbeat = datetime.now()
                    
                    return jsonify({
                        'status': 'alive',
                        'agent_id': self.agent_id,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    return jsonify({'status': 'error', 'message': str(e)}), 500
            
            @self.flask_app.route('/customer_status', methods=['GET'])
            def get_customer_status():
                """Get current customer agent status"""
                return jsonify({
                    'agent_id': self.agent_id,
                    'status': 'running' if self.is_running else 'stopped',
                    'connection_status': self.connection_status,
                    'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
                    'message_queue_size': self.message_queue.qsize(),
                    'timestamp': datetime.now().isoformat()
                })
            
            # Start Flask in separate thread
            def run_flask():
                self.flask_app.run(
                    host='0.0.0.0',
                    port=self.communication_port,
                    debug=False,
                    use_reloader=False,
                    threaded=True
                )
            
            self.flask_thread = threading.Thread(target=run_flask, daemon=True)
            self.flask_thread.start()
            
            # Wait a moment for server to start
            time.sleep(2)
            self.add_log(f"Flask server started on port {self.communication_port}")
            return True
            
        except Exception as e:
            self.add_log(f"Failed to initialize Flask server: {str(e)}")
            return False
    
    def validate_message(self, message: Dict[str, Any]) -> bool:
        """Validate incoming message format - UPDATED for better compatibility"""
        # More flexible validation to handle different message structures from app1.py
        required_fields = ['from_agent', 'timestamp']
        has_required = all(field in message for field in required_fields)
        
        # Also accept messages with 'type' field directly (for direct messages)
        has_type_field = 'type' in message
        
        # Accept messages with nested message structure
        has_nested_message = 'message' in message and isinstance(message['message'], dict)
        
        return has_required and (has_type_field or has_nested_message)

    
    def perform_handshake(self) -> bool:
        """Perform initial handshake with app1.py"""
        try:
            handshake_data = {
                'agent_id': self.agent_id,
                'message': 'Initial handshake from app2.py Customer Service Agent',
                'timestamp': datetime.now().isoformat(),
                'endpoints': {
                    'health': f'http://localhost:{self.communication_port}/health',
                    'receive_message': f'http://localhost:{self.communication_port}/receive_message',
                    'heartbeat': f'http://localhost:{self.communication_port}/heartbeat'
                }
            }
            
            response = requests.post(
                f"{self.app1_endpoint}/handshake",
                json=handshake_data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.add_log("Handshake successful with app1.py")
                return True
            else:
                self.add_log(f"Handshake failed: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            self.add_log("Connection error during handshake - app1.py may not be running")
            return False
        except requests.exceptions.Timeout:
            self.add_log("Handshake timeout - app1.py not responding")
            return False
        except Exception as e:
            self.add_log(f"Handshake error: {str(e)}")
            return False
    
    def send_message_with_retry(self, endpoint: str, message: Dict[str, Any]) -> bool:
        """Send message with retry logic to app1.py - ENHANCED DEBUG"""
        
        # Add required fields only if they are missing
        message.setdefault('message_id', str(uuid.uuid4()))
        message.setdefault('from_agent', self.agent_id)
        message.setdefault('timestamp', datetime.now().isoformat())

        url = f"{self.app1_endpoint}{endpoint}"
        
        # ENHANCED DEBUG
        self.add_log(f"🚀 Attempting to send message to: {url}")
        self.add_log(f"   Message type: {message.get('type', 'unknown')}")
        self.add_log(f"   Message ID: {message.get('message_id', 'unknown')}")
        
        # Test connection first
        try:
            test_response = requests.get(f"{self.app1_endpoint}/health", timeout=2)
            if test_response.status_code != 200:
                self.add_log(f"❌ App1 health check failed: {test_response.status_code}")
                return False
            else:
                self.add_log(f"✅ App1 health check passed")
        except Exception as e:
            self.add_log(f"❌ Cannot reach app1.py at {self.app1_endpoint}/health: {str(e)}")
            return False

        for attempt in range(self.max_retries):
            try:
                self.add_log(f"📤 Sending message (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(url, json=message, timeout=5)

                self.add_log(f"📬 Response received: {response.status_code}")
                if response.text:
                    self.add_log(f"   Response body: {response.text[:200]}")

                if response.status_code == 200:
                    self.connection_status = "connected"
                    self.retry_count = 0
                    self.add_log(f"✅ Message sent successfully to {endpoint}")
                    return True
                else:
                    self.add_log(f"❌ Message failed with status {response.status_code}")

            except requests.exceptions.ConnectionError as e:
                self.add_log(f"🌐 Connection error (attempt {attempt + 1}): {str(e)}")
            except requests.exceptions.Timeout as e:
                self.add_log(f"⏱ Timeout error (attempt {attempt + 1}): {str(e)}")
            except Exception as e:
                self.add_log(f"⚠️ Send error (attempt {attempt + 1}): {str(e)}")

            # Retry with exponential backoff
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                self.add_log(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

        self.connection_status = "disconnected"
        self.retry_count += 1
        self.add_log(f"💥 All {self.max_retries} attempts failed")
        return False
        
    def send_response_to_app1(self, response_message: Dict[str, Any]) -> bool:
        """Send response message back to app1.py"""
        try:
            # Ensure message has required fields
            response_message.setdefault('message_id', str(uuid.uuid4()))
            response_message.setdefault('from_agent', self.agent_id)
            response_message.setdefault('timestamp', datetime.now().isoformat())
            
            # Send to app1.py
            response = requests.post(
                f"{self.app1_endpoint}/receive_message",
                json=response_message,
                timeout=5
            )
            
            if response.status_code == 200:
                self.add_log(f"Response sent to app1.py: {response_message.get('type', 'unknown')}")
                return True
            else:
                self.add_log(f"Failed to send response to app1.py: {response.status_code}")
                return False
                
        except Exception as e:
            self.add_log(f"Error sending response to app1.py: {str(e)}")
            return False
    
    def start_message_processor(self):
        """Start message processing thread"""
        def process_messages():
            while self.is_running:
                try:
                    # Get message from queue (blocks with timeout)
                    message = self.message_queue.get(timeout=1)
                    self.handle_incoming_message(message)
                    self.message_queue.task_done()
                except:
                    continue  # Timeout is expected
        
        processor_thread = threading.Thread(target=process_messages, daemon=True)
        processor_thread.start()
        self.add_log("Message processor started")
    
    def handle_incoming_message(self, data: Dict[str, Any]):
        """Handle incoming messages from app1.py agents - UPDATED for flexibility"""
        # Handle both direct messages and nested message structures
        if 'type' in data:
            # Direct message format
            message_type = data.get('type', 'unknown')
            message_data = data
        else:
            # Nested message format
            message = data.get('message', {})
            message_type = message.get('type', 'unknown')
            message_data = message
        
        from_agent = data.get('from_agent', 'unknown')
        
        self.add_log(f"Processing message type '{message_type}' from {from_agent}")
        
        # Handle different message types from app1.py
        if message_type == 'service_completed':
            self.handle_service_completed_message(message_data)
        elif message_type == 'individual_service_completed':
            self.handle_individual_service_completed_message(message_data)
        elif message_type == 'customer_removed':
            self.handle_customer_removed_message(message_data)
        elif message_type == 'emergency_queue_clear':
            self.handle_emergency_queue_clear_message(message_data)
        elif message_type == 'queue_positions_updated':
            self.handle_queue_positions_updated_message(message_data)
        elif message_type == 'test_connection':
            self.handle_test_connection_message(message_data)
        else:
            self.add_log(f"Unknown message type: {message_type}")
    
    def handle_service_completed_message(self, message: Dict[str, Any]):
        """Handle service completion messages from app1.py"""
        service_data = message.get('data', {})
        customer_id = service_data.get('customer_id')
        service_time = service_data.get('service_time', 5.0)
        
        self.add_log(f"Service completed for customer {customer_id} in {service_time} minutes")
        
        # Send acknowledgment back to app1.py
        response_message = {
            'type': 'service_completion_acknowledged',
            'message': f'Customer service completion acknowledged for ID {customer_id}',
            'data': {
                'customer_id': customer_id,
                'acknowledged_at': datetime.now().isoformat()
            }
        }
        return response_message
    
    def handle_individual_service_completed_message(self, message: Dict[str, Any]):
        """Handle individual service completion messages"""
        service_data = message.get('data', {})
        customer_name = service_data.get('customer_name', 'Unknown')
        queue_position = service_data.get('queue_position', 'Unknown')
        
        self.add_log(f"Individual service completed: {customer_name} (Position {queue_position})")
    
    def handle_customer_removed_message(self, message: Dict[str, Any]):
        """Handle customer removal messages"""
        removal_data = message.get('data', {})
        customer_name = removal_data.get('customer_name', 'Unknown')
        reason = removal_data.get('reason', 'unknown')
        
        self.add_log(f"Customer {customer_name} removed from queue (Reason: {reason})")
    
    def handle_emergency_queue_clear_message(self, message: Dict[str, Any]):
        """Handle emergency queue clear messages"""
        clear_data = message.get('data', {})
        cleared_customers = clear_data.get('cleared_customers', 0)
        
        self.add_log(f"Emergency queue clear: {cleared_customers} customers affected")
    
    def handle_queue_positions_updated_message(self, message: Dict[str, Any]):
        """Handle queue position update messages"""
        update_data = message.get('data', {})
        queue_length = update_data.get('total_queue_length', 0)
        
        self.add_log(f"Queue positions updated: {queue_length} customers in queue")
    
    def handle_test_connection_message(self, message: Dict[str, Any]):
        """Handle test connection messages - UPDATED to send response"""
        test_data = message.get('data', {}) if 'data' in message else message
        test_id = test_data.get('test_id', 'unknown')
        
        self.add_log(f"Test connection received: {test_id}")
        
        # Send response back to app1.py
        response_message = {
            'type': 'test_response',
            'data': {
                'test_id': test_id,
                'response_time': datetime.now().isoformat(),
                'message': 'Test connection successful from Customer Service Agent'
            }
        }
        
        # Actually send the response
        self.send_response_to_app1(response_message)
    
    def start_heartbeat_monitor(self):
        """Start heartbeat monitoring"""
        def heartbeat_monitor():
            while self.is_running:
                try:
                    # Send heartbeat to app1.py
                    heartbeat_data = {
                        'agent_id': self.agent_id,
                        'status': 'alive',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    response = requests.post(
                        f"{self.app1_endpoint}/heartbeat",
                        json=heartbeat_data,
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        if self.connection_status == "disconnected":
                            self.connection_status = "connected"
                            self.add_log("Connection restored with app1.py")
                    else:
                        if self.connection_status == "connected":
                            self.connection_status = "degraded"
                            self.add_log("Heartbeat failed - connection degraded")
                    
                except:
                    if self.connection_status in ["connected", "degraded"]:
                        self.connection_status = "disconnected"
                        self.add_log("Heartbeat failed - connection lost")
                
                time.sleep(30)  # Send heartbeat every 30 seconds
        
        heartbeat_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
        heartbeat_thread.start()
        self.add_log("Heartbeat monitor started")
    
    def check_app1_health(self) -> bool:
        """Check if app1.py is healthy"""
        try:
            response = requests.get(f"{self.app1_endpoint}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            'status': self.connection_status,
            'app1_healthy': self.check_app1_health(),
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'retry_count': self.retry_count,
            'message_queue_size': self.message_queue.qsize(),
            'agent_id': self.agent_id
        }
    
    def stop_communication_system(self):
        """Stop the communication system"""
        self.is_running = False
        self.connection_status = "stopped"
        self.add_log("A2A communication system stopped")

class EnhancedCustomerServiceAgent:
    """Enhanced Customer Service Agent with robust A2A communication"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_running = False
        self.logs = []
        self.agent_id = f"customer_agent_{uuid.uuid4().hex[:8]}"
        
        # Initialize enhanced communication system
        self.a2a_comm = EnhancedA2ACommunication(self.agent_id, config)
        
        # Customer insights
        self.customer_insights = {
            'total_customers_today': 0,
            'average_satisfaction': 0.0,
            'peak_hours': [],
            'common_questions': [],
            'service_patterns': {}
        }
        
        # Queue prediction model
        self.prediction_model = None
        self.prediction_features = ['hour_of_day', 'day_of_week', 'current_queue_length', 'historical_avg']
    
    def start_autonomous_monitoring(self) -> bool:
        """Start autonomous monitoring with enhanced communication"""
        try:
            # Start A2A communication system
            if not self.a2a_comm.start_communication_system():
                self.add_log("A2A communication failed to start - continuing in local mode")
            
            self.is_running = True
            self.add_log("Enhanced Customer Service Agent started")
            
            # Start monitoring loop
            self.start_monitoring_loop()
            
            return True
            
        except Exception as e:
            self.add_log(f"Failed to start agent: {str(e)}")
            return False
    
    def start_monitoring_loop(self):
        """Start the main monitoring loop"""
        async def monitoring_loop():
            while self.is_running:
                try:
                    await self.run_customer_analysis_cycle()
                    await asyncio.sleep(90)  # Run analysis every 90 seconds
                except Exception as e:
                    self.add_log(f"Monitoring error: {str(e)}")
                    await asyncio.sleep(30)
        
        def run_loop():
            import asyncio
            asyncio.run(monitoring_loop())
        
        monitor_thread = threading.Thread(target=run_loop, daemon=True)
        monitor_thread.start()
    
    def send_message_to_app1(self, message: Dict[str, Any]) -> bool:
        """Send message to app1.py using enhanced communication"""
        # Add required fields if missing
        message.setdefault('message_id', str(uuid.uuid4()))
        message.setdefault('from_agent', self.agent_id)
        message.setdefault('timestamp', datetime.now().isoformat())

        # Send the message directly (not wrapped inside {'message': ...})
        return self.a2a_comm.send_message_with_retry('/receive_message', message)

    
    def get_communication_status(self) -> Dict[str, Any]:
        """Get A2A communication status"""
        return self.a2a_comm.get_connection_status()
    
    def handle_customer_join(self, user_id, user_profile):
        """Handle when customer joins queue with enhanced A2A messaging"""
        self.add_log(f"Customer joined: {user_profile.get('name', 'Unknown')} (ID: {user_id})")
        
        # Update insights
        self.customer_insights['total_customers_today'] += 1
        
        # Send structured message to app1.py
        join_message = {
            'type': 'customer_joined',
            'data': {
                'customer': {
                    'user_id': user_id,
                    'name': user_profile.get('name', 'Unknown'),
                    'join_time': datetime.now().isoformat()
                }
            }
        }
        
        success = self.send_message_to_app1(join_message)
        if success:
            self.add_log(f"Customer join notification sent to app1.py")
        
        # Generate and send prediction
        prediction = self.predict_wait_time()
        prediction_message = {
            'type': 'queue_prediction',
            'data': {
                'prediction': prediction,
                'customer_id': user_id
            }
        }
        
        self.send_message_to_app1(prediction_message)
        
        return prediction
    
    def handle_customer_leave(self, user_id):
        """Handle when customer leaves queue"""
        self.add_log(f"Customer left queue: {user_id}")
        
        # Send message to app1.py
        leave_message = {
            'type': 'customer_left',
            'data': {
                'customer_id': user_id,
                'leave_time': datetime.now().isoformat()
            }
        }
        
        self.send_message_to_app1(leave_message)
    
    def handle_feedback_submission(self, user_id, rating, comments):
        """Handle customer feedback submission"""
        self.add_log(f"Feedback received: {rating}/5 stars from user {user_id}")
        
        # Update satisfaction average
        current_avg = self.customer_insights['average_satisfaction']
        total_customers = self.customer_insights['total_customers_today']
        
        if total_customers > 0:
            new_avg = ((current_avg * (total_customers - 1)) + rating) / total_customers
            self.customer_insights['average_satisfaction'] = new_avg
        else:
            self.customer_insights['average_satisfaction'] = rating
        
        # Send feedback to app1.py
        feedback_message = {
            'type': 'customer_feedback',
            'data': {
                'feedback': {
                    'user_id': user_id,
                    'rating': rating,
                    'comments': comments[:100] if comments else None,
                    'average_satisfaction': self.customer_insights['average_satisfaction']
                }
            }
        }
        
        self.send_message_to_app1(feedback_message)
    
    def predict_wait_time(self, context_data=None):
        """Predict queue wait time using ML model"""
        try:
            current_queue = get_queue_size()
            current_hour = datetime.now().hour
            current_weekday = datetime.now().weekday()
            
            # Simple prediction based on queue size and time
            base_wait = current_queue * 5  # 5 minutes per person
            
            # Adjust for peak hours
            peak_hours = [11, 12, 13, 16, 17]  # 11am-1pm, 4-5pm
            if current_hour in peak_hours:
                base_wait *= 1.3  # 30% longer during peak
            
            # Weekend adjustment
            if current_weekday >= 5:  # Weekend
                base_wait *= 1.1  # 10% longer on weekends
            
            predicted_wait = max(1, int(base_wait))  # Minimum 1 minute
            
            return {
                'predicted_wait_time': predicted_wait,
                'queue_length': current_queue,
                'factors': {
                    'hour_of_day': current_hour,
                    'is_peak_hour': current_hour in peak_hours,
                    'is_weekend': current_weekday >= 5,
                    'base_calculation': f"{current_queue} × 5 minutes"
                },
                'confidence': 'medium',
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.add_log(f"Prediction error: {str(e)}")
            return {
                'predicted_wait_time': current_queue * 5,
                'error': str(e),
                'fallback': True
            }
    
    async def run_customer_analysis_cycle(self):
        """Run customer service analysis cycle with enhanced A2A communication"""
        try:
            self.add_log("Running customer service analysis")
            
            current_queue = get_queue_size()
            current_hour = datetime.now().hour
            
            # Generate insights
            insights = {
                'current_queue_length': current_queue,
                'hour_of_day': current_hour,
                'total_customers_today': self.customer_insights['total_customers_today'],
                'average_satisfaction': self.customer_insights['average_satisfaction'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Generate recommendations
            recommendations = []
            
            if current_queue > 10:
                recommendations.append("High queue volume - consider implementing express service")
            
            if self.customer_insights['average_satisfaction'] < 3.0 and self.customer_insights['total_customers_today'] > 5:
                recommendations.append("Low satisfaction detected - review service quality")
            
            if current_hour in [11, 12, 13]:  # Lunch hours
                recommendations.append("Peak lunch hour - ensure adequate staffing")
            
            # Send enhanced analysis to app1.py
            analysis_message = {
                'type': 'service_optimization',
                'data': {
                    'analysis': insights,
                    'recommendations': recommendations
                }
            }
            
            success = self.send_message_to_app1(analysis_message)
            
            if success:
                self.add_log(f"Analysis sent to app1.py - Queue: {current_queue}, Satisfaction: {self.customer_insights['average_satisfaction']:.1f}/5")
            else:
                self.add_log(f"Failed to send analysis to app1.py")
            
        except Exception as e:
            self.add_log(f"Customer analysis failed: {str(e)}")
    
    def stop_autonomous_monitoring(self):
        """Stop monitoring and communication"""
        self.is_running = False
        self.a2a_comm.stop_communication_system()
        self.add_log("Enhanced Customer Service Agent stopped")
    
    def add_log(self, message: str):
        """Add log message"""
        log_entry = f"{datetime.now().strftime('%H:%M:%S')} - [ENHANCED_CUSTOMER_AGENT] {message}"
        self.logs.append(log_entry)
        if len(self.logs) > 50:
            self.logs = self.logs[-50:]
        logging.info(message)
    
    def get_customer_insights(self):
        """Get current customer insights with communication status"""
        current_queue = get_queue_size()
        prediction = self.predict_wait_time()
        
        insights = {
            'current_status': {
                'queue_length': current_queue,
                'predicted_wait': prediction.get('predicted_wait_time', 0),
                'service_status': self.customer_insights.get('service_patterns', {}).get('current_status', {}).get('message', 'Normal service levels')
            },
            'daily_stats': {
                'total_customers': self.customer_insights['total_customers_today'],
                'average_satisfaction': self.customer_insights['average_satisfaction']
            },
            'recommendations': [],
            'communication_status': self.a2a_comm.connection_status if self.a2a_comm else 'inactive'
        }
        
        # Generate customer recommendations
        if current_queue == 0:
            insights['recommendations'].append("Perfect time to visit - no wait!")
        elif current_queue <= 3:
            insights['recommendations'].append("Short queue - good time for service")
        elif current_queue <= 8:
            insights['recommendations'].append("Moderate wait expected")
        else:
            insights['recommendations'].append("Long queue - consider visiting later if not urgent")
        
        return insights

# Initialize RAG system
@st.cache_resource
def initialize_rag():
    """Initialize the RAG system"""
    config = load_config()
    return SmartBranchRAG(config['GOOGLE_API_KEY'])

# Password hashing functions
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed_password):
    """Verify password against hash"""
    return hash_password(password) == hashed_password

# Database functions
def init_customer_database():
    """Initialize customer database with updated schema"""
    db_path = Path("data")
    db_path.mkdir(exist_ok=True)
    
    conn = sqlite3.connect('data/database.db')
    cursor = conn.cursor()
    
    # Create users table with username and password fields
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            age INTEGER,
            gender TEXT,
            face_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queue_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            join_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            estimated_wait INTEGER,
            actual_wait INTEGER,
            status TEXT DEFAULT 'waiting',
            position INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            rating INTEGER,
            comments TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create chat history table for chatbot
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create index for username lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(user_id)')
    
    conn.commit()
    conn.close()

# User authentication functions
def register_user(username, password, profile, face_id=None):
    """Register new user with username and password"""
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            conn.close()
            return None, "Username already exists"
        
        # Hash password
        password_hash = hash_password(password)
        
        # Insert new user
        cursor.execute('''
            INSERT INTO users (username, password_hash, name, email, phone, age, gender, face_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (username, password_hash, profile['name'], profile['email'], profile['phone'], 
              profile['age'], profile['gender'], face_id))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return user_id, "Registration successful"
        
    except sqlite3.IntegrityError as e:
        return None, f"Registration failed: Username already exists"
    except Exception as e:
        return None, f"Registration failed: {str(e)}"

def authenticate_user(username, password):
    """Authenticate user with username and password"""
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, password_hash, name, email, phone, age, gender
            FROM users WHERE username = ?
        ''', (username,))
        
        user = cursor.fetchone()
        conn.close()
        
        if user and verify_password(password, user[1]):
            return {
                'id': user[0],
                'username': username,
                'name': user[2],
                'email': user[3],
                'phone': user[4],
                'age': user[5],
                'gender': user[6]
            }, "Login successful"
        else:
            return None, "Invalid username or password"
            
    except Exception as e:
        return None, f"Login failed: {str(e)}"

def get_user_by_id(user_id):
    """Get user details by ID"""
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, name, email, phone, age, gender
            FROM users WHERE id = ?
        ''', (user_id,))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'name': user[2],
                'email': user[3],
                'phone': user[4],
                'age': user[5],
                'gender': user[6]
            }
        return None
            
    except Exception as e:
        st.error(f"Error getting user: {e}")
        return None

# Chat history functions
def save_chat_history(user_id, question, answer):
    """Save chat interaction to database"""
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO chat_history (user_id, question, answer)
            VALUES (?, ?, ?)
        ''', (user_id, question, answer))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")
        return False

def get_user_chat_history(user_id, limit=10):
    """Get user's recent chat history"""
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT question, answer, created_at
            FROM chat_history 
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'question': row[0],
                'answer': row[1],
                'timestamp': row[2]
            }
            for row in results
        ]
        
    except Exception as e:
        st.error(f"Error getting chat history: {e}")
        return []

# Face recognition functions (simplified)
@st.cache_resource
def load_face_detector():
    """Load OpenCV face detector"""
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_face_id(image):
    """Generate a simple face ID from image features"""
    if image is None:
        return None
    
    # Convert to grayscale and detect faces
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_detector = load_face_detector()
    faces = face_detector.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
    
    if len(faces) == 0:
        return None
    
    # Use the first face detected
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # Resize to standard size
    face_roi = cv2.resize(face_roi, (100, 100))
    
    # Create a simple hash from pixel values (not secure, for demo only)
    face_hash = hashlib.md5(face_roi.tobytes()).hexdigest()
    
    return face_hash

def recognize_user_face(image):
    """Recognize user from face image"""
    face_id = generate_face_id(image)
    if not face_id:
        return None, "No face detected"
    
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, name, email, phone, age, gender
            FROM users WHERE face_id = ?
        ''', (face_id,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'name': user[2],
                'email': user[3],
                'phone': user[4],
                'age': user[5],
                'gender': user[6]
            }, "User recognized"
        else:
            return None, "Face not recognized"
            
    except Exception as e:
        return None, f"Recognition failed: {str(e)}"

# Queue management functions
def get_queue_size():
    """Get current queue size"""
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM queue_entries WHERE status = 'waiting'")
        queue_size = cursor.fetchone()[0]
        conn.close()
        return queue_size
    except:
        return 0

def join_queue(user_id):
    """Enhanced join_queue function with agent integration"""
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        
        # Check if user already in queue
        cursor.execute("SELECT id FROM queue_entries WHERE user_id = ? AND status = 'waiting'", (user_id,))
        if cursor.fetchone():
            conn.close()
            return None, "Already in queue"
        
        # Get current queue size for position
        queue_size = get_queue_size()
        position = queue_size + 1
        
        # Estimate wait time (5 minutes per person ahead)
        estimated_wait = position * 5
        
        cursor.execute('''
            INSERT INTO queue_entries (user_id, estimated_wait, position)
            VALUES (?, ?, ?)
        ''', (user_id, estimated_wait, position))
        
        entry_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Enhanced agent notification
        if 'customer_agent' in st.session_state and st.session_state.get('customer_agent_running', False):
            user_profile = get_user_by_id(user_id) or {}
            prediction = st.session_state.customer_agent.handle_customer_join(user_id, user_profile)
        
        return entry_id, f"Added to queue at position {position}"
        
    except Exception as e:
        return None, f"Failed to join queue: {str(e)}"


def get_user_queue_status(user_id):
    """Get user's current queue status"""
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT position, estimated_wait, join_time, status
            FROM queue_entries 
            WHERE user_id = ? AND status = 'waiting'
            ORDER BY join_time DESC LIMIT 1
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            position, estimated_wait, join_time, status = result
            
            # Calculate updated wait time based on current position
            current_queue_size = get_queue_size()
            people_ahead = max(0, position - 1)
            updated_wait = people_ahead * 5  # 5 minutes per person
            
            return {
                'position': position,
                'people_ahead': people_ahead,
                'estimated_wait': updated_wait,
                'join_time': join_time,
                'status': status
            }
        
        return None
        
    except Exception as e:
        st.error(f"Error getting queue status: {e}")
        return None

def leave_queue(user_id):
    """Enhanced leave_queue function with agent integration"""
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE queue_entries 
            SET status = 'left' 
            WHERE user_id = ? AND status = 'waiting'
        ''', (user_id,))
        
        conn.commit()
        conn.close()
        
        # Enhanced agent notification
        if 'customer_agent' in st.session_state and st.session_state.get('customer_agent_running', False):
            st.session_state.customer_agent.handle_customer_leave(user_id)
        
        return True
        
    except Exception as e:
        st.error(f"Error leaving queue: {e}")
        return False
    
# Notification functions
def send_email_notification(user_email, subject, message, config):
    """Send email notification"""
    if not config['SMTP_HOST'] or not user_email or not config['SMTP_USER']:
        return False, "Email configuration incomplete"
    
    try:
        msg = MIMEMultipart()
        msg['From'] = config['FROM_EMAIL']
        msg['To'] = user_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(message, 'plain'))
        
        # Use SSL for port 465, TLS for others
        if config['SMTP_PORT'] == 465:
            server = smtplib.SMTP_SSL(config['SMTP_HOST'], config['SMTP_PORT'])
        else:
            server = smtplib.SMTP(config['SMTP_HOST'], config['SMTP_PORT'])
            server.starttls()
        
        server.login(config['SMTP_USER'], config['SMTP_PASS'])
        server.sendmail(config['FROM_EMAIL'], user_email, msg.as_string())
        server.quit()
        
        return True, "Email sent successfully"
        
    except Exception as e:
        return False, f"Email sending failed: {str(e)}"

def send_sms_notification(phone, message, config):
    """Send SMS notification via Twilio"""
    if not config['TWILIO_ACCOUNT_SID'] or not phone:
        return False, "SMS configuration incomplete"
    
    try:
        from twilio.rest import Client
        client = Client(config['TWILIO_ACCOUNT_SID'], config['TWILIO_AUTH_TOKEN'])
        
        message = client.messages.create(
            body=message,
            from_=config['TWILIO_PHONE_NUMBER'],
            to=phone
        )
        
        return True, "SMS sent successfully"
        
    except ImportError:
        return False, "Twilio library not installed"
    except Exception as e:
        return False, f"SMS sending failed: {str(e)}"

# Feedback functions
def save_feedback(user_id, rating, comments):
    """Enhanced save_feedback function with agent integration"""
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (user_id, rating, comments)
            VALUES (?, ?, ?)
        ''', (user_id, rating, comments))
        
        conn.commit()
        conn.close()
        
        # Enhanced agent notification
        if 'customer_agent' in st.session_state and st.session_state.get('customer_agent_running', False):
            st.session_state.customer_agent.handle_feedback_submission(user_id, rating, comments)
        
        return True
        
    except Exception as e:
        st.error(f"Failed to save feedback: {e}")
        return False
    
def get_user_feedback_history(user_id):
    """Get user's feedback history"""
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT rating, comments, created_at
            FROM feedback 
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 5
        ''', (user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'rating': row[0],
                'comments': row[1],
                'date': row[2]
            }
            for row in results
        ]
        
    except Exception as e:
        st.error(f"Error getting feedback history: {e}")
        return []

# PDF generation for receipts/documents
def create_simple_pdf(content, filename):
    """Create a simple text-based PDF"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        
        # Split content into lines
        lines = content.split('\n')
        y = height - 50  # Start from top
        
        for line in lines:
            if y < 50:  # New page if needed
                c.showPage()
                y = height - 50
            c.drawString(50, y, line)
            y -= 20
        
        c.save()
        return True
        
    except ImportError:
        # Fallback: create text file if reportlab not available
        with open(filename.replace('.pdf', '.txt'), 'w') as f:
            f.write(content)
        return True
    except Exception as e:
        st.error(f"PDF creation failed: {e}")
        return False

# SIDEBAR MANAGEMENT FUNCTIONS
def render_sidebar_system_control(config):
    """Render system control in sidebar"""
    st.sidebar.title("🤖 System Control")
    
    # Initialize session state
    if 'customer_agent' not in st.session_state:
        st.session_state.customer_agent = None
    if 'customer_agent_running' not in st.session_state:
        st.session_state.customer_agent_running = False
    
    # Agent Status Display
    agent_status = "🟢 Running" if st.session_state.customer_agent_running else "🔴 Stopped"
    st.sidebar.write(f"**Status:** {agent_status}")
    
    # Communication Status
    if st.session_state.customer_agent:
        comm_status = st.session_state.customer_agent.get_communication_status()
        conn_status = comm_status['status']
        status_color = {
            'connected': '🟢',
            'degraded': '🟡', 
            'disconnected': '🔴',
            'stopped': '⚫'
        }
        st.sidebar.write(f"**A2A:** {status_color.get(conn_status, '❓')}")
    
    # Enhanced Controls
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🚀 Start", disabled=st.session_state.customer_agent_running, key="sidebar_start"):
            try:
                customer_config = config.copy()
                st.session_state.customer_agent = EnhancedCustomerServiceAgent(customer_config)
                
                if st.session_state.customer_agent.start_autonomous_monitoring():
                    st.session_state.customer_agent_running = True
                    st.sidebar.success("Agent started!")
                    
                    # Verify communication system
                    time.sleep(1)
                    comm_status = st.session_state.customer_agent.get_communication_status()
                    if comm_status['status'] in ['connected', 'degraded']:
                        st.sidebar.success("✅ A2A Active!")
                    else:
                        st.sidebar.warning("⚠️ A2A Limited")
                else:
                    st.sidebar.error("Failed to start agent")
                    
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("⏹️ Stop", disabled=not st.session_state.customer_agent_running, key="sidebar_stop"):
            try:
                if st.session_state.customer_agent:
                    st.session_state.customer_agent.stop_autonomous_monitoring()
                st.session_state.customer_agent_running = False
                st.sidebar.success("Agent stopped!")
                
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")

    # Enhanced Agent Management Section
    with st.sidebar.expander("📊 Enhanced Customer Service Agent", expanded=False):
        if st.session_state.customer_agent_running and st.session_state.customer_agent:
            
            # Communication Status
            st.subheader("🔗 A2A Communication Status")
            comm_status = st.session_state.customer_agent.get_communication_status()
            
            col1, col2 = st.columns(2)
            with col1:
                status_color = {
                    'connected': '🟢',
                    'degraded': '🟡', 
                    'disconnected': '🔴',
                    'stopped': '⚫'
                }
                st.write(f"**Connection:** {status_color.get(comm_status['status'], '❓')} {comm_status['status'].title()}")
            
            with col2:
                app1_status = "🟢 Healthy" if comm_status['app1_healthy'] else "🔴 Unreachable"
                st.write(f"**App1:** {app1_status}")
            
            st.write(f"**Queue Size:** {comm_status['message_queue_size']}")
            
            if comm_status['last_heartbeat']:
                heartbeat_time = datetime.fromisoformat(comm_status['last_heartbeat'])
                st.write(f"**Last Heartbeat:** {heartbeat_time.strftime('%H:%M:%S')}")
            else:
                st.write("**Last Heartbeat:** Never")
            
            # Agent Insights
            st.subheader("🧠 Agent Insights")
            insights = st.session_state.customer_agent.get_customer_insights()
            
            # Current Status Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Queue", insights['current_status']['queue_length'])
            with col2:
                predicted_wait = insights['current_status']['predicted_wait']
                st.metric("Wait", f"{predicted_wait}m")
            
            st.metric("Customers Today", insights['daily_stats']['total_customers'])
            
            # Service Status
            service_status = insights['current_status']['service_status']
            st.info(f"📋 {service_status}")
            
            # Recommendations
            if insights['recommendations']:
                st.subheader("💡 Recommendations")
                for recommendation in insights['recommendations']:
                    st.info(recommendation)
            
            # Communication Testing
            st.subheader("🧪 Communication Testing")
            
            if st.button("📡 Test App1", key="test_app1_sidebar"):
                success = st.session_state.customer_agent.a2a_comm.check_app1_health()
                if success:
                    st.success("✅ App1 reachable!")
                else:
                    st.error("❌ Cannot reach App1")
            
            if st.button("🤝 Handshake", key="handshake_sidebar"):
                success = st.session_state.customer_agent.a2a_comm.perform_handshake()
                if success:
                    st.success("✅ Handshake successful!")
                else:
                    st.error("❌ Handshake failed")
            
            if st.button("📤 Test Message", key="test_msg_sidebar"):
                test_message = {
                    'type': 'test_connection',
                    'data': {
                        'message': 'Test from Customer Service Agent',
                        'test_id': str(uuid.uuid4()),
                        'timestamp': datetime.now().isoformat()
                    }
                }
                success = st.session_state.customer_agent.send_message_to_app1(test_message)
                if success:
                    st.success("✅ Test message sent!")
                else:
                    st.error("❌ Test message failed")
            
            # Enhanced Logs Display
            st.subheader("📋 Activity Logs")
            
            # Combine agent logs and communication logs
            all_logs = []
            if hasattr(st.session_state.customer_agent, 'logs'):
                all_logs.extend(st.session_state.customer_agent.logs)
            if hasattr(st.session_state.customer_agent.a2a_comm, 'logs'):
                all_logs.extend(st.session_state.customer_agent.a2a_comm.logs)
            
            # Sort by timestamp and show recent logs
            all_logs = sorted(all_logs)[-8:]  # Show last 8 logs for sidebar
            
            if all_logs:
                for log_entry in all_logs:
                    st.text(log_entry)
            else:
                st.info("No activity logs")
                
        else:
            st.info("Start the Enhanced Customer Service Agent to view details")
    
    # Quick Status Check
    if st.sidebar.button("🔍 Check Status", key="check_status_sidebar"):
        try:
            # Check Flask server
            response = requests.get("http://localhost:5002/customer_status", timeout=2)
            if response.status_code == 200:
                st.sidebar.success("✅ Flask running!")
                
                # Check app1.py connection if agent running
                if st.session_state.customer_agent_running and st.session_state.customer_agent:
                    app1_healthy = st.session_state.customer_agent.a2a_comm.check_app1_health()
                    if app1_healthy:
                        st.sidebar.success("✅ Connected to App1!")
                    else:
                        st.sidebar.warning("⚠️ App1 not reachable")
            else:
                st.sidebar.error(f"❌ Flask error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.sidebar.error("🔴 Flask not running")
        except Exception as e:
            st.sidebar.error(f"❌ Status check failed: {e}")
    
    # Quick Insights in Sidebar
    if st.session_state.customer_agent_running and st.session_state.customer_agent:
        try:
            insights = st.session_state.customer_agent.get_customer_insights()
            
            st.sidebar.divider()
            st.sidebar.subheader("📊 Quick Stats")
            st.sidebar.metric("Current Queue", insights['current_status']['queue_length'])
            st.sidebar.metric("Predicted Wait", f"{insights['current_status']['predicted_wait']}m")
            
            # Recent activity
            if st.session_state.customer_agent.logs:
                st.sidebar.write("**Recent Activity:**")
                for log in st.session_state.customer_agent.logs[-2:]:
                    st.sidebar.caption(log)
                    
        except Exception as e:
            st.sidebar.error(f"Error getting insights: {e}")

    # Debug Info in Sidebar
    with st.sidebar.expander("🔧 Debug Info", expanded=False):
        st.write("**Configuration:**")
        st.write(f"- APP1_ENDPOINT: {config.get('APP1_ENDPOINT', 'Not set')}")
        st.write(f"- PORT: {config.get('CUSTOMER_AGENT_PORT', 'Not set')}")
        st.write(f"- A2A: {config.get('ENABLE_A2A_COMMUNICATION', 'Not set')}")
        
        if st.session_state.customer_agent:
            st.write("**Agent Details:**")
            st.write(f"- Agent ID: {st.session_state.customer_agent.agent_id}")
            st.write(f"- Status: {st.session_state.customer_agent.a2a_comm.connection_status}")
        
        if st.button("🗑️ Reset Agent", key="reset_sidebar"):
            st.session_state.customer_agent = None
            st.session_state.customer_agent_running = False
            st.sidebar.success("Agent reset!")
            st.rerun()

# Main Streamlit app
def main():
    st.title("🎫 Smart Branch - Customer Service")
    
    # Initialize database
    init_customer_database()
    config = load_config()
    
    # Initialize session state
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None
    if 'queue_entry_id' not in st.session_state:
        st.session_state.queue_entry_id = None
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # SIDEBAR: System Control - Always visible
    render_sidebar_system_control(config)
    
    # Navigation based on login status
    if st.session_state.user_id is None:
        auth_tab()
    else:
        user_dashboard()

def auth_tab():
    """Authentication tab for registration/login"""
    st.header("Welcome to Smart Branch!")
    
    # Login/Register tabs
    tab1, tab2, tab3 = st.tabs(["🔐 Login", "📝 Register", "📸 Face Login"])
    
    with tab1:
        login_section()
    
    with tab2:
        registration_section()
    
    with tab3:
        face_login_section()

def login_section():
    """Username/password login"""
    st.subheader("Login with Username & Password")
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        submitted = st.form_submit_button("Login", type="primary")
        
        if submitted:
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                with st.spinner("Logging in..."):
                    user_profile, message = authenticate_user(username, password)
                    
                    if user_profile:
                        st.session_state.user_id = user_profile['id']
                        st.session_state.user_profile = user_profile
                        st.success(f"Welcome back, {user_profile['name']}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
    
    # Demo credentials info
    st.info("💡 Demo: After registration, you can login with your chosen username and password")

def face_login_section():
    """Face recognition login (optional)"""
    st.subheader("Login with Face Recognition")
    
    st.info("Take a photo to login with face recognition (if you registered with face ID)")
    
    # Camera capture
    camera_input = st.camera_input("Take a picture to login")
    
    if camera_input is not None:
        # Convert to OpenCV format
        image = Image.open(camera_input)
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        with st.spinner("Recognizing..."):
            user_profile, message = recognize_user_face(image_bgr)
            
            if user_profile:
                st.session_state.user_id = user_profile['id']
                st.session_state.user_profile = user_profile
                st.success(f"Welcome back, {user_profile['name']}!")
                time.sleep(1)
                st.rerun()
            else:
                st.error(message)
                st.info("Face not recognized. Please try regular login or register first.")

def registration_section():
    """New user registration"""
    st.subheader("Create New Account")
    
    with st.form("registration_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username*", placeholder="Choose a username")
            password = st.text_input("Password*", type="password", placeholder="Choose a password")
            confirm_password = st.text_input("Confirm Password*", type="password", placeholder="Confirm your password")
            
        with col2:
            name = st.text_input("Full Name*", placeholder="John Doe")
            email = st.text_input("Email", placeholder="john@example.com")
            phone = st.text_input("Phone (Optional)", placeholder="+1234567890")
        
        col3, col4 = st.columns(2)
        with col3:
            age = st.number_input("Age", min_value=1, max_value=120, value=25)
        with col4:
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        
        # Optional face capture
        st.write("**Optional: Face Photo for Face Recognition Login**")
        st.info("You can skip this and use username/password login only")
        camera_input = st.camera_input("Take your photo (optional)")
        
        submitted = st.form_submit_button("Create Account", type="primary")
        
        if submitted:
            # Validate required fields
            if not username or not password or not name:
                st.error("Username, password, and name are required")
            elif password != confirm_password:
                st.error("Passwords do not match")
            elif len(password) < 4:
                st.error("Password must be at least 4 characters long")
            else:
                with st.spinner("Creating account..."):
                    # Generate face ID if photo provided
                    face_id = None
                    if camera_input:
                        image = Image.open(camera_input)
                        image_array = np.array(image)
                        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                        face_id = generate_face_id(image_bgr)
                        
                        if not face_id:
                            st.warning("Could not detect face in photo. Account will be created without face recognition.")
                    
                    # Create profile
                    profile = {
                        'name': name,
                        'email': email,
                        'phone': phone,
                        'age': age,
                        'gender': gender.lower()
                    }
                    
                    # Register user
                    user_id, message = register_user(username, password, profile, face_id)
                    
                    if user_id:
                        st.success("Account created successfully!")
                        st.info("You can now login with your username and password")
                        
                        # Auto-login the user
                        profile['id'] = user_id
                        profile['username'] = username
                        st.session_state.user_id = user_id
                        st.session_state.user_profile = profile
                        
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(message)

def user_dashboard():
    """Clean customer-focused dashboard"""
    user = st.session_state.user_profile
    config = load_config()
    
    # Clean Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(f"Hello, {user['name']}! 👋")
        st.caption(f"Welcome to Smart Branch Customer Portal")
    with col2:
        if st.button("Logout", type="secondary"):
            # Clear session
            st.session_state.user_id = None
            st.session_state.user_profile = None
            st.session_state.queue_entry_id = None
            st.session_state.chat_messages = []
            st.rerun()
    
    # Customer-focused tabs only
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Queue Status", "📱 Notifications", "💭 Feedback", "🤖 Help Assistant"])
    
    with tab1:
        queue_status_tab(config)
    
    with tab2:
        notifications_tab(config)
    
    with tab3:
        feedback_tab(config)
    
    with tab4:
        chatbot_tab(config)

def chatbot_tab(config):
    """RAG-powered chatbot for customer queries"""
    st.subheader("🤖 Smart Branch Assistant")
    st.write("Ask me anything about Smart Branch system - queue management, services, or how things work!")
    
    # Initialize RAG system
    rag_system = initialize_rag()
    
    # API key status
    if not config['GOOGLE_API_KEY']:
        st.warning("⚠️ AI chatbot requires Google API key to be configured. You can still get basic answers from the knowledge base.")
        st.info("To enable full AI responses, please add your GOOGLE_API_KEY to the .env file.")
    else:
        st.success("✅ AI chatbot is ready!")
    
    # Chat interface
    chat_container = st.container()
    
    # Load chat history for the current user
    user_id = st.session_state.user_id
    if 'chat_initialized' not in st.session_state:
        st.session_state.chat_messages = get_user_chat_history(user_id, limit=5)
        st.session_state.chat_initialized = True
    
    # Display chat history
    with chat_container:
        if st.session_state.chat_messages:
            st.write("**Recent Conversations:**")
            for i, chat in enumerate(reversed(st.session_state.chat_messages[-3:])):  # Show last 3
                with st.expander(f"Q: {chat['question'][:50]}..." if len(chat['question']) > 50 else f"Q: {chat['question']}", expanded=False):
                    st.write(f"**Question:** {chat['question']}")
                    st.write(f"**Answer:** {chat['answer']}")
                    st.caption(f"Asked on: {chat['timestamp']}")
        else:
            st.info("No previous conversations. Ask your first question below!")
    
    # Current question input
    st.write("**Ask a New Question:**")
    
    # Predefined quick questions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("How do I join the queue?", key="quick_q1"):
            st.session_state.current_question = "How do I join the queue?"
        if st.button("How long will I wait?", key="quick_q2"):
            st.session_state.current_question = "How long will I wait in the queue?"
        if st.button("How do notifications work?", key="quick_q3"):
            st.session_state.current_question = "How do notifications work?"
    
    with col2:
        if st.button("Can I leave the queue?", key="quick_q4"):
            st.session_state.current_question = "Can I leave the queue once I join?"
        if st.button("What is face recognition?", key="quick_q5"):
            st.session_state.current_question = "How does face recognition login work?"
        if st.button("Technical requirements?", key="quick_q6"):
            st.session_state.current_question = "What are the technical requirements?"
    
    # Question input
    question = st.text_area(
        "Your Question:",
        value=st.session_state.get('current_question', ''),
        placeholder="Type your question about Smart Branch system...",
        height=100,
        key="question_input"
    )
    
    # Clear the current_question after displaying
    if 'current_question' in st.session_state:
        del st.session_state.current_question
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("Ask Question", type="primary", disabled=not question.strip())
    
    with col2:
        if st.button("Clear Chat History"):
            st.session_state.chat_messages = []
            # Also clear from database
            try:
                conn = sqlite3.connect('data/database.db')
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
                conn.commit()
                conn.close()
                st.success("Chat history cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing chat history: {e}")
    
    # Process question
    if ask_button and question.strip():
        with st.spinner("Thinking..."):
            # Get answer from RAG system
            answer = rag_system.get_answer(question.strip())
            
            # Save to database
            save_chat_history(user_id, question.strip(), answer)
            
            # Add to session state
            new_chat = {
                'question': question.strip(),
                'answer': answer,
                'timestamp': datetime.now().isoformat()
            }
            st.session_state.chat_messages.append(new_chat)
            
            # Display the new Q&A
            st.write("---")
            st.write("**Your Question:**")
            st.write(question)
            st.write("**Answer:**")
            st.write(answer)
            
            # Auto-scroll to the answer
            time.sleep(0.5)
            st.rerun()
    
    # Chat statistics
    with st.expander("💬 Chat Statistics", expanded=False):
        total_chats = len(get_user_chat_history(user_id, limit=1000))
        st.metric("Total Questions Asked", total_chats)
        
        if total_chats > 0:
            st.write("**Most Recent Topics:**")
            recent_chats = get_user_chat_history(user_id, limit=5)
            for chat in recent_chats:
                st.write(f"• {chat['question'][:60]}..." if len(chat['question']) > 60 else f"• {chat['question']}")
    
    # Help section
    with st.expander("❓ What can I ask about?", expanded=False):
        st.write("""
        **I can help you with questions about:**
        
        **Queue Management:**
        - How to join or leave the queue
        - Wait time estimations
        - Queue positions and updates
        - What happens if you miss your turn
        
        **Account & Login:**
        - Registration process
        - Login methods (username/password, face recognition)
        - Profile information
        - Security and privacy
        
        **Notifications:**
        - Email and SMS notifications
        - Notification settings
        - When you'll be notified
        
        **System Features:**
        - Feedback and rating system
        - Receipt generation
        - Technical requirements
        - Browser compatibility
        
        **General Information:**
        - Branch hours and capacity
        - How the system works
        - Contact information
        - Troubleshooting
        
        **Try asking specific questions like:**
        - "How accurate are the wait times?"
        - "What if I don't have a camera for face recognition?"
        - "Can I change my notification preferences?"
        - "What happens to my data?"
        """)

def queue_status_tab(config):
    """Queue status and wait time information"""
    st.subheader("Current Queue Status")
    
    user_id = st.session_state.user_id
    queue_status = get_user_queue_status(user_id)
    
    # Get overall queue information
    total_queue_size = get_queue_size()
    
    if queue_status is None:
        # User not in queue
        st.info(f"Current queue size: {total_queue_size} people")
        
        if st.button("Join Queue", type="primary"):
            entry_id, message = join_queue(user_id)
            if entry_id:
                st.session_state.queue_entry_id = entry_id
                st.success(message)
                
                # Send welcome email
                user = st.session_state.user_profile
                if user.get('email') and config['ENABLE_NOTIFICATIONS']:
                    welcome_message = f"""
Dear {user['name']},

Welcome to Smart Branch queue system!

You have successfully joined the queue:
- Position: {get_user_queue_status(user_id)['position'] if get_user_queue_status(user_id) else 'N/A'}
- Estimated wait time: {get_user_queue_status(user_id)['estimated_wait'] if get_user_queue_status(user_id) else 'N/A'} minutes

We'll keep you updated on your queue status.

Best regards,
Smart Branch Team
"""
                    success, msg = send_email_notification(
                        user['email'], 
                        "Queue Confirmation - Smart Branch", 
                        welcome_message, 
                        config
                    )
                    if success:
                        st.info("Confirmation email sent!")
                
                st.rerun()
            else:
                st.error(message)
    
    else:
        # User is in queue
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Your Position", queue_status['position'])
        
        with col2:
            st.metric("People Ahead", queue_status['people_ahead'])
        
        with col3:
            st.metric("Est. Wait Time", f"{queue_status['estimated_wait']} min")
        
        # Progress bar
        st.write("**Queue Progress:**")
        if total_queue_size > 0:
            progress = max(0, 1 - (queue_status['people_ahead'] / total_queue_size))
        else:
            progress = 1.0
        st.progress(progress)
        
        # Status message
        people_ahead = queue_status['people_ahead']
        if people_ahead == 0:
            st.success("🎉 You're next! Please proceed to the counter.")
            
            # Simulate service completion
            if st.button("Service Completed"):
                # Mark as completed and move to feedback
                leave_queue(user_id)
                st.success("Thank you for your visit!")
                st.rerun()
                
        elif people_ahead <= 3:
            st.warning(f"⏰ Almost your turn! {people_ahead} people ahead.")
        else:
            st.info(f"⏳ Please wait. {people_ahead} people ahead of you.")
        
        # Leave queue option
        if st.button("Leave Queue"):
            if leave_queue(user_id):
                st.session_state.queue_entry_id = None
                st.info("You've left the queue.")
                st.rerun()
        
        # Auto-refresh option
        if st.checkbox("Auto-refresh (30 seconds)"):
            time.sleep(30)
            st.rerun()

def notifications_tab(config):
    """Notification preferences and alerts"""
    st.subheader("Notification Settings")
    
    user = st.session_state.user_profile
    
    if not config['ENABLE_NOTIFICATIONS']:
        st.warning("Notifications are currently disabled in system configuration")
        return
    
    # Email configuration status
    st.write("**Email Configuration Status:**")
    if config['SMTP_USER'] and config['SMTP_PASS']:
        st.success("✅ Email system configured")
    else:
        st.error("❌ Email system not configured")
    
    # Notification preferences
    st.write("**Get notified when it's almost your turn:**")
    
    notification_methods = []
    
    if user.get('email'):
        email_notifications = st.checkbox(
            f"📧 Email notifications ({user['email']})",
            value=True
        )
        if email_notifications:
            notification_methods.append('email')
    
    if user.get('phone'):
        sms_notifications = st.checkbox(
            f"📱 SMS notifications ({user['phone']})",
            value=False
        )
        if sms_notifications:
            notification_methods.append('sms')
    
    # Test notification
    if notification_methods and st.button("Send Test Notification"):
        test_message = f"Hi {user['name']}, this is a test notification from Smart Branch!"
        
        success_count = 0
        messages = []
        
        for method in notification_methods:
            if method == 'email' and user.get('email'):
                success, msg = send_email_notification(
                    user['email'], 
                    "Test Notification - Smart Branch", 
                    test_message, 
                    config
                )
                messages.append(f"Email: {msg}")
                if success:
                    success_count += 1
                    
            elif method == 'sms' and user.get('phone'):
                success, msg = send_sms_notification(user['phone'], test_message, config)
                messages.append(f"SMS: {msg}")
                if success:
                    success_count += 1
        
        if success_count > 0:
            st.success(f"Test notification sent via {success_count} channel(s)!")
        
        # Show detailed messages
        for msg in messages:
            if "successfully" in msg.lower():
                st.success(msg)
            else:
                st.error(msg)
    
    # Auto-notification status
    queue_status = get_user_queue_status(st.session_state.user_id)
    if queue_status and notification_methods:
        if queue_status['people_ahead'] <= 3:
            st.info("🔔 You'll be notified when you're next in line!")
        else:
            st.info("✅ Notifications are active for your queue position")

def feedback_tab(config):
    """Feedback and rating system"""
    st.subheader("Your Feedback")
    
    user_id = st.session_state.user_id
    user = st.session_state.user_profile
    
    # Check if user recently completed service (mock check)
    queue_status = get_user_queue_status(user_id)
    
    if queue_status:
        st.info("Feedback will be available after your service is completed.")
        
        # For demo purposes, add a button to simulate service completion
        if st.button("Complete Service (Demo)", help="Simulate completing your service"):
            leave_queue(user_id)
            st.success("Service completed! Please provide your feedback below.")
            st.rerun()
        return
    
    # Initialize session state for receipt generation
    if 'receipt_data' not in st.session_state:
        st.session_state.receipt_data = None
    
    # Feedback form
    with st.form("feedback_form"):
        st.write("**How was your experience?**")
        
        # Rating
        rating = st.select_slider(
            "Overall Rating",
            options=[1, 2, 3, 4, 5],
            value=5,
            format_func=lambda x: "⭐" * x
        )
        
        # Comments
        comments = st.text_area(
            "Additional Comments (Optional)",
            placeholder="Tell us about your experience..."
        )
        
        submitted = st.form_submit_button("Submit Feedback", type="primary")
        
        if submitted:
            # Save feedback
            success = save_feedback(user_id, rating, comments)
            
            if success:
                st.success("Thank you for your feedback!")
                
                # Send receipt email if configured
                if user.get('email') and config['ENABLE_NOTIFICATIONS']:
                    receipt_message = f"""
Dear {user['name']},

Thank you for visiting Smart Branch today!

Visit Summary:
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Rating Given: {'⭐' * rating}
- Customer ID: {user_id}
- Username: @{user['username']}

We appreciate your feedback and hope to see you again soon!

Best regards,
Smart Branch Team
"""
                    
                    success, msg = send_email_notification(
                        user['email'], 
                        "Visit Receipt - Smart Branch", 
                        receipt_message, 
                        config
                    )
                    
                    if success:
                        st.info(f"Receipt sent to {user['email']}")
                        
                        # Store receipt data in session state for download
                        st.session_state.receipt_data = {
                            'content': receipt_message,
                            'filename': f"receipt_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        }
                        
                    else:
                        st.warning(f"Could not send receipt email: {msg}")
                
            else:
                st.error("Failed to save feedback")
    
    # Download button OUTSIDE the form
    if st.session_state.receipt_data:
        receipt_info = st.session_state.receipt_data
        
        st.subheader("Download Receipt")
        
        # Create the PDF file
        if create_simple_pdf(receipt_info['content'], receipt_info['filename']):
            try:
                with open(receipt_info['filename'], 'rb') as f:
                    st.download_button(
                        "📄 Download Receipt PDF",
                        f.read(),
                        file_name=receipt_info['filename'],
                        mime="application/pdf"
                    )
                    
                # Clean up the file after creating the download button
                if os.path.exists(receipt_info['filename']):
                    os.remove(receipt_info['filename'])
                    
            except Exception as e:
                st.error(f"Error creating PDF download: {e}")
        
        # Clear the receipt data after use
        if st.button("Clear Receipt Download"):
            st.session_state.receipt_data = None
            st.rerun()
    
    # Previous feedback
    st.subheader("Your Previous Feedback")
    feedback_history = get_user_feedback_history(user_id)
    
    if feedback_history:
        for feedback in feedback_history:
            st.write(f"**{feedback['date'][:10]}:** {'⭐' * feedback['rating']} - {feedback['comments'] or 'No comments'}")
    else:
        st.write("No previous feedback found")

if __name__ == "__main__":
    main()