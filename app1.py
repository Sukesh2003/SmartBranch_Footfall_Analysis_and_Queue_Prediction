import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import cv2
from PIL import Image
import os
import sqlite3
import json
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from dotenv import load_dotenv
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import requests
import json
from flask import Flask, request, jsonify
import threading
from queue import Queue as ThreadQueue
import uuid
from datetime import datetime, timedelta
from queue import Queue


# Load environment variables
load_dotenv()

# Initialize logging for agent
logging.basicConfig(level=logging.INFO)

# Agent configuration dataclass
@dataclass
class MonitoringConfig:
    """Configuration for the monitoring agent"""
    camera_check_interval: int = 30  # seconds
    database_update_interval: int = 60  # seconds
    max_capacity: int = 200
    alert_threshold: float = 0.8  # 80% capacity
    staff_ratio: int = 25  # 1 staff per 25 people
    smtp_host: str = ""
    smtp_port: int = 465
    smtp_user: str = ""
    smtp_pass: str = ""
    notification_emails: List[str] = None
    entrance_camera_index: int = 0  # Default camera for entrance
    exit_camera_index: int = 1      # Second camera for exit (if available)
    use_single_camera: bool = True   # Use single camera for both entrance/exit

    def __post_init__(self):
        if self.notification_emails is None:
            self.notification_emails = []

class CameraManager:
    """Manages camera operations for entrance and exit detection"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.entrance_camera = None
        self.exit_camera = None
        self.last_entrance_count = 0
        self.last_exit_count = 0

    def _detect_available_cameras(self):
        """Detect available camera indices with detailed logging"""
        available_cameras = []
        
        print("Scanning for available cameras...")  # Console debug
        
        # Test camera indices 0-4
        for index in range(5):
            try:
                print(f"Testing camera index {index}...")
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append(index)
                        print(f"✅ Camera {index}: Available ({frame.shape[1]}x{frame.shape[0]})")
                    else:
                        print(f"⚠️ Camera {index}: Opens but no valid frame")
                    cap.release()
                else:
                    print(f"❌ Camera {index}: Cannot open")
            except Exception as e:
                print(f"❌ Camera {index}: Exception - {str(e)}")
                continue
        
        print(f"Detection complete. Available cameras: {available_cameras}")
        return available_cameras

    def initialize_cameras(self):
        """Initialize camera connections with enhanced debugging"""
        try:
            self.add_log("Starting camera initialization...")
            
            # First, detect all available cameras
            available_cameras = self._detect_available_cameras()
            self.add_log(f"Available cameras detected: {available_cameras}")
            
            if not available_cameras:
                self.add_log("No cameras detected - falling back to simulation mode")
                return False
            
            # Try to initialize entrance camera
            entrance_idx = self.config.entrance_camera_index
            self.add_log(f"Attempting to initialize entrance camera at index {entrance_idx}")
            
            if entrance_idx in available_cameras:
                self.entrance_camera = cv2.VideoCapture(entrance_idx)
                if self.entrance_camera.isOpened():
                    # Test if we can actually read from the camera
                    ret, frame = self.entrance_camera.read()
                    if ret:
                        self.add_log(f"✅ Entrance camera {entrance_idx} initialized successfully")
                        print(f"Entrance camera frame shape: {frame.shape}")  # Debug output
                    else:
                        self.add_log(f"❌ Entrance camera {entrance_idx} opens but cannot read frames")
                        self.entrance_camera.release()
                        self.entrance_camera = None
                else:
                    self.add_log(f"❌ Cannot open entrance camera {entrance_idx}")
                    self.entrance_camera = None
            else:
                self.add_log(f"❌ Entrance camera index {entrance_idx} not available")
                # Try first available camera as fallback
                if available_cameras:
                    fallback_idx = available_cameras[0]
                    self.add_log(f"Trying fallback camera {fallback_idx} for entrance")
                    self.entrance_camera = cv2.VideoCapture(fallback_idx)
                    if self.entrance_camera.isOpened():
                        ret, frame = self.entrance_camera.read()
                        if ret:
                            self.config.entrance_camera_index = fallback_idx
                            self.add_log(f"✅ Entrance camera fallback to index {fallback_idx} successful")
                        else:
                            self.entrance_camera.release()
                            self.entrance_camera = None
                    else:
                        self.entrance_camera = None
            
            # Handle exit camera
            if not self.config.use_single_camera:
                exit_idx = self.config.exit_camera_index
                self.add_log(f"Attempting to initialize exit camera at index {exit_idx}")
                
                if exit_idx in available_cameras and exit_idx != self.config.entrance_camera_index:
                    self.exit_camera = cv2.VideoCapture(exit_idx)
                    if self.exit_camera.isOpened():
                        ret, frame = self.exit_camera.read()
                        if ret:
                            self.add_log(f"✅ Exit camera {exit_idx} initialized successfully")
                            print(f"Exit camera frame shape: {frame.shape}")  # Debug output
                        else:
                            self.add_log(f"❌ Exit camera {exit_idx} opens but cannot read frames")
                            self.exit_camera.release()
                            self.exit_camera = None
                            # Fall back to single camera mode
                            self.config.use_single_camera = True
                            self.exit_camera = self.entrance_camera
                            self.add_log("Falling back to single camera mode")
                    else:
                        self.add_log(f"❌ Cannot open exit camera {exit_idx}")
                        self.config.use_single_camera = True
                        self.exit_camera = self.entrance_camera
                        self.add_log("Falling back to single camera mode")
                else:
                    self.add_log(f"Exit camera index {exit_idx} not available or same as entrance")
                    self.config.use_single_camera = True
                    self.exit_camera = self.entrance_camera
                    self.add_log("Using single camera mode")
            else:
                self.exit_camera = self.entrance_camera
                self.add_log("Single camera mode - using entrance camera for both")
            
            success = self.entrance_camera is not None
            
            if success:
                camera_mode = "dual" if (not self.config.use_single_camera and 
                                    self.exit_camera != self.entrance_camera and 
                                    self.exit_camera is not None) else "single"
                self.add_log(f"Camera initialization successful in {camera_mode} mode")
                logging.info(f"Cameras initialized: entrance={self.entrance_camera is not None}, exit={self.exit_camera is not None}")
            else:
                self.add_log("❌ Camera initialization failed - using simulation mode")
                
            return success
            
        except Exception as e:
            error_msg = f"Camera initialization failed with error: {str(e)}"
            self.add_log(error_msg)
            logging.error(error_msg)
            return False
    
    def capture_and_detect(self, camera_type: str = "entrance") -> Dict[str, Any]:
        """Capture frame and detect people"""
        try:
            # Select appropriate camera
            camera = self.entrance_camera if camera_type == "entrance" else self.exit_camera
            
            if camera is None:
                # Fallback to simulated detection if camera not available
                return self._simulate_detection(camera_type)
            
            # Capture frame
            ret, frame = camera.read()
            if not ret:
                logging.warning(f"Failed to capture frame from {camera_type} camera")
                return self._simulate_detection(camera_type)
            
            # Detect faces/people
            faces = self._detect_faces_in_frame(frame)
            
            # Analyze demographics
            demographics = self._analyze_demographics(len(faces))
            
            # Apply movement heuristics for entrance vs exit
            people_count = self._apply_movement_logic(len(faces), camera_type)
            
            return {
                "camera_type": camera_type,
                "people_count": people_count,
                "raw_detections": len(faces),
                "demographics": demographics,
                "frame_captured": True,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            logging.error(f"Detection failed for {camera_type}: {e}")
            return {
                "camera_type": camera_type,
                "people_count": 0,
                "demographics": {"adults": 0, "kids": 0, "males": 0, "females": 0},
                "frame_captured": False,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    def _detect_faces_in_frame(self, frame):
        """Detect faces in the captured frame"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with improved parameters
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
            maxSize=(300, 300),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Filter overlapping detections
        if len(faces) > 1:
            faces = self._filter_overlapping_faces(faces)
        
        return faces
    
    def _filter_overlapping_faces(self, faces):
        """Remove overlapping face detections"""
        filtered_faces = []
        for i, (x1, y1, w1, h1) in enumerate(faces):
            is_duplicate = False
            for j, (x2, y2, w2, h2) in enumerate(faces):
                if i != j:
                    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    overlap_area = overlap_x * overlap_y
                    
                    area1 = w1 * h1
                    area2 = w2 * h2
                    
                    if overlap_area > 0.3 * min(area1, area2):
                        if area1 < area2:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                filtered_faces.append((x1, y1, w1, h1))
        
        return filtered_faces
    
    def _apply_movement_logic(self, detected_count: int, camera_type: str) -> int:
        """Apply logic to determine actual entrance/exit count"""
        if camera_type == "entrance":
            # Compare with previous count to detect new entries
            if detected_count > self.last_entrance_count:
                people_count = detected_count - self.last_entrance_count
                self.last_entrance_count = detected_count
                return people_count
            else:
                # Reset counter if no one detected (people have moved away)
                if detected_count == 0:
                    self.last_entrance_count = 0
                return 0
        
        elif camera_type == "exit":
            # Similar logic for exit
            if detected_count > self.last_exit_count:
                people_count = detected_count - self.last_exit_count
                self.last_exit_count = detected_count
                return people_count
            else:
                if detected_count == 0:
                    self.last_exit_count = 0
                return 0
        
        return 0
    
    def _analyze_demographics(self, people_count: int) -> Dict[str, int]:
        """Analyze demographics of detected people"""
        if people_count == 0:
            return {"adults": 0, "kids": 0, "males": 0, "females": 0}
        
        # Simple heuristic-based demographics (can be enhanced with ML models)
        import random
        random.seed(people_count)  # Consistent results for same count
        
        adults = max(1, int(people_count * random.uniform(0.7, 0.9)))
        kids = people_count - adults
        males = random.randint(0, people_count)
        females = people_count - males
        
        return {
            "adults": adults,
            "kids": kids,
            "males": males,
            "females": females
        }
    
    def _simulate_detection(self, camera_type: str) -> Dict[str, Any]:
        """Fallback simulation when camera is not available"""
        people_count = np.random.randint(0, 6)  # 0-5 people
        demographics = self._analyze_demographics(people_count)
        
        return {
            "camera_type": camera_type,
            "people_count": people_count,
            "raw_detections": people_count,
            "demographics": demographics,
            "frame_captured": False,  # Indicates simulation
            "timestamp": datetime.now().isoformat(),
            "status": "simulated"
        }
    
    def release_cameras(self):
        """Release camera resources"""
        if self.entrance_camera:
            self.entrance_camera.release()
        if self.exit_camera and self.exit_camera != self.entrance_camera:
            self.exit_camera.release()

    def add_log(self, message: str):
        """Add log message - Camera Manager version"""
        log_entry = f"{datetime.now().strftime('%H:%M:%S')} - [CAMERA] {message}"
        print(log_entry)  # Also print to console for debugging
        # If you have access to the parent agent, log there too
        if hasattr(self, 'agent') and self.agent:
            self.agent.add_log(message)

# Simple Autonomous Agent Implementation
class SmartBranchMonitoringAgent:
    """Enhanced autonomous monitoring agent with real camera integration"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.last_result = None
        self.logs = []
        self.camera_manager = CameraManager(config)
        self.camera_initialized = False
        
    def initialize_agent(self) -> bool:
        """Initialize agent and camera systems with enhanced debugging"""
        try:
            self.add_log("=== Starting Agent Initialization ===")
            self.add_log(f"Camera config: entrance={self.config.entrance_camera_index}, exit={self.config.exit_camera_index}")
            self.add_log(f"Single camera mode: {self.config.use_single_camera}")
            
            # Initialize cameras
            camera_init = self.camera_manager.initialize_cameras()
            if camera_init:
                self.camera_initialized = True
                self.add_log("✅ Camera system initialized successfully")
            else:
                self.add_log("⚠️ Camera initialization failed - using simulation mode")
                self.camera_initialized = False
            
            self.add_log("=== Agent Initialization Complete ===")
            return True
            
        except Exception as e:
            error_msg = f"Agent initialization failed: {str(e)}"
            self.add_log(error_msg)
            logging.error(error_msg)
            return False
    
    async def run_monitoring_cycle(self) -> Dict[str, Any]:
        """Enhanced monitoring cycle with real camera integration"""
        try:
            self.add_log("Starting enhanced monitoring cycle")
            
            # Step 1: Collect current data
            current_occupancy = get_current_occupancy()
            self.add_log(f"Current occupancy: {current_occupancy}")
            
            # Step 2: Process entrance camera
            self.add_log("Processing entrance camera...")
            entrance_result = self.camera_manager.capture_and_detect("entrance")
            entrance_count = entrance_result["people_count"]
            
            if entrance_result["frame_captured"]:
                self.add_log(f"Entrance: Detected {entrance_count} people via camera")
            else:
                self.add_log(f"Entrance: Detected {entrance_count} people (simulated)")
            
            # Step 3: Process exit camera
            self.add_log("Processing exit camera...")
            exit_result = self.camera_manager.capture_and_detect("exit")
            exit_count = exit_result["people_count"]
            
            if exit_result["frame_captured"]:
                self.add_log(f"Exit: Detected {exit_count} people via camera")
            else:
                self.add_log(f"Exit: Detected {exit_count} people (simulated)")
            
            # Step 4: Calculate new occupancy
            new_occupancy = max(0, current_occupancy + entrance_count - exit_count)
            self.add_log(f"Occupancy update: {current_occupancy} + {entrance_count} - {exit_count} = {new_occupancy}")
            
            # Step 5: Update database
            demographics = entrance_result["demographics"]  # Use entrance demographics
            
            insert_footfall_data(entrance_count, exit_count, new_occupancy, demographics)
            self.add_log("Database updated successfully")
            
            # Step 6: Analyze capacity
            capacity_usage = (new_occupancy / self.config.max_capacity) * 100
            
            # Step 7: Generate recommendations
            recommended_staff = max(1, (new_occupancy + self.config.staff_ratio - 1) // self.config.staff_ratio)
            
            # Step 8: Check for alerts
            alerts = []
            if capacity_usage >= 90:
                alerts.append(f"CRITICAL: Capacity at {capacity_usage:.1f}%")
                await self.send_alert("High Capacity", f"Capacity at {capacity_usage:.1f}%")
            elif capacity_usage >= 80:
                alerts.append(f"WARNING: Capacity at {capacity_usage:.1f}%")
            
            # Add camera status alerts
            if not entrance_result["frame_captured"] and entrance_result["status"] != "simulated":
                alerts.append("Entrance camera not responding")
            if not exit_result["frame_captured"] and exit_result["status"] != "simulated":
                alerts.append("Exit camera not responding")
            
            # Generate recommendations
            recommendations = [
                f"Recommended staff: {recommended_staff} for current occupancy of {new_occupancy}"
            ]
            
            if capacity_usage > 85:
                recommendations.append("Consider opening additional service counters")
            elif capacity_usage < 20:
                recommendations.append("Consider reducing staff during low occupancy periods")
            
            # Add camera-specific recommendations
            if not self.camera_initialized:
                recommendations.append("Setup physical cameras for more accurate detection")
            
            result = {
                "occupancy": new_occupancy,
                "capacity_usage": capacity_usage,
                "alerts": alerts,
                "recommendations": recommendations,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "entrance_count": entrance_count,
                "exit_count": exit_count,
                "camera_status": {
                    "entrance_active": entrance_result["frame_captured"],
                    "exit_active": exit_result["frame_captured"],
                    "mode": "camera" if self.camera_initialized else "simulation"
                },
                "detection_details": {
                    "entrance": entrance_result,
                    "exit": exit_result
                }
            }
            
            self.last_result = result
            self.add_log(f"Cycle complete - Occupancy: {new_occupancy}, Capacity: {capacity_usage:.1f}%")
            
            return result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.add_log(f"Cycle failed: {str(e)}")
            return error_result
    
    async def send_alert(self, alert_type: str, message: str):
        """Send alert notifications"""
        if not self.config.smtp_user or not self.config.notification_emails:
            return
        
        try:
            subject = f"Smart Branch Alert: {alert_type}"
            body = f"""
Smart Branch Monitoring Alert

Alert Type: {alert_type}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Message: {message}

Detection Mode: {'Camera' if self.camera_initialized else 'Simulation'}

This is an automated alert from the Smart Branch monitoring system.
"""
            if self.config.notification_emails:
                config_dict = {
                    'SMTP_HOST': self.config.smtp_host,
                    'SMTP_PORT': self.config.smtp_port,
                    'SMTP_USER': self.config.smtp_user,
                    'SMTP_PASS': self.config.smtp_pass,
                    'FROM_EMAIL': self.config.smtp_user
                }
                success, msg = send_email(
                    self.config.notification_emails[0],
                    subject,
                    body,
                    config=config_dict
                )
                if success:
                    self.add_log("Alert notification sent")
                
        except Exception as e:
            self.add_log(f"Alert sending failed: {str(e)}")
    
    def add_log(self, message: str):
        """Add log message"""
        log_entry = f"{datetime.now().strftime('%H:%M:%S')} - {message}"
        self.logs.append(log_entry)
        if len(self.logs) > 50:
            self.logs = self.logs[-50:]
        logging.info(message)
    
    def start_autonomous_monitoring(self):
        """Start continuous monitoring with camera integration"""
        # Initialize agent first
        if not self.initialize_agent():
            self.add_log("Failed to initialize agent")
            return
        
        self.is_running = True
        self.add_log(f"Agent started in {'camera' if self.camera_initialized else 'simulation'} mode")
        
        async def monitoring_loop():
            while self.is_running:
                try:
                    await self.run_monitoring_cycle()
                    await asyncio.sleep(self.config.camera_check_interval)
                except Exception as e:
                    self.add_log(f"Monitoring error: {str(e)}")
                    await asyncio.sleep(10)
        
        def run_loop():
            asyncio.run(monitoring_loop())
        
        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
    
    def stop_autonomous_monitoring(self):
        """Stop monitoring and release resources"""
        self.is_running = False
        self.camera_manager.release_cameras()
        self.add_log("Agent stopped and cameras released")

class EnhancedQueueManagementAgent:
    """Enhanced Queue Management Agent with robust A2A communication"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_running = False
        self.logs = []
        self.agent_id = f"queue_agent_{uuid.uuid4().hex[:8]}"
        
        # Initialize enhanced communication system
        self.a2a_comm = EnhancedA2ACommunication(self.agent_id, config)
        
        # Queue statistics
        self.queue_stats = {
            'total_served_today': 0,
            'average_service_time': 5.0,
            'peak_hours': [],
            'bottlenecks': []
        }
    def get_queue_insights(self) -> Dict[str, Any]:
        """Get comprehensive queue insights for the agent dashboard"""
        try:
            # Get current queue data
            current_queue = get_pending_queue()
            queue_history = get_queue_history(7)  # Last 7 days
            
            # Current status metrics
            current_status = {
                'queue_length': len(current_queue),
                'average_wait': current_queue['estimated_wait'].mean() if not current_queue.empty else 0,
                'longest_wait': current_queue['estimated_wait'].max() if not current_queue.empty else 0,
                'oldest_customer_wait': 0,
            }
            
            # Calculate oldest customer wait time
            if not current_queue.empty:
                current_time = datetime.now()
                oldest_join_time = pd.to_datetime(current_queue['join_time']).min()
                current_status['oldest_customer_wait'] = (current_time - oldest_join_time).total_seconds() / 60
            
            # Efficiency metrics
            efficiency_metrics = {
                'total_served_today': self.queue_stats['total_served_today'],
                'average_service_time': self.queue_stats['average_service_time'],
                'completion_rate': 0,
                'throughput_per_hour': 0
            }
            
            # Calculate completion rate and throughput from history
            if not queue_history.empty:
                total_entries = queue_history['total_entries'].sum()
                total_completed = queue_history['completed'].sum()
                efficiency_metrics['completion_rate'] = (total_completed / total_entries * 100) if total_entries > 0 else 0
                efficiency_metrics['throughput_per_hour'] = total_completed / (7 * 24) if total_completed > 0 else 0
            
            # Generate alerts
            alerts = []
            if current_status['queue_length'] > 10:
                alerts.append(f"High queue volume: {current_status['queue_length']} customers waiting")
            
            if current_status['average_wait'] > 30:
                alerts.append(f"Long wait times: {current_status['average_wait']:.1f} minutes average")
            
            if current_status['oldest_customer_wait'] > 60:
                alerts.append(f"Customer waiting over 1 hour: {current_status['oldest_customer_wait']:.1f} minutes")
            
            if efficiency_metrics['completion_rate'] < 80:
                alerts.append(f"Low completion rate: {efficiency_metrics['completion_rate']:.1f}%")
            
            # Generate recommendations
            recommendations = []
            
            if current_status['queue_length'] > 5:
                recommendations.append("Consider opening additional service counters")
            
            if current_status['average_wait'] > 20:
                recommendations.append("Implement queue optimization strategies")
            
            if efficiency_metrics['completion_rate'] < 85:
                recommendations.append("Review service processes to improve completion rate")
            
            if current_status['queue_length'] == 0:
                recommendations.append("Consider reducing staff during low demand periods")
            
            # Peak hours analysis
            peak_hours = ["10:00-11:00", "14:00-15:00", "16:00-17:00"]  # Mock data
            
            # Bottleneck analysis
            bottlenecks = []
            if current_status['average_wait'] > 15:
                bottlenecks.append("Service counter capacity")
            
            if efficiency_metrics['completion_rate'] < 90:
                bottlenecks.append("Service process efficiency")
            
            return {
                'current_status': current_status,
                'efficiency_metrics': efficiency_metrics,
                'alerts': alerts,
                'recommendations': recommendations,
                'peak_hours': peak_hours,
                'bottlenecks': bottlenecks,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.add_log(f"Error generating queue insights: {str(e)}")
            return {
                'current_status': {'queue_length': 0, 'average_wait': 0, 'longest_wait': 0, 'oldest_customer_wait': 0},
                'efficiency_metrics': {'total_served_today': 0, 'average_service_time': 5.0, 'completion_rate': 0, 'throughput_per_hour': 0},
                'alerts': ["Unable to generate insights due to error"],
                'recommendations': ["Check system status and data availability"],
                'peak_hours': [],
                'bottlenecks': [],
                'last_updated': datetime.now().isoformat()
            }
        
    def start_autonomous_monitoring(self) -> bool:
        """Start autonomous monitoring with enhanced communication"""
        try:
            # Start A2A communication system
            if not self.a2a_comm.start_communication_system():
                self.add_log("A2A communication failed to start - continuing in local mode")
            
            self.is_running = True
            self.add_log("Enhanced Queue Management Agent started")
            
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
                    await self.run_queue_analysis_cycle()
                    await asyncio.sleep(60)
                except Exception as e:
                    self.add_log(f"Monitoring error: {str(e)}")
                    await asyncio.sleep(30)
        
        def run_loop():
            import asyncio
            asyncio.run(monitoring_loop())
        
        monitor_thread = threading.Thread(target=run_loop, daemon=True)
        monitor_thread.start()
    
    def send_message_to_app2(self, message: Dict[str, Any]) -> bool:
        """Send message to app2.py using enhanced communication"""
        return self.a2a_comm.send_message_with_retry('/receive_message', message)
    
    def get_communication_status(self) -> Dict[str, Any]:
        """Get A2A communication status"""
        return self.a2a_comm.get_connection_status()
    
    def stop_autonomous_monitoring(self):
        """Stop monitoring and communication"""
        self.is_running = False
        self.a2a_comm.stop_communication_system()
        self.add_log("Enhanced Queue Management Agent stopped")
    
    def add_log(self, message: str):
        """Add log message"""
        log_entry = f"{datetime.now().strftime('%H:%M:%S')} - [ENHANCED_QUEUE_AGENT] {message}"
        self.logs.append(log_entry)
        if len(self.logs) > 50:
            self.logs = self.logs[-50:]
        logging.info(message)
    
    async def run_queue_analysis_cycle(self):
        """Run queue analysis with enhanced communication"""
        # Your existing queue analysis logic here
        # But now use self.send_message_to_app2() for communication
        pass

class EnhancedA2ACommunication:
    """FIXED Enhanced Agent-to-Agent Communication Manager for app1.py"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        # FIXED: Use the correct app2 endpoint from Streamlit
        self.app2_endpoint = config.get('APP2_ENDPOINT', 'http://localhost:8502')
        self.communication_port = config.get('QUEUE_AGENT_PORT', 5001)
        self.is_running = False
        self.message_queue = Queue()
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
            
            # Perform initial handshake with RETRY logic
            handshake_success = False
            for attempt in range(3):  # Try 3 times
                self.add_log(f"Attempting handshake with app2.py (attempt {attempt + 1}/3)")
                if self.perform_handshake():
                    handshake_success = True
                    break
                time.sleep(2)  # Wait 2 seconds between attempts
            
            if handshake_success:
                self.is_running = True
                self.connection_status = "connected"
                self.add_log("A2A communication system started successfully")
                return True
            else:
                self.add_log("Initial handshake failed after 3 attempts - running in degraded mode")
                self.is_running = True
                self.connection_status = "degraded"
                return True
                
        except Exception as e:
            self.add_log(f"Failed to start communication system: {str(e)}")
            return False
    
    def initialize_flask_server(self) -> bool:
        """Initialize Flask server with enhanced endpoints"""
        try:
            self.flask_app = Flask(f'agent_{self.agent_id}')
            
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
                """FIXED Handle handshake from app2.py agents"""
                try:
                    data = request.get_json() or {}
                    agent_id = data.get('agent_id', 'unknown')
                    endpoints = data.get('endpoints', {})
                    message = data.get('message', '')

                    self.add_log(f"Handshake received from {agent_id}")
                    self.add_log(f"   Message: {message}")
                    self.add_log(f"   Endpoints: {list(endpoints.keys())}")

                    # Update connection status
                    self.connection_status = "connected"
                    self.last_heartbeat = datetime.now()

                    return jsonify({
                        'status': 'success',
                        'message': 'Handshake successful - Queue Management Agent ready',
                        'agent_id': self.agent_id,
                        'timestamp': datetime.now().isoformat(),
                        'acknowledged_endpoints': endpoints
                    }), 200

                except Exception as e:
                    self.add_log(f"Error in handshake: {str(e)}")
                    return jsonify({
                        'status': 'error', 
                        'message': str(e),
                        'agent_id': self.agent_id
                    }), 500
                
            @self.flask_app.route('/receive_message', methods=['POST'])
            def receive_message():
                """Enhanced message receiving endpoint"""
                try:
                    data = request.get_json()
                    
                    # Validate message format
                    if not self.validate_message(data):
                        return jsonify({'status': 'error', 'message': 'Invalid message format'}), 400
                    
                    # Add to message queue for processing
                    self.message_queue.put(data)
                    
                    message_type = data.get('type', data.get('message', {}).get('type', 'unknown'))
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
                """Handle heartbeat from other agents"""
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
            
            @self.flask_app.route('/status', methods=['GET'])
            def get_status():
                """Get current agent status"""
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
        """FIXED Validate incoming message format for better compatibility"""
        # More flexible validation to handle different message structures from app2.py
        required_fields = ['from_agent', 'timestamp']
        has_required = all(field in message for field in required_fields)
        
        # Also accept messages with 'type' field directly
        has_type_field = 'type' in message
        
        # Accept messages with nested message structure  
        has_nested_message = 'message' in message and isinstance(message['message'], dict)
        
        return has_required and (has_type_field or has_nested_message)
    
    def perform_handshake(self) -> bool:
        """FIXED Perform initial handshake with app2.py with better error handling"""
        try:
            # First, test if app2.py is reachable
            self.add_log(f"Testing connection to app2.py at {self.app2_endpoint}")
            
            # Check health endpoint first
            try:
                health_response = requests.get(f"{self.app2_endpoint}/health", timeout=5)
                if health_response.status_code == 200:
                    self.add_log("App2.py health check passed")
                else:
                    self.add_log(f"App2.py health check failed: {health_response.status_code}")
                    return False
            except requests.exceptions.ConnectionError:
                self.add_log(f"Cannot connect to app2.py at {self.app2_endpoint} - is it running?")
                return False
            except Exception as e:
                self.add_log(f"Health check error: {str(e)}")
                return False
            
            # Perform handshake
            handshake_data = {
                'agent_id': self.agent_id,
                'message': 'Initial handshake from app1.py Queue Management Agent',
                'timestamp': datetime.now().isoformat(),
                'endpoints': {
                    'health': f'http://localhost:{self.communication_port}/health',
                    'receive_message': f'http://localhost:{self.communication_port}/receive_message',
                    'heartbeat': f'http://localhost:{self.communication_port}/heartbeat'
                }
            }
            
            self.add_log("Sending handshake request...")
            response = requests.post(
                f"{self.app2_endpoint}/handshake",
                json=handshake_data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.add_log("Handshake successful with app2.py")
                response_data = response.json()
                self.add_log(f"App2 response: {response_data.get('message', 'No message')}")
                return True
            else:
                self.add_log(f"Handshake failed: HTTP {response.status_code}")
                self.add_log(f"Response: {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            self.add_log(f"Connection error during handshake - app2.py may not be running on {self.app2_endpoint}")
            return False
        except requests.exceptions.Timeout:
            self.add_log("Handshake timeout - app2.py not responding within 10 seconds")
            return False
        except Exception as e:
            self.add_log(f"Handshake error: {str(e)}")
            return False
    
    def send_message_with_retry(self, endpoint: str, message: Dict[str, Any]) -> bool:
        """FIXED Send message with retry logic and enhanced debugging"""
        
        # Add required fields if missing
        message.setdefault('message_id', str(uuid.uuid4()))
        message.setdefault('from_agent', self.agent_id)
        message.setdefault('timestamp', datetime.now().isoformat())

        url = f"{self.app2_endpoint}{endpoint}"
        
        # ENHANCED DEBUG
        self.add_log(f"Attempting to send message to: {url}")
        self.add_log(f"   Message type: {message.get('type', 'unknown')}")
        self.add_log(f"   Message ID: {message.get('message_id', 'unknown')}")
        
        # Test connection first
        try:
            test_response = requests.get(f"{self.app2_endpoint}/health", timeout=2)
            if test_response.status_code != 200:
                self.add_log(f"App2 health check failed: {test_response.status_code}")
                return False
            else:
                self.add_log(f"App2 health check passed")
        except Exception as e:
            self.add_log(f"Cannot reach app2.py at {self.app2_endpoint}/health: {str(e)}")
            return False

        for attempt in range(self.max_retries):
            try:
                self.add_log(f"Sending message (attempt {attempt + 1}/{self.max_retries})")
                response = requests.post(url, json=message, timeout=5)

                self.add_log(f"Response received: {response.status_code}")
                if response.text:
                    self.add_log(f"   Response body: {response.text[:200]}")

                if response.status_code == 200:
                    self.connection_status = "connected"
                    self.retry_count = 0
                    self.add_log(f"Message sent successfully to {endpoint}")
                    return True
                else:
                    self.add_log(f"Message failed with status {response.status_code}")

            except requests.exceptions.ConnectionError as e:
                self.add_log(f"Connection error (attempt {attempt + 1}): {str(e)}")
            except requests.exceptions.Timeout as e:
                self.add_log(f"Timeout error (attempt {attempt + 1}): {str(e)}")
            except Exception as e:
                self.add_log(f"Send error (attempt {attempt + 1}): {str(e)}")

            # Retry with exponential backoff
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                self.add_log(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

        self.connection_status = "disconnected"
        self.retry_count += 1
        self.add_log(f"All {self.max_retries} attempts failed")
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
        """FIXED Handle incoming messages from app2.py agents with better format handling"""
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
        
        # Handle different message types from app2.py
        if message_type == 'customer_joined':
            self.handle_customer_joined_message(message_data)
        elif message_type == 'customer_left':
            self.handle_customer_left_message(message_data)
        elif message_type == 'customer_feedback':
            self.handle_customer_feedback_message(message_data)
        elif message_type == 'queue_prediction':
            self.handle_queue_prediction_message(message_data)
        elif message_type == 'test_connection':
            self.handle_test_connection_message(message_data)
        else:
            self.add_log(f"Unknown message type: {message_type}")
    
    def handle_customer_joined_message(self, message: Dict[str, Any]):
        """Handle customer joined messages from app2.py"""
        customer_data = message.get('data', {}).get('customer', {})
        customer_name = customer_data.get('name', 'Unknown')
        user_id = customer_data.get('user_id', 'Unknown')
        
        self.add_log(f"Customer joined: {customer_name} (ID: {user_id})")
    
    def handle_customer_left_message(self, message: Dict[str, Any]):
        """Handle customer left messages from app2.py"""
        customer_data = message.get('data', {})
        customer_id = customer_data.get('customer_id', 'Unknown')
        
        self.add_log(f"Customer left queue: {customer_id}")
    
    def handle_customer_feedback_message(self, message: Dict[str, Any]):
        """Handle customer feedback messages from app2.py"""
        feedback_data = message.get('data', {}).get('feedback', {})
        rating = feedback_data.get('rating', 0)
        user_id = feedback_data.get('user_id', 'Unknown')
        
        self.add_log(f"Feedback received: {rating}/5 stars from user {user_id}")
    
    def handle_queue_prediction_message(self, message: Dict[str, Any]):
        """Handle queue prediction messages from app2.py"""
        prediction_data = message.get('data', {})
        prediction = prediction_data.get('prediction', {})
        wait_time = prediction.get('predicted_wait_time', 0)
        
        self.add_log(f"Queue prediction received: {wait_time} minutes")
    
    def handle_test_connection_message(self, message: Dict[str, Any]):
        """FIXED Handle test connection messages and send response"""
        test_data = message.get('data', {}) if 'data' in message else message
        test_id = test_data.get('test_id', 'unknown')
        
        self.add_log(f"Test connection received: {test_id}")
        
        # Send response back to app2.py
        response_message = {
            'type': 'test_response',
            'data': {
                'test_id': test_id,
                'response_time': datetime.now().isoformat(),
                'message': 'Test connection successful from Queue Management Agent'
            }
        }
        
        # Send the response
        self.send_response_to_app2(response_message)
    
    def send_response_to_app2(self, response_message: Dict[str, Any]) -> bool:
        """Send response message back to app2.py"""
        try:
            # Ensure message has required fields
            response_message.setdefault('message_id', str(uuid.uuid4()))
            response_message.setdefault('from_agent', self.agent_id)
            response_message.setdefault('timestamp', datetime.now().isoformat())
            
            # Send to app2.py
            response = requests.post(
                f"{self.app2_endpoint}/receive_message",
                json=response_message,
                timeout=5
            )
            
            if response.status_code == 200:
                self.add_log(f"Response sent to app2.py: {response_message.get('type', 'unknown')}")
                return True
            else:
                self.add_log(f"Failed to send response to app2.py: {response.status_code}")
                return False
                
        except Exception as e:
            self.add_log(f"Error sending response to app2.py: {str(e)}")
            return False
    
    def start_heartbeat_monitor(self):
        """Start heartbeat monitoring"""
        def heartbeat_monitor():
            while self.is_running:
                try:
                    # Send heartbeat to app2.py
                    heartbeat_data = {
                        'agent_id': self.agent_id,
                        'status': 'alive',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    response = requests.post(
                        f"{self.app2_endpoint}/heartbeat",
                        json=heartbeat_data,
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        if self.connection_status == "disconnected":
                            self.connection_status = "connected"
                            self.add_log("Connection restored with app2.py")
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
    
    def check_app2_health(self) -> bool:
        """Check if app2.py is healthy"""
        try:
            response = requests.get(f"{self.app2_endpoint}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            'status': self.connection_status,
            'app2_healthy': self.check_app2_health(),
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


# Configure page
st.set_page_config(
    page_title="Smart Branch Admin Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
def load_config():
    """FIXED load configuration with correct A2A communication settings"""
    config = {
        'CAPACITY_MAX': int(os.getenv('CAPACITY_MAX', '200')),
        'STAFF_PER_N': int(os.getenv('STAFF_PER_N', '25')),
        'ENABLE_DEMOGRAPHICS': os.getenv('ENABLE_DEMOGRAPHICS', 'true').lower() == 'true',
        'DB_PATH': os.getenv('DB_PATH', './data/database.db'),
        'SMTP_HOST': os.getenv('SMTP_HOST', ''),
        'SMTP_PORT': int(os.getenv('SMTP_PORT', '465')),
        'SMTP_USER': os.getenv('SMTP_USER', ''),
        'SMTP_PASS': os.getenv('SMTP_PASS', ''),
        'FROM_EMAIL': os.getenv('FROM_EMAIL', ''),
        'ENABLE_NOTIFICATIONS': os.getenv('ENABLE_NOTIFICATIONS', 'true').lower() == 'true',
        'ENABLE_AUTONOMOUS_AGENT': os.getenv('ENABLE_AUTONOMOUS_AGENT', 'true').lower() == 'true',
        'AGENT_MONITORING_INTERVAL': int(os.getenv('AGENT_MONITORING_INTERVAL', '30')),
        'ENTRANCE_CAMERA_INDEX': int(os.getenv('ENTRANCE_CAMERA_INDEX', '0')),
        'EXIT_CAMERA_INDEX': int(os.getenv('EXIT_CAMERA_INDEX', '1')), 
        'USE_SINGLE_CAMERA': os.getenv('USE_SINGLE_CAMERA', 'false').lower() == 'true',
        'CAMERA_FALLBACK_MODE': os.getenv('CAMERA_FALLBACK_MODE', 'simulation').lower(),
        
        # FIXED A2A Communication settings - app1.py should connect to app2.py
        'APP2_ENDPOINT': os.getenv('APP2_ENDPOINT', 'http://localhost:8502'),  # FIXED: Changed from 5002 to 8502 (Streamlit default)
        'QUEUE_AGENT_PORT': int(os.getenv('QUEUE_AGENT_PORT', '5001')),
        'ENABLE_A2A_COMMUNICATION': os.getenv('ENABLE_A2A_COMMUNICATION', 'true').lower() == 'true'
    }
    
    # DEBUG: Print configuration
    print("=== APP1.PY A2A CONFIG DEBUG ===")
    print(f"APP2_ENDPOINT: {config['APP2_ENDPOINT']}")
    print(f"QUEUE_AGENT_PORT: {config['QUEUE_AGENT_PORT']}")
    print(f"ENABLE_A2A_COMMUNICATION: {config['ENABLE_A2A_COMMUNICATION']}")
    print("================================")
    
    return config
    
   
# Email function
def send_email(to_email, subject, body, attachment_path=None, config=None):
    """Send email with optional attachment"""
    if not config or not config['SMTP_USER'] or not config['SMTP_PASS']:
        return False, "Email configuration missing"
    
    try:
        msg = MIMEMultipart()
        msg['From'] = config['FROM_EMAIL']
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(attachment_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
            msg.attach(part)
        
        # Use SSL for port 465
        if config['SMTP_PORT'] == 465:
            server = smtplib.SMTP_SSL(config['SMTP_HOST'], config['SMTP_PORT'])
        else:
            server = smtplib.SMTP(config['SMTP_HOST'], config['SMTP_PORT'])
            server.starttls()
        
        server.login(config['SMTP_USER'], config['SMTP_PASS'])
        server.sendmail(config['FROM_EMAIL'], to_email, msg.as_string())
        server.quit()
        
        return True, "Email sent successfully"
        
    except Exception as e:
        return False, f"Email sending failed: {str(e)}"

# Database setup
def init_database():
    """Initialize SQLite database"""
    db_path = Path("data")
    db_path.mkdir(exist_ok=True)
    
    conn = sqlite3.connect('data/database.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS footfall (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            entrance_count INTEGER DEFAULT 0,
            exit_count INTEGER DEFAULT 0,
            occupancy INTEGER DEFAULT 0,
            adults INTEGER DEFAULT 0,
            kids INTEGER DEFAULT 0,
            males INTEGER DEFAULT 0,
            females INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    
    conn.commit()
    conn.close()

# Queue management functions
def get_pending_queue():
    """Get all pending queue entries with customer details"""
    try:
        conn = sqlite3.connect('data/database.db')
        df = pd.read_sql_query('''
            SELECT 
                q.id as queue_id,
                q.position,
                q.estimated_wait,
                q.join_time,
                q.status,
                u.id as user_id,
                u.name,
                u.email,
                u.phone,
                u.age,
                u.gender
            FROM queue_entries q
            JOIN users u ON q.user_id = u.id
            WHERE q.status = 'waiting'
            ORDER BY q.join_time ASC
        ''', conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

def update_queue_positions():
    """Update queue positions after changes"""
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        
        # Get all waiting entries ordered by join time
        cursor.execute('''
            SELECT id FROM queue_entries 
            WHERE status = 'waiting' 
            ORDER BY join_time ASC
        ''')
        
        queue_entries = cursor.fetchall()
        
        # Update positions
        for i, (entry_id,) in enumerate(queue_entries, 1):
            estimated_wait = i * 5  # 5 minutes per position
            cursor.execute('''
                UPDATE queue_entries 
                SET position = ?, estimated_wait = ?
                WHERE id = ?
            ''', (i, estimated_wait, entry_id))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error updating queue positions: {e}")
        return False

def remove_from_queue(queue_id, reason="removed"):
    """Remove a customer from queue"""
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        
        # Update status
        cursor.execute('''
            UPDATE queue_entries 
            SET status = ?, actual_wait = (
                CASE 
                    WHEN status = 'waiting' THEN 
                        ROUND((julianday('now') - julianday(join_time)) * 24 * 60)
                    ELSE actual_wait
                END
            )
            WHERE id = ?
        ''', (reason, queue_id))
        
        conn.commit()
        conn.close()
        
        # Update positions for remaining customers
        update_queue_positions()
        return True
        
    except Exception as e:
        st.error(f"Error removing from queue: {e}")
        return False

def mark_queue_served(queue_id):
    """Mark a customer as served"""
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        
        # Update status to completed and calculate actual wait time
        cursor.execute('''
            UPDATE queue_entries 
            SET status = 'completed',
                actual_wait = ROUND((julianday('now') - julianday(join_time)) * 24 * 60)
            WHERE id = ?
        ''', (queue_id,))
        
        conn.commit()
        conn.close()
        
        # Update positions for remaining customers
        update_queue_positions()
        return True
        
    except Exception as e:
        st.error(f"Error marking as served: {e}")
        return False

def get_queue_history(days=7):
    """Get queue history for analytics"""
    try:
        conn = sqlite3.connect('data/database.db')
        df = pd.read_sql_query('''
            SELECT 
                DATE(q.join_time) as date,
                COUNT(q.id) as total_entries,
                COUNT(CASE WHEN q.status = 'completed' THEN 1 END) as completed,
                COUNT(CASE WHEN q.status = 'removed' THEN 1 END) as removed,
                COUNT(CASE WHEN q.status = 'left' THEN 1 END) as left_queue,
                AVG(CASE WHEN q.actual_wait IS NOT NULL THEN q.actual_wait END) as avg_actual_wait,
                AVG(q.estimated_wait) as avg_estimated_wait
            FROM queue_entries q
            WHERE q.join_time >= date('now', '-{} days')
            GROUP BY DATE(q.join_time)
            ORDER BY date DESC
        '''.format(days), conn)
        
        conn.close()
        return df
    except Exception as e:
        st.error(f"Queue history error: {e}")
        return pd.DataFrame()

def send_queue_notification(user_email, user_name, message_type, config):
    """Send notifications to queue customers"""
    if not config['ENABLE_NOTIFICATIONS'] or not user_email:
        return False, "Notifications disabled or no email"
    
    messages = {
        'next': f"Hi {user_name}, you're next in line! Please proceed to the counter.",
        'removed': f"Hi {user_name}, you've been removed from the queue. Please contact us if you have questions.",
        'completed': f"Hi {user_name}, thank you for your visit! Your service has been completed."
    }
    
    subjects = {
        'next': "You're Next - Smart Branch",
        'removed': "Queue Update - Smart Branch", 
        'completed': "Service Completed - Smart Branch"
    }
    
    return send_email(
        user_email,
        subjects.get(message_type, "Queue Update - Smart Branch"),
        messages.get(message_type, "Queue status updated."),
        config=config
    )

# Customer data functions (keeping existing functions)
def get_all_customers():
    """Get all registered customers"""
    try:
        conn = sqlite3.connect('data/database.db')
        df = pd.read_sql_query('''
            SELECT 
                u.id,
                u.name,
                u.email,
                u.phone,
                u.age,
                u.gender,
                u.created_at,
                COUNT(q.id) as queue_visits,
                MAX(q.join_time) as last_visit,
                AVG(f.rating) as avg_rating
            FROM users u
            LEFT JOIN queue_entries q ON u.id = q.user_id
            LEFT JOIN feedback f ON u.id = f.user_id
            GROUP BY u.id
            ORDER BY u.created_at DESC
        ''', conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

def get_current_queue():
    """Get current queue status with customer details"""
    try:
        conn = sqlite3.connect('data/database.db')
        df = pd.read_sql_query('''
            SELECT 
                q.id,
                q.position,
                q.estimated_wait,
                q.join_time,
                u.name,
                u.email,
                u.phone,
                u.age
            FROM queue_entries q
            JOIN users u ON q.user_id = u.id
            WHERE q.status = 'waiting'
            ORDER BY q.join_time ASC
        ''', conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

def export_customers_to_csv():
    """Export customer data to CSV"""
    try:
        customers_df = get_all_customers()
        if customers_df.empty:
            return None, "No customer data found"
        
        # Create export directory
        export_dir = Path("data/exports")
        export_dir.mkdir(exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"customers_{timestamp}.csv"
        filepath = export_dir / filename
        
        # Save CSV
        customers_df.to_csv(filepath, index=False)
        
        return filepath, f"Customer data exported successfully to {filename}"
        
    except Exception as e:
        return None, f"Export failed: {str(e)}"

def get_queue_analytics():
    """Get queue analytics data"""
    try:
        conn = sqlite3.connect('data/database.db')
        
        # Get queue statistics for the last 30 days
        df = pd.read_sql_query('''
            SELECT 
                DATE(q.join_time) as date,
                COUNT(q.id) as total_entries,
                AVG(q.estimated_wait) as avg_wait_time,
                COUNT(CASE WHEN q.status = 'completed' THEN 1 END) as completed_services
            FROM queue_entries q
            WHERE q.join_time >= date('now', '-30 days')
            GROUP BY DATE(q.join_time)
            ORDER BY date DESC
        ''', conn)
        
        conn.close()
        return df
    except Exception as e:
        st.error(f"Queue analytics error: {e}")
        return pd.DataFrame()

# FIXED: Improved people detection functions
@st.cache_resource
def load_face_detector():
    """Load OpenCV face detector with better parameters"""
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_people_simple(frame, face_detector):
    """Improved people detection using face detection as proxy"""
    if frame is None:
        return []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Improved parameters for better multiple face detection
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,        # How much the image size is reduced at each scale
        minNeighbors=4,         # How many neighbors each candidate rectangle should have to retain it
        minSize=(30, 30),       # Minimum possible face size
        maxSize=(300, 300),     # Maximum possible face size
        flags=cv2.CASCADE_SCALE_IMAGE  # Can be used to improve detection
    )
    
    # Filter overlapping detections (remove duplicates)
    if len(faces) > 1:
        filtered_faces = []
        for i, (x1, y1, w1, h1) in enumerate(faces):
            is_duplicate = False
            for j, (x2, y2, w2, h2) in enumerate(faces):
                if i != j:
                    # Calculate overlap
                    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    overlap_area = overlap_x * overlap_y
                    
                    area1 = w1 * h1
                    area2 = w2 * h2
                    
                    # If overlap is more than 30% of either face, consider it duplicate
                    if overlap_area > 0.3 * min(area1, area2):
                        # Keep the larger detection
                        if area1 < area2:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                filtered_faces.append((x1, y1, w1, h1))
        
        return filtered_faces
    
    return faces

def classify_demographics_mock(faces):
    """Improved mock demographics classification"""
    total_faces = len(faces)
    if total_faces == 0:
        return {"adults": 0, "kids": 0, "males": 0, "females": 0, "total_detected": 0}
    
    # Improved heuristic based on face size and position
    adults = 0
    kids = 0
    
    # Calculate average face size for reference
    if total_faces > 0:
        avg_face_area = sum(w * h for (x, y, w, h) in faces) / total_faces
        
        for (x, y, w, h) in faces:
            face_area = w * h
            face_ratio = w / h if h > 0 else 1
            
            # More sophisticated classification
            # Kids typically have smaller faces and different proportions
            if face_area < 0.7 * avg_face_area and face_ratio < 1.2:
                kids += 1
            else:
                adults += 1
    else:
        # Fallback: assume all are adults if no size analysis possible
        adults = total_faces
    
    # Gender distribution (still mock, but more varied)
    import random
    random.seed(total_faces)  # Consistent results for same number of faces
    
    male_ratio = 0.4 + random.random() * 0.4  # Between 40-80% male
    males = int(total_faces * male_ratio)
    females = total_faces - males
    
    return {
        "adults": adults, 
        "kids": kids, 
        "males": males, 
        "females": females,
        "total_detected": total_faces  # Add this for debugging
    }

def draw_detections_on_image(image_array, faces):
    """Draw rectangles around detected faces for visualization"""
    # Convert to BGR if needed
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        display_image = image_array.copy()
        # Convert RGB to BGR for OpenCV
        display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
    else:
        display_image = image_array.copy()
    
    # Draw rectangles around faces
    for i, (x, y, w, h) in enumerate(faces):
        # Different colors for different faces
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        color = colors[i % len(colors)]
        
        cv2.rectangle(display_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(display_image, f'Person {i+1}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Convert back to RGB for display
    display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
    return display_image

# Analytics functions (keeping existing functions)
def get_current_occupancy():
    """Get current occupancy from database"""
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT occupancy FROM footfall ORDER BY timestamp DESC LIMIT 1')
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 0
    except:
        return 0

def insert_footfall_data(entrance_count, exit_count, occupancy, demographics):
    """Insert footfall data into database"""
    try:
        conn = sqlite3.connect('data/database.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO footfall (entrance_count, exit_count, occupancy, adults, kids, males, females)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (entrance_count, exit_count, occupancy, 
              demographics['adults'], demographics['kids'], 
              demographics['males'], demographics['females']))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

def get_analytics_data(days_back=7):
    """Get analytics data from database"""
    try:
        conn = sqlite3.connect('data/database.db')
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        df = pd.read_sql_query('''
            SELECT * FROM footfall 
            WHERE timestamp >= ? 
            ORDER BY timestamp
        ''', conn, params=[start_date])
        
        conn.close()
        return df
    except:
        # Return sample data if no database
        dates = pd.date_range(start=datetime.now()-timedelta(days=days_back), 
                             end=datetime.now(), freq='H')
        return pd.DataFrame({
            'timestamp': dates,
            'occupancy': np.random.randint(10, 150, len(dates)),
            'entrance_count': np.random.randint(0, 20, len(dates)),
            'exit_count': np.random.randint(0, 20, len(dates)),
            'adults': np.random.randint(5, 100, len(dates)),
            'kids': np.random.randint(0, 30, len(dates))
        })

def recommend_staff(occupancy, staff_per_n=25):
    """Recommend staff based on occupancy"""
    if occupancy <= 0:
        return 1
    return max(1, (occupancy + staff_per_n - 1) // staff_per_n)

def get_weekend_uplift(df):
    """Calculate weekend vs weekday uplift"""
    if df.empty:
        return {"weekday_avg": 0, "weekend_avg": 0, "uplift": 0}
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format='mixed')
    df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
    
    weekday_avg = df[~df['is_weekend']]['occupancy'].mean()
    weekend_avg = df[df['is_weekend']]['occupancy'].mean()
    
    uplift = ((weekend_avg / weekday_avg) - 1) * 100 if weekday_avg > 0 else 0
    
    return {
        "weekday_avg": weekday_avg,
        "weekend_avg": weekend_avg,
        "uplift": uplift
    }

def create_enhanced_queue_agent(config):
    """FIXED Factory function to create enhanced queue agent with correct endpoints"""
    enhanced_config = config.copy()
    
    # FIXED: Ensure correct endpoint configuration
    enhanced_config['APP2_ENDPOINT'] = config.get('APP2_ENDPOINT', 'http://localhost:8502')  # Where app2.py runs (Streamlit)
    
    print(f"Creating queue agent with APP2_ENDPOINT: {enhanced_config['APP2_ENDPOINT']}")
    print(f"This agent will run on port: {enhanced_config['QUEUE_AGENT_PORT']}")
    
    return EnhancedQueueManagementAgent(enhanced_config)

# Streamlit UI
def main():
    st.title("🏢 Smart Branch Admin Dashboard")
    
    # Initialize database
    init_database()
    config = load_config()
    
    # Sidebar navigation - ADD THE AGENT TAB HERE
    st.sidebar.title("Navigation")
    tab_selection = st.sidebar.radio(
        "Select Dashboard",
        [
            "🔴 Live Monitor", 
            "🤖 Autonomous Agent",  # <-- ADD THIS LINE
            "👥 Customer Management", 
            "🎯 Queue Management", 
            "📊 Queue Analytics", 
            "📈 Analytics", 
            "👥 Staffing", 
            "📤 Export", 
            "📧 Email Test"
        ]
    )
    
    if tab_selection == "🔴 Live Monitor":
        live_monitor_tab(config)
    elif tab_selection == "🤖 Autonomous Agent":  # <-- ADD THIS CONDITION
        autonomous_agent_tab(config)
    elif tab_selection == "👥 Customer Management":
        customer_management_tab(config)
    elif tab_selection == "🎯 Queue Management":
        queue_management_tab(config)
    elif tab_selection == "📊 Queue Analytics":
        queue_analytics_tab()
    elif tab_selection == "📈 Analytics":
        analytics_tab()
    elif tab_selection == "👥 Staffing":
        staffing_tab(config)
    elif tab_selection == "📤 Export":
        export_tab()
    elif tab_selection == "📧 Email Test":
        email_test_tab(config)

def autonomous_agent_tab(config):
    """Autonomous monitoring agent dashboard with enhanced camera detection"""
    st.header("🤖 Autonomous Monitoring Agent")
    
    # Camera Detection Section - NEW ADDITION
    st.subheader("📹 Camera Detection & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Detect Available Cameras"):
            # Detect available cameras
            available_cameras = []
            for index in range(5):
                try:
                    cap = cv2.VideoCapture(index)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            available_cameras.append(index)
                            st.success(f"✅ Camera {index}: Available ({frame.shape[1]}x{frame.shape[0]})")
                        else:
                            st.warning(f"⚠️ Camera {index}: Opens but no frame")
                        cap.release()
                    else:
                        st.error(f"❌ Camera {index}: Cannot open")
                except Exception as e:
                    st.error(f"❌ Camera {index}: Error - {str(e)}")
            
            if len(available_cameras) >= 2:
                st.success(f"🎉 Found {len(available_cameras)} cameras - Dual camera mode possible!")
                st.info(f"Recommended: Entrance={available_cameras[0]}, Exit={available_cameras[1]}")
            elif len(available_cameras) == 1:
                st.warning(f"⚠️ Found only 1 camera - Will use single camera mode")
                st.info(f"Camera index: {available_cameras[0]}")
            else:
                st.error("❌ No cameras detected - Will use simulation mode")
    
    with col2:
        # Camera Configuration Display
        with st.expander("🔧 Camera Configuration", expanded=False):
            st.write("**Current Settings:**")
            st.write(f"- Entrance Camera: Index {config.get('ENTRANCE_CAMERA_INDEX', 0)}")
            st.write(f"- Exit Camera: Index {config.get('EXIT_CAMERA_INDEX', 1)}")
            st.write(f"- Single Camera Mode: {'✅' if config.get('USE_SINGLE_CAMERA', False) else '❌'}")
            
            st.write("**To Configure Cameras:**")
            st.code("""
# Add to your .env file:
ENTRANCE_CAMERA_INDEX=0
EXIT_CAMERA_INDEX=1  
USE_SINGLE_CAMERA=false
CAMERA_FALLBACK_MODE=simulation
            """)
    
    # Initialize session state for agent
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'agent_running' not in st.session_state:
        st.session_state.agent_running = False
    if 'agent_logs' not in st.session_state:
        st.session_state.agent_logs = []
    
    # Agent configuration
    st.subheader("⚙️ Agent Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        monitoring_interval = st.number_input(
            "Monitoring Interval (seconds)", 
            min_value=10, 
            max_value=300, 
            value=30,
            help="How often the agent checks cameras"
        )
    
    with col2:
        capacity_threshold = st.slider(
            "Alert Threshold (%)", 
            min_value=50, 
            max_value=95, 
            value=80,
            help="Capacity percentage that triggers alerts"
        )
    
    with col3:
        # UPDATED: More granular camera options
        camera_mode = st.selectbox(
            "Camera Mode",
            options=["auto_detect", "single_camera", "dual_camera", "simulation"],
            index=0,
            help="Choose camera configuration mode"
        )
        
        # Set use_single_camera based on selection
        if camera_mode == "single_camera":
            use_single_camera = True
        elif camera_mode == "dual_camera":
            use_single_camera = False
        else:  # auto_detect or simulation
            use_single_camera = config.get('USE_SINGLE_CAMERA', False)
    
    # Email configuration status
    st.subheader("📧 Notification Configuration")
    
    email_configured = bool(config.get('SMTP_USER') and config.get('SMTP_PASS'))
    
    if email_configured:
        st.success("✅ Email notifications configured")
        st.write(f"SMTP Server: {config.get('SMTP_HOST', 'Not set')}")
        st.write(f"From Email: {config.get('SMTP_USER', 'Not set')}")
    else:
        st.warning("⚠️ Email notifications not configured")
        st.write("Configure email settings in .env file to receive alerts")
        
        with st.expander("📋 Email Setup Guide"):
            st.write("""
            To enable email notifications, add these to your .env file:
            ```
            SMTP_HOST=smtp.gmail.com
            SMTP_PORT=465
            SMTP_USER=your-email@gmail.com
            SMTP_PASS=your-app-password
            FROM_EMAIL=your-email@gmail.com
            ```
            
            For Gmail:
            1. Enable 2-factor authentication
            2. Generate an app password
            3. Use the app password (not your regular password)
            """)
    
    # Agent Status
    st.subheader("🔄 Agent Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.agent_running:
            st.success("🟢 Agent Running")
        else:
            st.error("🔴 Agent Stopped")
    
    with col2:
        if st.session_state.agent:
            camera_mode_display = "Camera" if st.session_state.agent.camera_initialized else "Simulation"
            st.info(f"Mode: {camera_mode_display}")
        else:
            st.info("Mode: Not initialized")
    
    with col3:
        current_occupancy = get_current_occupancy()
        st.metric("Current Occupancy", current_occupancy)
    
    # Control buttons
    st.subheader("🎛️ Agent Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 Start Agent", type="primary", disabled=st.session_state.agent_running):
            try:
                # UPDATED: Create monitoring config with enhanced camera settings
                monitoring_config = MonitoringConfig(
                    camera_check_interval=monitoring_interval,
                    max_capacity=config.get('CAPACITY_MAX', 200),
                    alert_threshold=capacity_threshold / 100,
                    staff_ratio=config.get('STAFF_PER_N', 25),
                    smtp_host=config.get('SMTP_HOST', ''),
                    smtp_port=config.get('SMTP_PORT', 465),
                    smtp_user=config.get('SMTP_USER', ''),
                    smtp_pass=config.get('SMTP_PASS', ''),
                    notification_emails=[config.get('SMTP_USER', '')] if config.get('SMTP_USER') else [],
                    entrance_camera_index=config.get('ENTRANCE_CAMERA_INDEX', 0),
                    exit_camera_index=config.get('EXIT_CAMERA_INDEX', 1),
                    use_single_camera=use_single_camera
                )
                
                # Create and start agent
                st.session_state.agent = SmartBranchMonitoringAgent(monitoring_config)
                st.session_state.agent.start_autonomous_monitoring()
                st.session_state.agent_running = True
                
                st.success("🚀 Agent started successfully!")
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed to start agent: {str(e)}")
    
    with col2:
        if st.button("⏹️ Stop Agent", disabled=not st.session_state.agent_running):
            try:
                if st.session_state.agent:
                    st.session_state.agent.stop_autonomous_monitoring()
                st.session_state.agent_running = False
                st.session_state.agent = None
                
                st.success("⏹️ Agent stopped successfully!")
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed to stop agent: {str(e)}")
    
    with col3:
        if st.button("🔄 Run Single Cycle", disabled=st.session_state.agent_running):
            try:
                if not st.session_state.agent:
                    # UPDATED: Include camera configuration for single cycle
                    monitoring_config = MonitoringConfig(
                        camera_check_interval=monitoring_interval,
                        max_capacity=config.get('CAPACITY_MAX', 200),
                        alert_threshold=capacity_threshold / 100,
                        staff_ratio=config.get('STAFF_PER_N', 25),
                        smtp_host=config.get('SMTP_HOST', ''),
                        smtp_port=config.get('SMTP_PORT', 465),
                        smtp_user=config.get('SMTP_USER', ''),
                        smtp_pass=config.get('SMTP_PASS', ''),
                        notification_emails=[config.get('SMTP_USER', '')] if config.get('SMTP_USER') else [],
                        entrance_camera_index=config.get('ENTRANCE_CAMERA_INDEX', 0),
                        exit_camera_index=config.get('EXIT_CAMERA_INDEX', 1),
                        use_single_camera=use_single_camera
                    )
                    st.session_state.agent = SmartBranchMonitoringAgent(monitoring_config)
                
                # Run single cycle
                import asyncio
                result = asyncio.run(st.session_state.agent.run_monitoring_cycle())
                
                if result["status"] == "success":
                    st.success("✅ Single cycle completed!")
                    
                    # Show results
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Occupancy", result.get("occupancy", 0))
                    with col_b:
                        st.metric("Capacity Usage", f"{result.get('capacity_usage', 0):.1f}%")
                    with col_c:
                        st.metric("Alerts", len(result.get("alerts", [])))
                    
                    if result.get("alerts"):
                        st.warning("⚠️ Alerts: " + ", ".join(result["alerts"]))
                    
                    if result.get("recommendations"):
                        st.info("💡 Recommendations: " + ", ".join(result["recommendations"]))
                else:
                    st.error(f"Cycle failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Failed to run cycle: {str(e)}")
    
    with col4:
        if st.button("📋 Clear Logs"):
            if st.session_state.agent:
                st.session_state.agent.logs = []
            st.session_state.agent_logs = []
            st.success("📋 Logs cleared!")

    col5 = st.columns(1)[0]  # Create a 5th column
    with col5:
        if st.button("🧪 Test Direct Connection"):
            try:
                response = requests.get("http://localhost:5002/health", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Can reach app2.py")
                    st.write(response.json())
                else:
                    st.error(f"❌ App2.py returned {response.status_code}")
            except Exception as e:
                st.error(f"❌ Cannot reach app2.py: {str(e)}")
    # Real-time monitoring display
    if st.session_state.agent_running and st.session_state.agent:
        st.subheader("📊 Real-time Monitoring")
        
        # Auto-refresh every 10 seconds when agent is running
        placeholder = st.empty()
        
        with placeholder.container():
            if st.session_state.agent.last_result:
                result = st.session_state.agent.last_result
                
                # Current metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Occupancy", 
                        result.get("occupancy", 0),
                        delta=result.get("entrance_count", 0) - result.get("exit_count", 0)
                    )
                
                with col2:
                    capacity_usage = result.get("capacity_usage", 0)
                    st.metric("Capacity Usage", f"{capacity_usage:.1f}%")
                
                with col3:
                    st.metric("Entrance Count", result.get("entrance_count", 0))
                
                with col4:
                    st.metric("Exit Count", result.get("exit_count", 0))
                
                # Alerts section
                if result.get("alerts"):
                    st.subheader("⚠️ Active Alerts")
                    for alert in result["alerts"]:
                        st.warning(alert)
                
                # Recommendations section
                if result.get("recommendations"):
                    st.subheader("💡 Recommendations")
                    for recommendation in result["recommendations"]:
                        st.info(recommendation)
                
                # ENHANCED: Camera status with more detail
                camera_status = result.get("camera_status", {})
                if camera_status:
                    st.subheader("📹 Camera Status")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        entrance_status = "🟢 Active" if camera_status.get("entrance_active") else "🔴 Inactive"
                        st.write(f"Entrance Camera: {entrance_status}")
                    
                    with col2:
                        exit_status = "🟢 Active" if camera_status.get("exit_active") else "🔴 Inactive"
                        st.write(f"Exit Camera: {exit_status}")
                    
                    with col3:
                        mode = camera_status.get("mode", "unknown")
                        st.write(f"Mode: {mode.title()}")
        
        # Auto-refresh when agent is running
        time.sleep(5)
        st.rerun()
    
    # Agent logs
    st.subheader("📝 Agent Logs")
    
    if st.session_state.agent and st.session_state.agent.logs:
        log_container = st.container()
        with log_container:
            # Show last 20 log entries
            for log_entry in st.session_state.agent.logs[-20:]:
                st.text(log_entry)
    else:
        st.info("No logs available. Start the agent to see monitoring activity.")
    
    # Performance metrics
    if st.session_state.agent_running:
        st.subheader("📈 Performance Metrics")
        
        # Get recent analytics data for charts
        df = get_analytics_data(1)  # Last day
        
        if not df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Occupancy trend
                fig_occupancy = px.line(
                    df.tail(20), 
                    x='timestamp', 
                    y='occupancy',
                    title="Recent Occupancy Trend"
                )
                st.plotly_chart(fig_occupancy, use_container_width=True)
            
            with col2:
                # Entry/exit flow
                fig_flow = go.Figure()
                recent_df = df.tail(10)
                fig_flow.add_trace(go.Bar(name='Entries', x=recent_df['timestamp'], y=recent_df['entrance_count']))
                fig_flow.add_trace(go.Bar(name='Exits', x=recent_df['timestamp'], y=recent_df['exit_count']))
                fig_flow.update_layout(title="Recent Entry/Exit Flow", barmode='group')
                st.plotly_chart(fig_flow, use_container_width=True)
    
    # ENHANCED: Agent information with camera troubleshooting
    with st.expander("ℹ️ About Autonomous Agent & Camera Setup"):
        st.write("""
        **The Autonomous Monitoring Agent provides:**
        
        🔄 **Continuous Operation**: Monitors branch activity 24/7 without human intervention
        
        📹 **Smart Detection**: Uses computer vision to detect and count people entering/exiting
        
        📊 **Real-time Analytics**: Updates occupancy data and generates insights automatically
        
        ⚠️ **Intelligent Alerts**: Sends notifications when capacity thresholds are exceeded
        
        💡 **Smart Recommendations**: Suggests staffing levels and operational improvements
        
        📧 **Notifications**: Sends email alerts to administrators (requires email setup)
        
        **Monitoring Process:**
        1. Capture camera feeds (entrance & exit)
        2. Detect and count people using AI
        3. Update database with new occupancy data
        4. Analyze capacity and generate alerts
        5. Send notifications if thresholds exceeded
        6. Generate staffing recommendations
        7. Repeat cycle every 30 seconds
        
        **Camera Troubleshooting:**
        - **No Cameras**: System automatically falls back to simulation mode
        - **Single Camera**: Set USE_SINGLE_CAMERA=true in .env file
        - **Camera Indices**: 0=built-in webcam, 1=external USB camera
        - **Permission Issues**: Ensure camera permissions are granted
        - **Multiple Apps**: Close other applications using the camera
        - **USB Cameras**: Try different USB ports if camera not detected
        """)
# FIXED: Live Monitor Tab with improved people counting
def live_monitor_tab(config):
    """Live monitoring dashboard with fixed people counting"""
    st.header("Live Monitor")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh (10 seconds)", value=False)
    refresh_button = st.button("Refresh Now")
    
    if refresh_button or auto_refresh:
        # Simulate camera feed processing
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Entrance Camera")
            
            # Camera input or sample frame
            camera_input = st.camera_input("Take entrance photo (for demo)")
            
            if camera_input:
                # Process the image
                image = Image.open(camera_input)
                image_array = np.array(image)
                
                # Convert PIL to OpenCV format
                frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                # Detect faces/people
                face_detector = load_face_detector()
                faces = detect_people_simple(frame, face_detector)
                
                # Classify demographics
                demographics = classify_demographics_mock(faces)
                
                # Draw detection boxes on image for visualization
                detected_image = draw_detections_on_image(image_array, faces)
                
                # Display results with detection boxes
                st.image(detected_image, caption=f"Detected: {len(faces)} people entering")
                st.write(f"Adults: {demographics['adults']}, Kids: {demographics['kids']}")
                st.write(f"Males: {demographics['males']}, Females: {demographics['females']}")
                
                # FIXED: Update database (simulate entrance) - Properly count all detected people
                current_occupancy = get_current_occupancy()
                people_entering = len(faces)  # This now correctly counts all detected faces
                new_occupancy = current_occupancy + people_entering
                
                # Show the calculation for transparency
                st.info(f"Occupancy Update: {current_occupancy} + {people_entering} = {new_occupancy}")
                
                insert_footfall_data(
                    entrance_count=people_entering,  # Use the actual count of detected people
                    exit_count=0,
                    occupancy=new_occupancy,
                    demographics=demographics
                )
                
                # Show success message
                if people_entering > 0:
                    st.success(f"✅ {people_entering} person(s) added to occupancy count")
                else:
                    st.warning("No people detected in the image")
                
            else:
                st.info("Take a photo to simulate entrance detection")
        
        with col2:
            st.subheader("Exit Camera")
            exit_camera = st.camera_input("Take exit photo (for demo)", key="exit")
            
            if exit_camera:
                image = Image.open(exit_camera)
                image_array = np.array(image)
                frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                face_detector = load_face_detector()
                faces = detect_people_simple(frame, face_detector)
                
                # Draw detection boxes on image for visualization
                detected_image = draw_detections_on_image(image_array, faces)
                
                st.image(detected_image, caption=f"Detected: {len(faces)} people exiting")
                
                # FIXED: Update database (simulate exit) - Properly count all detected people
                current_occupancy = get_current_occupancy()
                people_exiting = len(faces)  # This now correctly counts all detected faces
                new_occupancy = max(0, current_occupancy - people_exiting)  # Prevent negative occupancy
                
                # Show the calculation for transparency
                st.info(f"Occupancy Update: {current_occupancy} - {people_exiting} = {new_occupancy}")
                
                insert_footfall_data(
                    entrance_count=0,
                    exit_count=people_exiting,  # Use the actual count of detected people
                    occupancy=new_occupancy,
                    demographics={"adults": 0, "kids": 0, "males": 0, "females": 0}
                )
                
                # Show success message
                if people_exiting > 0:
                    st.success(f"✅ {people_exiting} person(s) removed from occupancy count")
                else:
                    st.warning("No people detected in the image")
                    
            else:
                st.info("Take a photo to simulate exit detection")
    
    # Display current metrics
    st.subheader("Current Status")
    col1, col2, col3, col4 = st.columns(4)
    
    current_occupancy = get_current_occupancy()
    
    with col1:
        st.metric("Current Occupancy", current_occupancy)
    
    with col2:
        capacity_usage = (current_occupancy / config['CAPACITY_MAX']) * 100
        st.metric("Capacity Usage", f"{capacity_usage:.1f}%")
    
    with col3:
        recommended_staff = recommend_staff(current_occupancy, config['STAFF_PER_N'])
        st.metric("Recommended Staff", recommended_staff)
    
    with col4:
        st.metric("Max Capacity", config['CAPACITY_MAX'])
    
    # Capacity progress bar
    progress = min(current_occupancy / config['CAPACITY_MAX'], 1.0)
    st.progress(progress)
    
    # Debug section (optional - can be removed in production)
    with st.expander("Debug Information"):
        st.write("This section shows the detection and counting logic for debugging:")
        st.write("- Each detected face/person is counted individually")
        st.write("- Entrance: Current occupancy + detected people = new occupancy")
        st.write("- Exit: Current occupancy - detected people = new occupancy (minimum 0)")
        st.write("- The face detection algorithm counts each distinct face region in the image")
        st.write("- Detection boxes are drawn around each detected person")
        st.write("- Overlapping detections are filtered to prevent double counting")
    
    if auto_refresh:
        time.sleep(10)
        st.rerun()

# NEW: Queue Management Tab
def queue_management_tab(config):
    """Complete Queue management dashboard with enhanced intelligent agent"""
    st.header("🎯 Queue Management with Enhanced Intelligent Agent")
    
    # Initialize queue agent in session state
    if 'queue_agent' not in st.session_state:
        st.session_state.queue_agent = None
    if 'queue_agent_running' not in st.session_state:
        st.session_state.queue_agent_running = False
    
    # Agent Control Section
    st.subheader("🤖 Enhanced Queue Management Agent")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 Start Queue Agent", type="primary", disabled=st.session_state.queue_agent_running):
            try:
                # Create enhanced queue management agent
                queue_config = config.copy()
                queue_config['APP2_ENDPOINT'] = 'http://localhost:8502'  # Default app2.py endpoint
                queue_config['QUEUE_AGENT_PORT'] = 5001
                
                # Use the enhanced agent with improved A2A communication
                st.session_state.queue_agent = EnhancedQueueManagementAgent(queue_config)
                
                if st.session_state.queue_agent.start_autonomous_monitoring():
                    st.session_state.queue_agent_running = True
                    st.success("🚀 Enhanced Queue Agent started successfully!")
                else:
                    st.error("Failed to start Enhanced Queue Agent")
                    
            except Exception as e:
                st.error(f"Failed to start agent: {str(e)}")
    
    with col2:
        if st.button("⏹️ Stop Queue Agent", disabled=not st.session_state.queue_agent_running):
            try:
                if st.session_state.queue_agent:
                    st.session_state.queue_agent.stop_autonomous_monitoring()
                st.session_state.queue_agent_running = False
                st.success("⏹️ Enhanced Queue Agent stopped!")
                
            except Exception as e:
                st.error(f"Failed to stop agent: {str(e)}")
    
    with col3:
        agent_status = "🟢 Running" if st.session_state.queue_agent_running else "🔴 Stopped"
        st.write(f"**Status:** {agent_status}")
    
    with col4:
        if st.session_state.queue_agent:
            communication_port = st.session_state.queue_agent.a2a_comm.communication_port
            st.write(f"**Port:** {communication_port}")
    
    # Enhanced Communication Status Display
    if st.session_state.queue_agent:
        st.subheader("🔗 A2A Communication Status")
        
        comm_status = st.session_state.queue_agent.get_communication_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = {
                'connected': '🟢',
                'degraded': '🟡', 
                'disconnected': '🔴',
                'stopped': '⚫'
            }
            st.write(f"**Connection:** {status_color.get(comm_status['status'], '❓')} {comm_status['status'].title()}")
        
        with col2:
            app2_status = "🟢 Healthy" if comm_status['app2_healthy'] else "🔴 Unreachable"
            st.write(f"**App2 Health:** {app2_status}")
        
        with col3:
            st.write(f"**Queue Size:** {comm_status['message_queue_size']}")
        
        with col4:
            if comm_status['last_heartbeat']:
                heartbeat_time = datetime.fromisoformat(comm_status['last_heartbeat'])
                st.write(f"**Last Heartbeat:** {heartbeat_time.strftime('%H:%M:%S')}")
            else:
                st.write("**Last Heartbeat:** Never")
        
        # Connection details in expandable section
        with st.expander("📊 Detailed Communication Status"):
            st.json(comm_status)
    
    # Agent Insights Section
    if st.session_state.queue_agent:
        st.subheader("🧠 Agent Insights")
        
        # Get insights (you may need to implement this method in EnhancedQueueManagementAgent)
        try:
            insights = st.session_state.queue_agent.get_queue_insights()
            
            # Current Status Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Queue Length", insights['current_status']['queue_length'])
            
            with col2:
                avg_wait = insights['current_status']['average_wait']
                st.metric("Avg Wait Time", f"{avg_wait:.1f} min" if avg_wait > 0 else "0 min")
            
            with col3:
                served_today = insights['efficiency_metrics']['total_served_today']
                st.metric("Served Today", served_today)
            
            # Alerts
            if insights['alerts']:
                st.subheader("⚠️ Agent Alerts")
                for alert in insights['alerts']:
                    st.warning(alert)
            
            # Recommendations
            if insights['recommendations']:
                st.subheader("💡 Agent Recommendations")
                for recommendation in insights['recommendations']:
                    st.info(recommendation)
        except Exception as e:
            st.warning(f"Could not retrieve agent insights: {str(e)}")
        
        # Enhanced Agent Logs with A2A Communication logs
        st.subheader("📋 Agent Activity Logs")
        
        # Combine agent logs and communication logs
        all_logs = []
        if hasattr(st.session_state.queue_agent, 'logs'):
            all_logs.extend(st.session_state.queue_agent.logs)
        if hasattr(st.session_state.queue_agent.a2a_comm, 'logs'):
            all_logs.extend(st.session_state.queue_agent.a2a_comm.logs)
        
        # Sort by timestamp (assuming logs have timestamps)
        all_logs = sorted(all_logs)[-15:]  # Show last 15 logs
        
        if all_logs:
            log_container = st.container()
            with log_container:
                for log_entry in all_logs:
                    st.text(log_entry)
        else:
            st.info("No agent logs available")
        
        # Test Communication Button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧪 Test A2A Communication"):
                if st.session_state.queue_agent:
                    test_message = {
                        'type': 'test_connection',
                        'data': {
                            'message': 'Enhanced test message from Queue Management Agent',
                            'test_id': str(uuid.uuid4())
                        }
                    }
                    success = st.session_state.queue_agent.send_message_to_app2(test_message)
                    if success:
                        st.success("✅ Test message sent successfully!")
                    else:
                        st.error("❌ Test message failed to send")
                else:
                    st.warning("Agent not initialized")
        
        with col2:
            if st.button("🔄 Refresh Communication Status"):
                st.rerun()
    
    # Divider between agent and regular queue management
    st.divider()
    
    # Get pending queue data
    pending_queue = get_pending_queue()
    
    # Queue statistics
    col1, col2, col3, col4 = st.columns(4)
    
    total_waiting = len(pending_queue)
    
    with col1:
        st.metric("Total in Queue", total_waiting)
    
    with col2:
        avg_wait = pending_queue['estimated_wait'].mean() if not pending_queue.empty else 0
        st.metric("Avg Wait Time", f"{avg_wait:.0f} min")
    
    with col3:
        # Get queue history for today
        queue_history = get_queue_history(1)
        completed_today = queue_history['completed'].sum() if not queue_history.empty else 0
        st.metric("Served Today", completed_today)
    
    with col4:
        if st.button("🔄 Refresh Queue", type="secondary"):
            st.rerun()
    
    if total_waiting == 0:
        st.info("✅ No customers currently in queue")
        # Show queue history even when queue is empty
        st.subheader("📈 Queue History (Last 7 Days)")
        
        history_df = get_queue_history(7)
        if not history_df.empty:
            # Create stacked bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Completed',
                x=history_df['date'],
                y=history_df['completed'],
                marker_color='green'
            ))
            
            fig.add_trace(go.Bar(
                name='Removed/Left',
                x=history_df['date'],
                y=history_df['removed'] + history_df['left_queue'],
                marker_color='red'
            ))
            
            fig.update_layout(
                title="Daily Queue Outcomes",
                xaxis_title="Date",
                yaxis_title="Number of Customers",
                barmode='stack'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No queue history data available")
        
        return
    
    # Current Queue Display
    st.subheader("Current Queue")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Serve Next Customer", type="primary"):
            if not pending_queue.empty:
                first_customer = pending_queue.iloc[0]
                if mark_queue_served(first_customer['queue_id']):
                    st.success(f"✅ {first_customer['name']} marked as served!")
                    
                    # Enhanced agent notification about service completion
                    if st.session_state.queue_agent and st.session_state.queue_agent_running:
                        service_data = {
                            'customer_id': first_customer['user_id'],
                            'service_time': 5.0,  # Default service time
                            'satisfaction': None
                        }
                        # Send notification to app2.py via A2A communication
                        message = {
                            'type': 'service_completed',
                            'data': service_data
                        }
                        st.session_state.queue_agent.send_message_to_app2(message)
                    
                    # Send notification
                    if config['ENABLE_NOTIFICATIONS'] and first_customer['email']:
                        success, msg = send_queue_notification(
                            first_customer['email'], 
                            first_customer['name'], 
                            'completed', 
                            config
                        )
                        if success:
                            st.info("📧 Completion notification sent")
                    
                    time.sleep(1)
                    st.rerun()
    
    with col2:
        # Bulk actions
        if st.button("⚡ Update All Positions"):
            if update_queue_positions():
                st.success("📊 Queue positions updated!")
                
                # Notify agent about queue position updates
                if st.session_state.queue_agent and st.session_state.queue_agent_running:
                    message = {
                        'type': 'queue_positions_updated',
                        'data': {
                            'total_queue_length': len(get_pending_queue()),
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                    st.session_state.queue_agent.send_message_to_app2(message)
                
                time.sleep(1)
                st.rerun()
    
    with col3:
        # Emergency clear (with confirmation)
        if st.button("🚨 Emergency Clear Queue", type="secondary"):
            st.session_state.confirm_clear = True
    
    # Emergency clear confirmation
    if st.session_state.get('confirm_clear', False):
        st.error("⚠️ This will remove ALL customers from the queue!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm Clear All"):
                try:
                    conn = sqlite3.connect('data/database.db')
                    cursor = conn.cursor()
                    cursor.execute("UPDATE queue_entries SET status = 'removed' WHERE status = 'waiting'")
                    conn.commit()
                    conn.close()
                    
                    # Enhanced agent notification about queue clear
                    if st.session_state.queue_agent and st.session_state.queue_agent_running:
                        st.session_state.queue_agent.add_log("Emergency queue clear executed")
                        # Send notification to app2.py
                        message = {
                            'type': 'emergency_queue_clear',
                            'data': {
                                'cleared_customers': total_waiting,
                                'timestamp': datetime.now().isoformat(),
                                'reason': 'emergency_clear'
                            }
                        }
                        st.session_state.queue_agent.send_message_to_app2(message)
                    
                    st.success("🗑️ All customers removed from queue")
                    st.session_state.confirm_clear = False
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing queue: {e}")
        
        with col2:
            if st.button("❌ Cancel"):
                st.session_state.confirm_clear = False
                st.rerun()
    
    # Queue table with actions
    st.write("---")
    
    for index, customer in pending_queue.iterrows():
        col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 2, 2, 1, 2])
        
        with col1:
            st.write(f"**#{customer['position']}**")
        
        with col2:
            st.write(f"**{customer['name']}**")
            if customer['email']:
                st.caption(f"📧 {customer['email']}")
        
        with col3:
            st.write(f"🕒 {customer['estimated_wait']} min wait")
            join_time = datetime.fromisoformat(customer['join_time'])
            st.caption(f"Joined: {join_time.strftime('%H:%M')}")
        
        with col4:
            if customer['phone']:
                st.write(f"📱 {customer['phone']}")
            st.caption(f"Age: {customer['age']}, {customer['gender'].title()}")
        
        with col5:
            # Serve this customer
            if st.button("✅ Serve", key=f"serve_{customer['queue_id']}"):
                if mark_queue_served(customer['queue_id']):
                    st.success(f"✅ {customer['name']} served!")
                    
                    # Enhanced agent notification about service completion
                    if st.session_state.queue_agent and st.session_state.queue_agent_running:
                        service_data = {
                            'customer_id': customer['user_id'],
                            'customer_name': customer['name'],
                            'service_time': 5.0,
                            'satisfaction': None,
                            'queue_position': customer['position']
                        }
                        message = {
                            'type': 'individual_service_completed',
                            'data': service_data
                        }
                        st.session_state.queue_agent.send_message_to_app2(message)
                    
                    # Send notification
                    if config['ENABLE_NOTIFICATIONS'] and customer['email']:
                        send_queue_notification(
                            customer['email'], 
                            customer['name'], 
                            'completed', 
                            config
                        )
                    
                    time.sleep(1)
                    st.rerun()
        
        with col6:
            # Remove from queue
            if st.button("❌ Remove", key=f"remove_{customer['queue_id']}"):
                if remove_from_queue(customer['queue_id'], "removed"):
                    st.success(f"🗑️ {customer['name']} removed from queue")
                    
                    # Enhanced agent notification about customer removal
                    if st.session_state.queue_agent and st.session_state.queue_agent_running:
                        removal_data = {
                            'customer_id': customer['user_id'],
                            'customer_name': customer['name'],
                            'reason': 'manual_removal',
                            'queue_position': customer['position']
                        }
                        message = {
                            'type': 'customer_removed',
                            'data': removal_data
                        }
                        st.session_state.queue_agent.send_message_to_app2(message)
                    
                    # Send notification
                    if config['ENABLE_NOTIFICATIONS'] and customer['email']:
                        send_queue_notification(
                            customer['email'], 
                            customer['name'], 
                            'removed', 
                            config
                        )
                    
                    time.sleep(1)
                    st.rerun()
        
        st.divider()
    
    # Queue History Section
    st.subheader("📈 Queue History (Last 7 Days)")
    
    history_df = get_queue_history(7)
    if not history_df.empty:
        # Create stacked bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Completed',
            x=history_df['date'],
            y=history_df['completed'],
            marker_color='green'
        ))
        
        fig.add_trace(go.Bar(
            name='Removed/Left',
            x=history_df['date'],
            y=history_df['removed'] + history_df['left_queue'],
            marker_color='red'
        ))
        
        fig.update_layout(
            title="Daily Queue Outcomes",
            xaxis_title="Date",
            yaxis_title="Number of Customers",
            barmode='stack'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_served = history_df['completed'].sum()
            st.metric("Total Served (7d)", total_served)
        
        with col2:
            total_entries = history_df['total_entries'].sum()
            completion_rate = (total_served / total_entries * 100) if total_entries > 0 else 0
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        
        with col3:
            avg_actual_wait = history_df['avg_actual_wait'].mean()
            st.metric("Avg Actual Wait", f"{avg_actual_wait:.0f} min")
        
        with col4:
            avg_estimated_wait = history_df['avg_estimated_wait'].mean()
            st.metric("Avg Estimated Wait", f"{avg_estimated_wait:.0f} min")
        
        # Detailed history table
        with st.expander("📋 Detailed History Data"):
            st.dataframe(
                history_df,
                column_config={
                    'date': 'Date',
                    'total_entries': 'Total Entries',
                    'completed': 'Completed',
                    'removed': 'Removed',
                    'left_queue': 'Left Queue',
                    'avg_actual_wait': st.column_config.NumberColumn('Avg Actual Wait (min)', format="%.1f"),
                    'avg_estimated_wait': st.column_config.NumberColumn('Avg Estimated Wait (min)', format="%.1f")
                },
                use_container_width=True
            )
    else:
        st.info("No queue history data available")
    
    # Enhanced Agent Communication Status (if agent is running)
    if st.session_state.queue_agent_running:
        with st.expander("📡 Advanced A2A Communication Settings"):
            st.write("**Enhanced A2A Communication Configuration:**")
            st.write(f"- App2 Endpoint: {config.get('APP2_ENDPOINT', 'http://localhost:8502')}")
            st.write(f"- Communication Port: {config.get('QUEUE_AGENT_PORT', 5001)}")
            st.write(f"- Agent ID: {st.session_state.queue_agent.agent_id if st.session_state.queue_agent else 'N/A'}")
            st.write(f"- Retry Count: {st.session_state.queue_agent.a2a_comm.retry_count}")
            st.write(f"- Max Retries: {st.session_state.queue_agent.a2a_comm.max_retries}")
            
            # Communication health check
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🏥 Check App2 Health"):
                    if st.session_state.queue_agent:
                        is_healthy = st.session_state.queue_agent.a2a_comm.check_app2_health()
                        if is_healthy:
                            st.success("✅ App2 is healthy and responding")
                        else:
                            st.error("❌ App2 is not responding or unhealthy")
            
            with col2:
                if st.button("🤝 Perform Handshake"):
                    if st.session_state.queue_agent:
                        handshake_success = st.session_state.queue_agent.a2a_comm.perform_handshake()
                        if handshake_success:
                            st.success("✅ Handshake successful")
                        else:
                            st.error("❌ Handshake failed")
            
            # Show recent communication logs
            st.write("**Recent A2A Communication Activity:**")
            if hasattr(st.session_state.queue_agent.a2a_comm, 'logs') and st.session_state.queue_agent.a2a_comm.logs:
                recent_comm_logs = st.session_state.queue_agent.a2a_comm.logs[-5:]
                for log in recent_comm_logs:
                    st.text(log)
            else:
                st.info("No recent communication activity")
    
    # Auto-refresh for active monitoring (reduced frequency to prevent excessive communication)
    if st.session_state.queue_agent_running:
        time.sleep(45)  # Refresh every 45 seconds when agent is active
        st.rerun()
def customer_management_tab(config):
    """Customer management dashboard"""
    st.header("Customer Management")
    
    # Get customer data
    customers_df = get_all_customers()
    current_queue = get_current_queue()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(customers_df)
        st.metric("Total Registered", total_customers)
    
    with col2:
        current_queue_size = len(current_queue)
        st.metric("Current Queue", current_queue_size)
    
    with col3:
        if not customers_df.empty:
            avg_age = customers_df['age'].mean() if 'age' in customers_df.columns else 0
            st.metric("Avg Customer Age", f"{avg_age:.1f}")
        else:
            st.metric("Avg Customer Age", "N/A")
    
    with col4:
        today_registrations = 0
        if not customers_df.empty and 'created_at' in customers_df.columns:
            today = datetime.now().date()
            customers_df['created_at'] = pd.to_datetime(customers_df['created_at'])
            today_registrations = len(customers_df[customers_df['created_at'].dt.date == today])
        st.metric("Today's Registrations", today_registrations)
    
    # Current Queue Status
    st.subheader("Current Queue Status")
    if not current_queue.empty:
        st.dataframe(
            current_queue[['position', 'name', 'email', 'phone', 'estimated_wait', 'join_time']],
            column_config={
                'position': 'Position',
                'name': 'Name',
                'email': 'Email',
                'phone': 'Phone',
                'estimated_wait': 'Wait Time (min)',
                'join_time': 'Join Time'
            },
            use_container_width=True
        )
        
        # Export current queue
        if st.button("Export Current Queue CSV"):
            try:
                export_dir = Path("data/exports")
                export_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"current_queue_{timestamp}.csv"
                filepath = export_dir / filename
                
                current_queue.to_csv(filepath, index=False)
                
                with open(filepath, 'rb') as f:
                    st.download_button(
                        label="Download Queue CSV",
                        data=f.read(),
                        file_name=filename,
                        mime="text/csv"
                    )
                st.success(f"Queue data exported: {filename}")
            except Exception as e:
                st.error(f"Export failed: {e}")
    else:
        st.info("No customers currently in queue")
    
    # All Registered Customers
    st.subheader("All Registered Customers")
    if not customers_df.empty:
        # Search/filter functionality
        col1, col2 = st.columns(2)
        with col1:
            search_term = st.text_input("Search by name or email")
        with col2:
            age_filter = st.selectbox("Filter by age group", ["All", "18-25", "26-35", "36-50", "50+"])
        
        # Apply filters
        filtered_df = customers_df.copy()
        
        if search_term:
            filtered_df = filtered_df[
                filtered_df['name'].str.contains(search_term, case=False, na=False) |
                filtered_df['email'].str.contains(search_term, case=False, na=False)
            ]
        
        if age_filter != "All":
            if age_filter == "18-25":
                filtered_df = filtered_df[(filtered_df['age'] >= 18) & (filtered_df['age'] <= 25)]
            elif age_filter == "26-35":
                filtered_df = filtered_df[(filtered_df['age'] >= 26) & (filtered_df['age'] <= 35)]
            elif age_filter == "36-50":
                filtered_df = filtered_df[(filtered_df['age'] >= 36) & (filtered_df['age'] <= 50)]
            elif age_filter == "50+":
                filtered_df = filtered_df[filtered_df['age'] > 50]
        
        # Display filtered results
        st.write(f"Showing {len(filtered_df)} of {len(customers_df)} customers")
        
        if not filtered_df.empty:
            # Format the dataframe for display
            display_df = filtered_df.copy()
            if 'created_at' in display_df.columns:
                display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            if 'last_visit' in display_df.columns:
                display_df['last_visit'] = pd.to_datetime(display_df['last_visit'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                display_df[['name', 'email', 'phone', 'age', 'gender', 'created_at', 'queue_visits', 'last_visit', 'avg_rating']],
                column_config={
                    'name': 'Name',
                    'email': 'Email',
                    'phone': 'Phone',
                    'age': 'Age',
                    'gender': 'Gender',
                    'created_at': 'Registered',
                    'queue_visits': 'Total Visits',
                    'last_visit': 'Last Visit',
                    'avg_rating': st.column_config.NumberColumn('Avg Rating', format="%.1f")
                },
                use_container_width=True
            )
            
            # Export filtered customers
            if st.button("Export Filtered Customer Data"):
                try:
                    export_dir = Path("data/exports")
                    export_dir.mkdir(exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"customers_filtered_{timestamp}.csv"
                    filepath = export_dir / filename
                    
                    filtered_df.to_csv(filepath, index=False)
                    
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            label="Download Customer CSV",
                            data=f.read(),
                            file_name=filename,
                            mime="text/csv"
                        )
                    st.success(f"Customer data exported: {filename}")
                except Exception as e:
                    st.error(f"Export failed: {e}")
        else:
            st.write("No customers match the current filters")
    else:
        st.info("No registered customers found")
    
    # Customer Demographics
    if not customers_df.empty:
        st.subheader("Customer Demographics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            if 'age' in customers_df.columns:
                age_bins = [0, 18, 25, 35, 50, 100]
                age_labels = ['<18', '18-25', '26-35', '36-50', '50+']
                customers_df['age_group'] = pd.cut(customers_df['age'], bins=age_bins, labels=age_labels, right=False)
                age_counts = customers_df['age_group'].value_counts()
                
                fig_age = px.pie(
                    values=age_counts.values,
                    names=age_counts.index,
                    title="Age Distribution"
                )
                st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Gender distribution
            if 'gender' in customers_df.columns:
                gender_counts = customers_df['gender'].value_counts()
                
                fig_gender = px.pie(
                    values=gender_counts.values,
                    names=gender_counts.index,
                    title="Gender Distribution"
                )
                st.plotly_chart(fig_gender, use_container_width=True)

def queue_analytics_tab():
    """Queue analytics dashboard"""
    st.header("Queue Analytics")
    
    # Get queue analytics data
    queue_df = get_queue_analytics()
    
    if not queue_df.empty:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_entries = queue_df['total_entries'].sum()
            st.metric("Total Queue Entries (30 days)", total_entries)
        
        with col2:
            avg_wait = queue_df['avg_wait_time'].mean()
            st.metric("Avg Wait Time", f"{avg_wait:.1f} min")
        
        with col3:
            total_completed = queue_df['completed_services'].sum()
            st.metric("Services Completed", total_completed)
        
        with col4:
            completion_rate = (total_completed / total_entries * 100) if total_entries > 0 else 0
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        
        # Daily queue entries chart
        st.subheader("Daily Queue Activity")
        fig_daily = px.bar(
            queue_df,
            x='date',
            y='total_entries',
            title="Daily Queue Entries"
        )
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Wait time trends
        st.subheader("Average Wait Time Trends")
        fig_wait = px.line(
            queue_df,
            x='date',
            y='avg_wait_time',
            title="Average Wait Time by Date"
        )
        st.plotly_chart(fig_wait, use_container_width=True)
        
        # Data table
        st.subheader("Detailed Queue Data")
        st.dataframe(
            queue_df,
            column_config={
                'date': 'Date',
                'total_entries': 'Queue Entries',
                'avg_wait_time': st.column_config.NumberColumn('Avg Wait Time (min)', format="%.1f"),
                'completed_services': 'Completed Services'
            },
            use_container_width=True
        )
    else:
        st.info("No queue analytics data available")

def analytics_tab():
    """Analytics dashboard"""
    st.header("Analytics Dashboard")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.selectbox("Time Range", [1, 7, 30], index=1, 
                                format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}")
    
    with col2:
        if st.button("Refresh Analytics"):
            st.rerun()
    
    # Get data
    df = get_analytics_data(days_back)
    
    if not df.empty:
        # Occupancy trend
        st.subheader("Occupancy Trend")
        fig_line = px.line(df, x='timestamp', y='occupancy', 
                          title="Occupancy Over Time")
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Entry/Exit flow
        st.subheader("Entry/Exit Flow")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='Entries', x=df['timestamp'], y=df['entrance_count']))
        fig_bar.add_trace(go.Bar(name='Exits', x=df['timestamp'], y=df['exit_count']))
        fig_bar.update_layout(title="Entry/Exit Counts", barmode='group')
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Demographics
        if 'adults' in df.columns:
            st.subheader("Demographics Breakdown")
            col1, col2 = st.columns(2)
            
            with col1:
                # Age demographics
                age_data = {
                    'Adults': df['adults'].sum(),
                    'Kids': df['kids'].sum()
                }
                if sum(age_data.values()) > 0:
                    fig_pie = px.pie(values=list(age_data.values()),
                                   names=list(age_data.keys()),
                                   title="Age Distribution")
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Gender demographics
                if 'males' in df.columns:
                    gender_data = {
                        'Males': df['males'].sum(),
                        'Females': df['females'].sum()
                    }
                    if sum(gender_data.values()) > 0:
                        fig_pie2 = px.pie(values=list(gender_data.values()), 
                                        names=list(gender_data.keys()),
                                        title="Gender Distribution")
                        st.plotly_chart(fig_pie2, use_container_width=True)
        
        # Peak hours analysis
        st.subheader("Peak Hours Analysis")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format='mixed')
        df['hour'] = df['timestamp'].dt.hour
        hourly_avg = df.groupby('hour')['occupancy'].mean()
        
        fig_peak = go.Figure()
        fig_peak.add_trace(go.Bar(
            x=hourly_avg.index,
            y=hourly_avg.values,
            name="Average Occupancy"
        ))
        fig_peak.update_layout(
            title="Average Occupancy by Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Average Occupancy"
        )
        st.plotly_chart(fig_peak, use_container_width=True)
        
        # Show peak hours
        top_3_hours = hourly_avg.nlargest(3)
        st.write(f"**Peak Hours:** {', '.join([f'{h}:00' for h in top_3_hours.index])}")
    
    else:
        st.warning("No analytics data available. Use the Live Monitor to collect data first.")

def staffing_tab(config):
    """Staffing recommendations dashboard"""
    st.header("Staffing Recommendations")
    
    current_occupancy = get_current_occupancy()
    current_staff = recommend_staff(current_occupancy, config['STAFF_PER_N'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Occupancy", current_occupancy)
    
    with col2:
        st.metric("Recommended Staff", current_staff)
    
    with col3:
        st.metric("Staff Ratio", f"1:{config['STAFF_PER_N']}")
    
    # Weekend projection
    st.subheader("Weekend vs Weekday Analysis")
    df = get_analytics_data(30)  # Last 30 days
    weekend_data = get_weekend_uplift(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Weekday Avg", f"{weekend_data['weekday_avg']:.0f}")
    
    with col2:
        st.metric("Weekend Avg", f"{weekend_data['weekend_avg']:.0f}")
    
    with col3:
        st.metric("Weekend Uplift", f"{weekend_data['uplift']:.1f}%")
    
    # Staffing guidelines
    st.subheader("Staffing Guidelines")
    st.write(f"- **Base Ratio:** 1 staff member per {config['STAFF_PER_N']} customers")
    st.write(f"- **Minimum Staff:** 1 (even when occupancy is 0)")
    st.write(f"- **Peak Capacity:** {config['CAPACITY_MAX']} customers")
    st.write(f"- **Maximum Staff Needed:** {recommend_staff(config['CAPACITY_MAX'], config['STAFF_PER_N'])}")
    
    # Capacity gauge
    utilization = (current_occupancy / config['CAPACITY_MAX']) * 100
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = utilization,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Capacity Utilization (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    st.plotly_chart(fig_gauge, use_container_width=True)

def export_tab():
    """Data export dashboard"""
    st.header("Export Analytics Data")
    
    # Multiple export options
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Footfall Analytics**")
        export_range = st.selectbox("Export Range", [7, 30, 90], 
                                   format_func=lambda x: f"Last {x} days")
        
        if st.button("Generate Footfall Export"):
            df = get_analytics_data(export_range)
            
            if not df.empty:
                # Create export directory
                export_dir = Path("data/exports")
                export_dir.mkdir(exist_ok=True)
                
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"analytics_{export_range}days_{timestamp}.csv"
                filepath = export_dir / filename
                
                # Save CSV
                df.to_csv(filepath, index=False)
                
                st.success(f"Export saved: {filename}")
                
                # Display preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Download button
                with open(filepath, 'rb') as f:
                    st.download_button(
                        label="Download CSV",
                        data=f.read(),
                        file_name=filename,
                        mime="text/csv"
                    )
            else:
                st.warning("No data available to export")
    
    with col2:
        st.write("**Customer Data**")
        customer_export_type = st.selectbox("Export Type", 
                                           ["All Customers", "Current Queue", "Recent Registrations"])
        
        if st.button("Generate Customer Export"):
            if customer_export_type == "All Customers":
                customers_df = get_all_customers()
                export_data = customers_df
                filename_prefix = "all_customers"
            elif customer_export_type == "Current Queue":
                export_data = get_current_queue()
                filename_prefix = "current_queue"
            else:  # Recent Registrations
                customers_df = get_all_customers()
                if not customers_df.empty:
                    # Filter for last 7 days
                    customers_df['created_at'] = pd.to_datetime(customers_df['created_at'])
                    week_ago = datetime.now() - timedelta(days=7)
                    export_data = customers_df[customers_df['created_at'] >= week_ago]
                else:
                    export_data = customers_df
                filename_prefix = "recent_customers"
            
            if not export_data.empty:
                # Create export directory
                export_dir = Path("data/exports")
                export_dir.mkdir(exist_ok=True)
                
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{filename_prefix}_{timestamp}.csv"
                filepath = export_dir / filename
                
                # Save CSV
                export_data.to_csv(filepath, index=False)
                
                st.success(f"Export saved: {filename}")
                
                # Display preview
                st.subheader("Data Preview")
                st.dataframe(export_data.head())
                
                # Download button
                with open(filepath, 'rb') as f:
                    st.download_button(
                        label="Download Customer CSV",
                        data=f.read(),
                        file_name=filename,
                        mime="text/csv"
                    )
            else:
                st.warning("No customer data available to export")
    
    # Show recent exports
    st.subheader("Recent Exports")
    export_dir = Path("data/exports")
    if export_dir.exists():
        exports = list(export_dir.glob("*.csv"))
        if exports:
            for export_file in sorted(exports, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
                mod_time = datetime.fromtimestamp(export_file.stat().st_mtime)
                st.write(f"📄 {export_file.name} ({mod_time.strftime('%Y-%m-%d %H:%M')})")
        else:
            st.write("No recent exports")

def email_test_tab(config):
    """Email testing functionality"""
    st.header("📧 Email Testing & Reports")
    
    # Email configuration status
    st.subheader("Email Configuration Status")
    
    if config['SMTP_USER'] and config['SMTP_PASS']:
        st.success("✅ Email configuration found")
        st.write(f"SMTP Host: {config['SMTP_HOST']}")
        st.write(f"SMTP Port: {config['SMTP_PORT']}")
        st.write(f"From Email: {config['FROM_EMAIL']}")
    else:
        st.error("❌ Email configuration incomplete")
        st.write("Please check your .env file settings")
    
    # Test email section
    st.subheader("Send Test Email")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_email = st.text_input("Recipient Email", placeholder="test@example.com")
        
    with col2:
        email_subject = st.text_input("Subject", value="Smart Branch Test Email")
    
    email_body = st.text_area("Message", value="This is a test email from Smart Branch system.", height=100)
    
    # Generate analytics report for email
    if st.button("Send Analytics Report Email"):
        if not test_email:
            st.error("Please enter recipient email")
        elif not config['SMTP_USER']:
            st.error("Email not configured")
        else:
            # Generate report
            df = get_analytics_data(7)  # Last 7 days
            customers_df = get_all_customers()
            
            if not df.empty or not customers_df.empty:
                # Create CSV reports
                analytics_path = "data/analytics_report.csv"
                customers_path = "data/customers_report.csv"
                
                if not df.empty:
                    df.to_csv(analytics_path, index=False)
                if not customers_df.empty:
                    customers_df.to_csv(customers_path, index=False)
                
                # Email body with summary
                current_occupancy = get_current_occupancy()
                total_entries = df['entrance_count'].sum() if not df.empty else 0
                avg_occupancy = df['occupancy'].mean() if not df.empty else 0
                total_customers = len(customers_df)
                
                report_body = f"""
Smart Branch Analytics Report

Report Period: Last 7 days
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary:
- Current Occupancy: {current_occupancy}
- Total Entries (7 days): {total_entries}
- Average Occupancy: {avg_occupancy:.1f}
- Total Registered Customers: {total_customers}

Please find the detailed analytics and customer data attached as CSV files.

Best regards,
Smart Branch System
"""
                
                # Send with both attachments if available
                attachment_path = analytics_path if not df.empty else customers_path
                
                success, message = send_email(
                    to_email=test_email,
                    subject="Smart Branch Analytics Report",
                    body=report_body,
                    attachment_path=attachment_path,
                    config=config
                )
                
                if success:
                    st.success("Analytics report email sent successfully!")
                else:
                    st.error(f"Failed to send email: {message}")
            else:
                st.warning("No analytics or customer data available for report")
    
    # Send customer data report
    if st.button("Send Customer Report Email"):
        if not test_email:
            st.error("Please enter recipient email")
        elif not config['SMTP_USER']:
            st.error("Email not configured")
        else:
            customers_df = get_all_customers()
            current_queue = get_current_queue()
            
            if not customers_df.empty:
                # Create customer report
                customers_path = "data/customers_detailed_report.csv"
                customers_df.to_csv(customers_path, index=False)
                
                # Email body
                total_customers = len(customers_df)
                current_queue_size = len(current_queue)
                today_registrations = 0
                
                if 'created_at' in customers_df.columns:
                    today = datetime.now().date()
                    customers_df['created_at'] = pd.to_datetime(customers_df['created_at'])
                    today_registrations = len(customers_df[customers_df['created_at'].dt.date == today])
                
                report_body = f"""
Smart Branch Customer Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Customer Summary:
- Total Registered Customers: {total_customers}
- Current Queue Size: {current_queue_size}
- Today's New Registrations: {today_registrations}

Please find the detailed customer database attached as CSV.

Best regards,
Smart Branch System
"""
                
                success, message = send_email(
                    to_email=test_email,
                    subject="Smart Branch Customer Report",
                    body=report_body,
                    attachment_path=customers_path,
                    config=config
                )
                
                if success:
                    st.success("Customer report email sent successfully!")
                else:
                    st.error(f"Failed to send email: {message}")
            else:
                st.warning("No customer data available for report")
    
    # Simple test email
    if st.button("Send Test Email"):
        if not test_email:
            st.error("Please enter recipient email")
        elif not config['SMTP_USER']:
            st.error("Email not configured")
        else:
            success, message = send_email(
                to_email=test_email,
                subject=email_subject,
                body=email_body,
                config=config
            )
            
            if success:
                st.success("Test email sent successfully!")
            else:
                st.error(f"Failed to send email: {message}")
    
    # Email logs (mock)
    st.subheader("Recent Email Activity")
    st.write("📧 Customer report sent - 1 hour ago")
    st.write("📊 Weekly analytics report - 1 day ago")
    st.write("⚠️ Capacity alert - 2 days ago")

if __name__ == "__main__":
    main()