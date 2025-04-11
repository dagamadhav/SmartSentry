import json
import os
import time
from datetime import datetime
from collections import defaultdict, Counter
import threading
import pandas as pd
from ..config.settings import ALERT_LOG_FILE, TIME_WINDOWS

class AlertManager:
    def __init__(self):
        self.alerts = []
        self.last_backup = time.time()
        self.last_summary = time.time()
        self.alert_lock = threading.Lock()
        self.object_last_alert = defaultdict(float)
        self.load_alerts()
        
        # Start background threads
        self.start_background_tasks()

    def load_alerts(self):
        """Load existing alerts from file"""
        try:
            os.makedirs(os.path.dirname(ALERT_LOG_FILE), exist_ok=True)
            if os.path.exists(ALERT_LOG_FILE):
                with open(ALERT_LOG_FILE, 'r') as f:
                    self.alerts = json.load(f)
            else:
                self.alerts = []
                self.save_alerts()
        except Exception as e:
            print(f"Error loading alerts: {str(e)}")
            self.alerts = []
            self.save_alerts()

    def save_alerts(self):
        """Save alerts to file"""
        with self.alert_lock:
            try:
                with open(ALERT_LOG_FILE, 'w') as f:
                    json.dump(self.alerts, f, indent=4)
            except Exception as e:
                print(f"Error saving alerts: {str(e)}")

    def add_alert(self, object_name, confidence, category):
        """Add new alert if cooldown period has passed"""
        try:
            current_time = time.time()
            
            # Check cooldown period based on category
            cooldown = {
                "HUMAN": 5.0,
                "ANIMAL": 10.0,
                "VEHICLE": 10.0,
                "HIGH_PRIORITY": 2.0,
                "LOW_PRIORITY": 15.0,
                "UNKNOWN": 5.0
            }.get(category, 5.0)
            
            if current_time - self.object_last_alert[object_name] < cooldown:
                return False
            
            with self.alert_lock:
                alert = {
                    "object": object_name,
                    "confidence": float(confidence),
                    "category": category,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "unix_timestamp": current_time,
                    "status": "new"
                }
                self.alerts.append(alert)
                self.object_last_alert[object_name] = current_time
                
                # Save immediately for important alerts
                if category in ["HUMAN", "HIGH_PRIORITY"]:
                    self.save_alerts()
                
                return True
        except Exception as e:
            print(f"Error adding alert: {str(e)}")
            return False

    def get_alert_summary(self, time_window="medium"):
        """Get summary of alerts within specified time window"""
        try:
            current_time = time.time()
            window_seconds = TIME_WINDOWS.get(time_window, TIME_WINDOWS["medium"])
            
            with self.alert_lock:
                recent_alerts = [
                    alert for alert in self.alerts
                    if current_time - alert["unix_timestamp"] <= window_seconds
                ]
            
            if not recent_alerts:
                return {
                    "total_alerts": 0,
                    "window_duration": f"{window_seconds/60:.1f} minutes",
                    "by_category": {},
                    "top_objects": {},
                    "average_confidence": 0
                }
            
            # Count by category and object
            category_counts = Counter(alert["category"] for alert in recent_alerts)
            object_counts = Counter(alert["object"] for alert in recent_alerts)
            
            # Calculate average confidence
            avg_confidence = sum(alert["confidence"] for alert in recent_alerts) / len(recent_alerts)
            
            # Get most frequent objects
            top_objects = dict(object_counts.most_common(5))
            
            return {
                "total_alerts": len(recent_alerts),
                "window_duration": f"{window_seconds/60:.1f} minutes",
                "by_category": dict(category_counts),
                "top_objects": top_objects,
                "average_confidence": avg_confidence
            }
        except Exception as e:
            print(f"Error getting alert summary: {str(e)}")
            return {
                "total_alerts": 0,
                "window_duration": "0 minutes",
                "by_category": {},
                "top_objects": {},
                "average_confidence": 0
            }

    def export_alerts(self, filename="alerts_export.csv"):
        """Export alerts to CSV file"""
        try:
            with self.alert_lock:
                if self.alerts:
                    df = pd.DataFrame(self.alerts)
                    df.to_csv(filename, index=False)
                    return True
            return False
        except Exception as e:
            print(f"Error exporting alerts: {str(e)}")
            return False

    def start_background_tasks(self):
        """Start background threads for periodic tasks"""
        def backup_task():
            while True:
                time.sleep(60)
                current_time = time.time()
                if current_time - self.last_backup >= 60:
                    self.save_alerts()
                    self.last_backup = current_time
        
        backup_thread = threading.Thread(target=backup_task, daemon=True)
        backup_thread.start() 