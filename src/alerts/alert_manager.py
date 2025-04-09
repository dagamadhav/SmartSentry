import json
import os
import time
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import threading
import pandas as pd
from ..config.settings import ALERT_LOG_FILE, TIME_WINDOWS, OBJECT_CATEGORIES

class AlertManager:
    def __init__(self):
        self.alerts = []
        self.last_backup = time.time()
        self.last_summary = time.time()
        self.alert_lock = threading.Lock()
        self.object_last_alert = defaultdict(float)  # Track last alert time for each object
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
                self.save_alerts()  # Create empty file
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error reading {ALERT_LOG_FILE}: {str(e)}, starting with empty alerts")
            self.alerts = []
            self.save_alerts()  # Create empty file

    def save_alerts(self):
        """Save alerts to file"""
        with self.alert_lock:
            with open(ALERT_LOG_FILE, 'w') as f:
                json.dump(self.alerts, f, indent=4)

    def add_alert(self, object_name, confidence, category):
        """Add new alert if cooldown period has passed"""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.object_last_alert[object_name] < OBJECT_CATEGORIES[category]["alert_cooldown"]:
            return False
        
        with self.alert_lock:
            try:
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
                
                # Save immediately for real-time updates
                self.save_alerts()
                return True
            except Exception as e:
                print(f"Error adding alert: {str(e)}")
                return False

    def get_alert_summary(self, time_window="medium"):
        """Get summary of alerts within specified time window"""
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
                "by_object": {},
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

    def export_alerts(self, filename="alerts_export.csv"):
        """Export alerts to CSV file"""
        with self.alert_lock:
            if self.alerts:
                df = pd.DataFrame(self.alerts)
                df.to_csv(filename, index=False)
                return True
        return False

    def cleanup_old_alerts(self, max_age_days=7):
        """Remove alerts older than specified days"""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        with self.alert_lock:
            self.alerts = [
                alert for alert in self.alerts
                if current_time - alert["unix_timestamp"] <= max_age_seconds
            ]
            self.save_alerts()

    def start_background_tasks(self):
        """Start background threads for periodic tasks"""
        def backup_task():
            while True:
                time.sleep(60)  # Check every minute
                current_time = time.time()
                if current_time - self.last_backup >= 60:
                    self.save_alerts()
                    self.last_backup = current_time
        
        backup_thread = threading.Thread(target=backup_task, daemon=True)
        backup_thread.start()

    def get_object_duration(self, object_name, time_window="medium"):
        """Calculate how long an object has been detected in the given time window"""
        window_seconds = TIME_WINDOWS.get(time_window, TIME_WINDOWS["medium"])
        current_time = time.time()
        
        with self.alert_lock:
            object_alerts = [
                alert for alert in self.alerts
                if alert["object"] == object_name and
                current_time - alert["unix_timestamp"] <= window_seconds
            ]
        
        if not object_alerts:
            return 0
        
        # Calculate approximate duration based on alert frequency and cooldown
        category = next(cat for cat, info in OBJECT_CATEGORIES.items() 
                       if object_name in info["objects"])
        cooldown = OBJECT_CATEGORIES[category]["alert_cooldown"]
        
        # Estimate duration as number of alerts * cooldown period
        estimated_duration = len(object_alerts) * cooldown
        return min(estimated_duration, window_seconds)  # Cap at window size 