import json
import os
from datetime import datetime
import threading
from queue import Queue
import time
import glob
import shutil

class AlertManager:
    def __init__(self, base_dir='data/alerts'):
        self.base_dir = base_dir
        self.alert_queue = Queue()
        self.running = False
        self.lock = threading.Lock()
        
        # Create alerts directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Clean up old files (optional, uncomment if needed)
        # self._cleanup_old_files()
        
        # Create new alert file with timestamp
        self.alert_file = self._create_new_alert_file()
        print(f"Created new alert file: {self.alert_file}")
    
    def _cleanup_old_files(self, max_age_days=7):
        """Clean up alert files older than max_age_days"""
        try:
            current_time = datetime.now()
            pattern = os.path.join(self.base_dir, "alerts_*.json")
            for file in glob.glob(pattern):
                file_time = datetime.fromtimestamp(os.path.getctime(file))
                if (current_time - file_time).days > max_age_days:
                    os.remove(file)
        except Exception as e:
            print(f"Error cleaning up old files: {str(e)}")
    
    def _create_new_alert_file(self):
        """Create a new alert file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alerts_{timestamp}.json"
        filepath = os.path.join(self.base_dir, filename)
        
        # Remove any existing alerts1.json if it exists
        old_file = os.path.join(self.base_dir, "alerts1.json")
        if os.path.exists(old_file):
            try:
                os.remove(old_file)
            except Exception as e:
                print(f"Error removing old file: {str(e)}")
        
        initial_data = {
            "session_info": {
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": None,
                "filename": filename
            },
            "alert_summary": {
                "total_alerts": 0,
                "time_window": "24 hours",
                "categories": {
                    "HIGH_PRIORITY": 0,
                    "MEDIUM_PRIORITY": 0,
                    "LOW_PRIORITY": 0,
                    "IGNORED": 0
                }
            },
            "alerts": [],
            "system_performance": {
                "total_frames_processed": 0,
                "average_processing_time": "0.0 seconds",
                "false_positives": 0,
                "false_negatives": 0,
                "accuracy": "0%"
            },
            "alert_statistics": {
                "by_category": {
                    "HIGH_PRIORITY": 0,
                    "MEDIUM_PRIORITY": 0,
                    "LOW_PRIORITY": 0,
                    "IGNORED": 0
                },
                "by_object": {}
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(initial_data, f, indent=4)
        
        return filepath
    
    def get_available_sessions(self):
        """Get list of available alert sessions"""
        pattern = os.path.join(self.base_dir, "alerts_*.json")
        files = glob.glob(pattern)
        sessions = []
        
        for file in files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    sessions.append({
                        'filename': os.path.basename(file),
                        'start_time': data['session_info']['start_time'],
                        'end_time': data['session_info']['end_time'],
                        'total_alerts': data['alert_summary']['total_alerts']
                    })
            except Exception as e:
                print(f"Error reading session file {file}: {str(e)}")
        
        return sorted(sessions, key=lambda x: x['start_time'], reverse=True)
    
    def add_alert(self, object_type, confidence, category, description=""):
        """Add a new alert to the queue"""
        alert = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "object": object_type,
            "confidence": confidence,
            "category": category,
            "description": description,
            "status": "NEW"
        }
        self.alert_queue.put(alert)
    
    def start_processing(self):
        """Start the alert processing thread"""
        self.running = True
        self.process_thread = threading.Thread(target=self._process_alerts)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def stop_processing(self):
        """Stop the alert processing thread"""
        self.running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
            # Update end time in the alert file
            self._update_session_end_time()
    
    def _update_session_end_time(self):
        """Update the end time in the current session file"""
        with self.lock:
            try:
                with open(self.alert_file, 'r') as f:
                    data = json.load(f)
                
                data['session_info']['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with open(self.alert_file, 'w') as f:
                    json.dump(data, f, indent=4)
            except Exception as e:
                print(f"Error updating session end time: {str(e)}")
    
    def _process_alerts(self):
        """Process alerts from the queue and update the alert file"""
        while self.running:
            try:
                # Process alerts in the queue
                while not self.alert_queue.empty():
                    alert = self.alert_queue.get()
                    self._update_alert_file(alert)
                    self.alert_queue.task_done()
                
                # Sleep briefly to prevent high CPU usage
                time.sleep(0.1)
            except Exception as e:
                print(f"Error processing alerts: {str(e)}")
    
    def _update_alert_file(self, new_alert):
        """Update the alert file with a new alert"""
        with self.lock:
            try:
                # Read current data
                with open(self.alert_file, 'r') as f:
                    data = json.load(f)
                
                # Update alerts list
                data['alerts'].append(new_alert)
                
                # Update summary
                data['alert_summary']['total_alerts'] += 1
                data['alert_summary']['categories'][new_alert['category']] += 1
                
                # Update statistics
                data['alert_statistics']['by_category'][new_alert['category']] += 1
                
                # Update object statistics
                if new_alert['object'] in data['alert_statistics']['by_object']:
                    data['alert_statistics']['by_object'][new_alert['object']] += 1
                else:
                    data['alert_statistics']['by_object'][new_alert['object']] = 1
                
                # Write updated data
                with open(self.alert_file, 'w') as f:
                    json.dump(data, f, indent=4)
                    
            except Exception as e:
                print(f"Error updating alert file: {str(e)}")
    
    def update_performance_metrics(self, frames_processed, processing_time, false_positives, false_negatives, accuracy):
        """Update system performance metrics"""
        with self.lock:
            try:
                with open(self.alert_file, 'r') as f:
                    data = json.load(f)
                
                data['system_performance'].update({
                    "total_frames_processed": frames_processed,
                    "average_processing_time": f"{processing_time:.1f} seconds",
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "accuracy": f"{accuracy:.1f}%"
                })
                
                with open(self.alert_file, 'w') as f:
                    json.dump(data, f, indent=4)
                    
            except Exception as e:
                print(f"Error updating performance metrics: {str(e)}")
    
    def get_alert_summary(self, session_file=None):
        """Get the current alert summary"""
        try:
            file_to_read = session_file if session_file else self.alert_file
            with open(file_to_read, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading alert summary: {str(e)}")
            return None 