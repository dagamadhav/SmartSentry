import time
from datetime import datetime
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os

class NotificationManager:
    def __init__(self, config_file="config/notification_config.json"):
        self.high_priority_alerts = {}
        self.alert_threshold = 2  # seconds
        self.notification_thread = None
        self.running = False
        self.load_config(config_file)
        
    def load_config(self, config_file):
        """Load notification configuration"""
        default_config = {
            "email": {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "21btrse014@jainuniveristy.ac.in",
                "sender_password": "Madhav21btrse014",  # You'll need to add your app password here
                "recipient_email": "dagamadhav1@gmail.com"
            },
            "notification_sound": True,
            "desktop_notification": True
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = default_config
                # Create config directory if it doesn't exist
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=4)
                print(f"Created notification config file at {config_file}")
                print("Please add your Gmail app password to enable email notifications")
        except Exception as e:
            print(f"Error loading notification config: {str(e)}")
            self.config = default_config
    
    def start(self):
        """Start the notification monitoring thread"""
        self.running = True
        self.notification_thread = threading.Thread(target=self._monitor_alerts)
        self.notification_thread.daemon = True
        self.notification_thread.start()
    
    def stop(self):
        """Stop the notification monitoring thread"""
        self.running = False
        if self.notification_thread:
            self.notification_thread.join()
    
    def add_alert(self, alert_id, alert_type, message):
        """Add a new high-priority alert"""
        current_time = time.time()
        self.high_priority_alerts[alert_id] = {
            'type': alert_type,
            'message': message,
            'start_time': current_time,
            'notified': False
        }
    
    def remove_alert(self, alert_id):
        """Remove an alert"""
        if alert_id in self.high_priority_alerts:
            del self.high_priority_alerts[alert_id]
    
    def _monitor_alerts(self):
        """Monitor high-priority alerts and send notifications"""
        while self.running:
            current_time = time.time()
            alerts_to_notify = []
            
            # Check for alerts that have exceeded the threshold
            for alert_id, alert in self.high_priority_alerts.items():
                if not alert['notified'] and (current_time - alert['start_time']) >= self.alert_threshold:
                    alerts_to_notify.append(alert)
                    alert['notified'] = True
            
            # Send notifications for alerts that need it
            if alerts_to_notify:
                self._send_notifications(alerts_to_notify)
            
            time.sleep(0.5)  # Check every 500ms
    
    def _send_notifications(self, alerts):
        """Send notifications for alerts"""
        # Prepare notification message
        message = "High Priority Alert!\n\n"
        for alert in alerts:
            message += f"- {alert['type']}: {alert['message']}\n"
        
        # Send email notification if enabled
        if self.config['email']['enabled'] and self.config['email']['sender_password']:
            self._send_email_notification(message)
        
        # Send desktop notification if enabled
        if self.config['desktop_notification']:
            self._send_desktop_notification(message)
        
        # Play notification sound if enabled
        if self.config['notification_sound']:
            self._play_notification_sound()
    
    def _send_email_notification(self, message):
        """Send email notification"""
        try:
            if not all([self.config['email']['sender_email'], 
                       self.config['email']['sender_password'],
                       self.config['email']['recipient_email']]):
                print("Email configuration incomplete. Please check all email settings.")
                return
            
            print(f"Attempting to send email to {self.config['email']['recipient_email']}")
            print(f"Using SMTP server: {self.config['email']['smtp_server']}:{self.config['email']['smtp_port']}")
            
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['sender_email']
            msg['To'] = self.config['email']['recipient_email']
            msg['Subject'] = "Security Alert: High Priority Detection"
            
            msg.attach(MIMEText(message, 'plain'))
            
            print("Connecting to SMTP server...")
            server = smtplib.SMTP(self.config['email']['smtp_server'], 
                                self.config['email']['smtp_port'])
            
            print("Starting TLS connection...")
            server.starttls()
            
            print("Attempting login...")
            server.login(self.config['email']['sender_email'],
                        self.config['email']['sender_password'])
            
            print("Sending email...")
            server.send_message(msg)
            server.quit()
            print("Email notification sent successfully")
            
        except smtplib.SMTPAuthenticationError as e:
            print(f"SMTP Authentication Error: {str(e)}")
            print("Please verify your email and app password are correct.")
            print("Make sure you're using an App Password, not your regular Gmail password.")
        except smtplib.SMTPException as e:
            print(f"SMTP Error: {str(e)}")
            print("Please check your internet connection and SMTP settings.")
        except Exception as e:
            print(f"Error sending email notification: {str(e)}")
            print("Please check your email configuration in notification_config.json")
    
    def _send_desktop_notification(self, message):
        """Send desktop notification"""
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast("Security Alert",
                             message,
                             duration=10,
                             threaded=True)
        except Exception as e:
            print(f"Error sending desktop notification: {str(e)}")
    
    def _play_notification_sound(self):
        """Play notification sound"""
        try:
            import winsound
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
        except Exception as e:
            print(f"Error playing notification sound: {str(e)}") 