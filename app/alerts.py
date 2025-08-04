from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import logging
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import json

class AlertSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertStatus(str, Enum):
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    FALSE_POSITIVE = "FALSE_POSITIVE"

class AlertType(str, Enum):
    THRESHOLD = "THRESHOLD"
    ANOMALY = "ANOMALY"
    SCHEDULED = "SCHEDULED"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"

class AlertRule(BaseModel):
    """Defines a rule for generating alerts"""
    name: str
    description: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: str  # JSON string representing the condition
    check_interval: int = 300  # seconds
    active: bool = True
    recipients: List[str] = []
    metadata: Dict[str, Any] = {}

class Alert(BaseModel):
    """Represents an alert instance"""
    id: str
    rule_name: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.OPEN
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}
    affected_resources: List[Dict[str, Any]] = []

class AlertManager:
    """Manages alert generation and processing"""
    
    def __init__(self, db_session=None):
        self.logger = logging.getLogger(__name__)
        self.db_session = db_session
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
    
    def add_rule(self, rule: AlertRule) -> bool:
        """Add or update an alert rule"""
        self.rules[rule.name] = rule
        return True
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove an alert rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            return True
        return False
    
    def check_alerts(self, query_results: Dict[str, Any], query_text: str = None) -> Dict[str, Any]:
        """
        Check query results against alert rules and return any triggered alerts.
        
        Args:
            query_results: Dictionary containing query results with 'results' key
            query_text: The original query text (optional)
            
        Returns:
            Dictionary with 'alert_triggered' flag and 'alert_details' if any alerts were triggered
        """
        triggered_alerts = []
        
        # If no results or empty results, return default response
        if not query_results or 'results' not in query_results or not query_results['results']:
            return {
                'alert_triggered': False,
                'alert_details': None
            }
            
        # Get the results list
        results = query_results['results']
        
        # Check each rule against each result
        for rule in self.rules.values():
            if not rule.active:
                continue
                
            for result in results:
                try:
                    # Check if this result triggers the rule
                    alert = self.evaluate_rule(rule, result)
                    if alert:
                        alert_details = {
                            'rule_name': rule.name,
                            'severity': rule.severity.value,
                            'description': f"Alert triggered by rule: {rule.name}",
                            'details': result,
                            'timestamp': datetime.utcnow().isoformat(),
                            'query': query_text
                        }
                        triggered_alerts.append(alert_details)
                except Exception as e:
                    self.logger.error(f"Error evaluating rule {rule.name} on result {result}: {str(e)}")
        
        # Return the appropriate response based on whether any alerts were triggered
        if triggered_alerts:
            return {
                'alert_triggered': True,
                'alert_details': triggered_alerts[0]  # Return the first alert for now
            }
        else:
            return {
                'alert_triggered': False,
                'alert_details': None
            }
        
    def evaluate_rule(self, rule: AlertRule, data: Dict[str, Any]) -> Optional[Alert]:
        """Evaluate a rule against the provided data"""
        try:
            # Here you would implement the actual rule evaluation logic
            # This is a simplified example that checks if the condition is met
            condition = json.loads(rule.condition)
            field = condition.get('field')
            operator = condition.get('operator')
            value = condition.get('value')
            
            if field not in data:
                return None
                
            field_value = data[field]
            condition_met = False
            
            # Simple condition evaluation
            if operator == 'gt':
                condition_met = field_value > value
            elif operator == 'lt':
                condition_met = field_value < value
            elif operator == 'eq':
                condition_met = field_value == value
            elif operator == 'neq':
                condition_met = field_value != value
            elif operator == 'contains':
                condition_met = value in field_value
            
            if condition_met:
                return self._create_alert(rule, data)
                
        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule.name}: {str(e)}")
        
        return None
    
    def _create_alert(self, rule: AlertRule, data: Dict[str, Any]) -> Alert:
        """Create a new alert instance"""
        alert_id = f"alert_{int(datetime.utcnow().timestamp())}"
        
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            title=f"{rule.severity} Alert: {rule.name}",
            description=rule.description,
            severity=rule.severity,
            metadata={
                **rule.metadata,
                "trigger_data": data
            }
        )
        
        self.active_alerts[alert_id] = alert
        self._notify(alert)
        
        return alert
    
    def _notify(self, alert: Alert):
        """Send alert notifications"""
        # This would be connected to your notification system
        self.logger.info(f"ALERT: {alert.title} - {alert.description}")
        
        # Example: Send email, Slack message, etc.
        if self.db_session:
            # Save to database
            pass
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """Mark an alert as resolved"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.updated_at = datetime.utcnow()
            alert.resolved_at = datetime.utcnow()
            if resolution_notes:
                alert.metadata["resolution_notes"] = resolution_notes
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            # Log resolution
            self.logger.info(f"RESOLVED ALERT {alert_id}: {resolution_notes}")
            
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history (would be from database in production)"""
        # This is a placeholder - in production, this would query a database
        return []

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create an alert manager
    manager = AlertManager()
    
    # Define an alert rule
    rule = AlertRule(
        name="high_severity_alerts",
        description="Alert when high severity alerts exceed threshold",
        alert_type=AlertType.THRESHOLD,
        severity=AlertSeverity.HIGH,
        condition=json.dumps({
            "field": "high_severity_count",
            "operator": "gt",
            "value": 10
        }),
        recipients=["security-team@example.com"],
        metadata={"category": "security"}
    )
    
    # Add the rule
    manager.add_rule(rule)
    
    # Simulate checking some data
    test_data = {
        "high_severity_count": 15,
        "medium_severity_count": 25,
        "low_severity_count": 50
    }
    
    # Evaluate the rule
    alert = manager.evaluate_rule(rule, test_data)
    
    if alert:
        print(f"Generated alert: {alert.title}")
        print(f"Description: {alert.description}")
        print(f"Severity: {alert.severity}")
