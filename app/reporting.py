from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from xhtml2pdf import pisa
import base64
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import json
import os
try:
    from .alerts import AlertManager, AlertSeverity, Alert
except ImportError:
    from alerts import AlertManager, AlertSeverity, Alert
try:
    from .visualization import SecurityVisualizer, ChartType
except ImportError:
    from visualization import SecurityVisualizer, ChartType

class ReportFormat(str, Enum):
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    JSON = "json"

class ReportType(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"

class ReportTemplate:
    """Handles report template management"""
    
    def __init__(self, template_dir: str = "templates/reports"):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
    
    def get_template(self, template_name: str):
        """Get a template by name"""
        return self.env.get_template(f"{template_name}.html")
    
    async def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a template with the given context"""
        template = self.get_template(template_name)
        return await template.render_async(**context)

class ReportGenerator:
    """Generates reports from templates and data"""
    
    def __init__(self, template_dir: str = "templates/reports"):
        self.template = ReportTemplate(template_dir)
        self.visualizer = SecurityVisualizer()
        self.logger = logging.getLogger(__name__)
    
    async def generate_report(
        self,
        report_type: ReportType,
        format: ReportFormat = ReportFormat.PDF,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        title: str = "Security Report",
        description: str = "Automated security report",
        data: Optional[Dict[str, Any]] = None,
        template_name: str = "default"
    ) -> bytes:
        """Generate a report in the specified format"""
        # Set default date range if not provided
        end_date = end_date or datetime.utcnow()
        if not start_date:
            if report_type == ReportType.DAILY:
                start_date = end_date - timedelta(days=1)
            elif report_type == ReportType.WEEKLY:
                start_date = end_date - timedelta(weeks=1)
            elif report_type == ReportType.MONTHLY:
                start_date = end_date - timedelta(days=30)
            else:
                start_date = end_date - timedelta(days=1)  # Default to daily
        
        # Prepare context
        context = {
            "title": title,
            "description": description,
            "generated_at": datetime.utcnow(),
            "start_date": start_date,
            "end_date": end_date,
            "report_type": report_type.value,
            "data": data or {}
        }
        
        # Add visualizations if needed
        if data and "alerts" in data:
            context["alert_chart"] = self._generate_alert_chart(data["alerts"])
        
        # Render the template
        try:
            html_content = await self.template.render_template(template_name, context)
            
            # Convert to requested format
            if format == ReportFormat.HTML:
                return html_content.encode('utf-8')
            
            elif format == ReportFormat.PDF:
                return await self._generate_pdf(html_content)
                
            elif format == ReportFormat.CSV:
                return self._generate_csv(data or {})
                
            elif format == ReportFormat.JSON:
                return json.dumps({
                    "metadata": {
                        "title": title,
                        "description": description,
                        "generated_at": context["generated_at"].isoformat(),
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "report_type": report_type.value
                    },
                    "data": data
                }, indent=2).encode('utf-8')
                
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise
    
    def _generate_alert_chart(self, alerts: List[Dict[str, Any]]) -> str:
        """Generate a chart for alerts data"""
        if not alerts:
            return ""
            
        # Process alerts data for visualization
        severity_counts = {}
        for alert in alerts:
            severity = alert.get("severity", "UNKNOWN")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        chart_data = [{"severity": k, "count": v} for k, v in severity_counts.items()]
        
        # Generate and return base64 encoded chart image
        return self.visualizer.create_plot(
            chart_data,
            ChartType.BAR,
            x="severity",
            y="count",
            title="Alerts by Severity",
            color="severity"
        )
    
    async def _generate_pdf(self, html_content: str) -> bytes:
        """Convert HTML content to PDF using xhtml2pdf"""
        try:
            from io import BytesIO
            pdf_bytes = BytesIO()
            pisa.CreatePDF(html_content, dest=pdf_bytes)
            return pdf_bytes.getvalue()
        except Exception as e:
            self.logger.error(f"Error generating PDF: {str(e)}")
            raise
    
    def _generate_csv(self, data: Dict[str, Any]) -> bytes:
        """Convert data to CSV format"""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        if data and "alerts" in data and data["alerts"]:
            writer.writerow(data["alerts"][0].keys())
            # Write data rows
            for row in data["alerts"]:
                writer.writerow(row.values())
        
        return output.getvalue().encode('utf-8')

class ReportScheduler:
    """Handles scheduling and distribution of reports"""
    
    def __init__(self, report_dir: str = "reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    async def schedule_report(
        self,
        report_type: ReportType,
        recipients: List[str],
        schedule: str = "0 9 * * *",  # Default: 9 AM daily
        format: ReportFormat = ReportFormat.PDF,
        **report_kwargs
    ) -> str:
        """Schedule a recurring report"""
        from crontab import CronTab
        
        # Generate a unique ID for this scheduled report
        report_id = f"{report_type}_{int(datetime.utcnow().timestamp())}"
        
        # In a production system, this would save to a database
        schedule_file = self.report_dir / f"{report_id}.json"
        schedule_data = {
            "report_id": report_id,
            "report_type": report_type.value,
            "schedule": schedule,
            "recipients": recipients,
            "format": format.value,
            "last_run": None,
            "next_run": self._calculate_next_run(schedule),
            "created_at": datetime.utcnow().isoformat(),
            "report_kwargs": report_kwargs
        }
        
        async with aiofiles.open(schedule_file, 'w') as f:
            await f.write(json.dumps(schedule_data, indent=2, default=str))
        
        # In production, this would add to the system's crontab
        self._update_crontab()
        
        return report_id
    
    def _calculate_next_run(self, schedule: str) -> str:
        """Calculate the next run time from a cron schedule"""
        from crontab import CronTab
        from datetime import datetime
        
        # This is a simplified version - in production, use a proper scheduler
        cron = CronTab(schedule)
        next_run = datetime.now() + timedelta(seconds=int(cron.next()))
        return next_run.isoformat()
    
    def _update_crontab(self):
        """Update the system crontab with scheduled reports"""
        # In a real implementation, this would update the system crontab
        # or add the job to a task queue like Celery
        pass
    
    async def send_report(
        self,
        report_content: bytes,
        recipients: List[str],
        subject: str = "Security Report",
        format: ReportFormat = ReportFormat.PDF,
        smtp_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send a report via email"""
        if not smtp_config:
            smtp_config = {
                "host": os.getenv("SMTP_HOST", "smtp.example.com"),
                "port": int(os.getenv("SMTP_PORT", 587)),
                "username": os.getenv("SMTP_USERNAME"),
                "password": os.getenv("SMTP_PASSWORD"),
                "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true"
            }
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = smtp_config.get("from", "security-reports@example.com")
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject
            
            # Add text part
            msg.attach(MIMEText("Please find the attached security report.", "plain"))
            
            # Add attachment
            filename = f"security_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            if format == ReportFormat.PDF:
                filename += ".pdf"
                attachment = MIMEApplication(report_content, _subtype="pdf")
            elif format == ReportFormat.CSV:
                filename += ".csv"
                attachment = MIMEApplication(report_content, _subtype="csv")
            else:  # Default to HTML
                filename += ".html"
                attachment = MIMEApplication(report_content, _subtype="html")
            
            attachment.add_header("Content-Disposition", "attachment", filename=filename)
            msg.attach(attachment)
            
            # Send email
            async with aiosmtplib.SMTP(
                hostname=smtp_config["host"],
                port=smtp_config["port"],
                use_tls=smtp_config["use_tls"]
            ) as server:
                if smtp_config.get("username") and smtp_config.get("password"):
                    await server.login(smtp_config["username"], smtp_config["password"])
                
                await server.send_message(msg)
                
            self.logger.info(f"Report sent to {', '.join(recipients)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending report: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize components
        report_generator = ReportGenerator()
        report_scheduler = ReportScheduler()
        
        # Sample data
        sample_data = {
            "alerts": [
                {"id": "1", "severity": "HIGH", "title": "Unauthorized Access", "description": "Multiple failed login attempts"},
                {"id": "2", "severity": "MEDIUM", "title": "Vulnerability Found", "description": "Outdated software detected"},
                {"id": "3", "severity": "HIGH", "title": "Data Exfiltration", "description": "Suspicious data transfer detected"},
            ]
        }
        
        # Generate a report
        report = await report_generator.generate_report(
            report_type=ReportType.DAILY,
            format=ReportFormat.PDF,
            title="Daily Security Report",
            description="Automated daily security report",
            data=sample_data
        )
        
        # Save the report
        with open("daily_security_report.pdf", "wb") as f:
            f.write(report)
        
        # Schedule a weekly report
        report_id = await report_scheduler.schedule_report(
            report_type=ReportType.WEEKLY,
            recipients=["security-team@example.com"],
            schedule="0 9 * * 1",  # 9 AM every Monday
            format=ReportFormat.PDF,
            title="Weekly Security Report",
            description="Automated weekly security report",
            data=sample_data
        )
        
        print(f"Generated report and scheduled weekly report with ID: {report_id}")
    
    asyncio.run(main())
