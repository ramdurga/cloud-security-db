import os
import json
import asyncio
from typing import Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks, File, UploadFile, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import logging
import time
import json
import os
import asyncio
from pathlib import Path

# Import core modules - handle both module and direct execution
try:
    from .database import get_db, get_db_schema
    from .llm_service import LLMQueryService
    from .agent_service import SecurityDBAgent, AgentService
    from .visualization import SecurityVisualizer, ChartType
    from .alerts import AlertManager, Alert, AlertRule, AlertSeverity, AlertStatus, AlertType
    from .reporting import ReportGenerator, ReportScheduler, ReportFormat, ReportType
    from .security_integration import SecurityToolManager, SecurityToolConfig, SecurityToolAuth, SecurityToolType
    from .debug import router as debug_router
except ImportError:
    from database import get_db, get_db_schema
    from llm_service import LLMQueryService
    from agent_service import SecurityDBAgent, AgentService
    from visualization import SecurityVisualizer, ChartType
    from alerts import AlertManager, Alert, AlertRule, AlertSeverity, AlertStatus, AlertType
    from reporting import ReportGenerator, ReportScheduler, ReportFormat, ReportType
    from security_integration import SecurityToolManager, SecurityToolConfig, SecurityToolAuth, SecurityToolType
    from debug import router as debug_router
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging
import asyncio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cloud Security DB API",
    description="API for querying cloud security data with AI-powered analysis",
    version="1.0.0",
    contact={
        "name": "Security Team",
        "email": "security@example.com"
    },
    license_info={
        "name": "Proprietary"
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=json.loads(os.getenv("ALLOWED_ORIGINS", "[\"*\"]")),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include debug routes with empty prefix since the routes already include /debug
app.include_router(debug_router, prefix="", tags=["debug"])

# Create necessary directories
os.makedirs("reports", exist_ok=True)
os.makedirs("static/charts", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize services
llm_service = LLMQueryService()
alert_manager = AlertManager()
report_generator = ReportGenerator()
report_scheduler = ReportScheduler()
security_tool_manager = SecurityToolManager()

class QueryRequest(BaseModel):
    query: str
    use_agent: bool = False
    generate_chart: bool = False
    chart_type: Optional[ChartType] = None
    chart_x: Optional[str] = None
    chart_y: Optional[str] = None
    chart_title: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    sql: str
    results: List[Dict[str, Any]]
    error: Optional[str] = None
    execution_time: float
    agent_steps: Optional[List[Dict[str, Any]]] = None
    chart_data: Optional[str] = None  # Base64 encoded chart image
    alert_triggered: bool = False
    alert_details: Optional[Dict[str, Any]] = None

class AlertRequest(BaseModel):
    rule_name: str
    condition: Dict[str, Any]
    severity: AlertSeverity
    description: str
    recipients: List[str]
    active: bool = True

class ReportRequest(BaseModel):
    report_type: ReportType
    format: ReportFormat = ReportFormat.PDF
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    title: str = "Security Report"
    description: str = "Automated security report"
    recipients: List[str] = []
    schedule: Optional[str] = None  # Cron expression for scheduling

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.exception("Unhandled exception occurred")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

@app.get("/schema")
async def get_schema(db: Session = Depends(get_db)):
    """Get the database schema"""
    return get_db_schema(db)

# Visualization endpoints
@app.get("/visualization/types")
async def get_visualization_types():
    """Get available visualization types"""
    return SecurityVisualizer.get_chart_types()

# Alert endpoints
@app.post("/alerts/rules", status_code=201)
async def create_alert_rule(rule: AlertRule):
    """Create a new alert rule"""
    try:
        if alert_manager.add_rule(rule):
            return {"status": "success", "message": "Alert rule created", "rule": rule.dict()}
        else:
            raise HTTPException(status_code=400, detail="Failed to add alert rule")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/alerts/rules")
async def list_alert_rules():
    """List all alert rules"""
    return [rule.dict() for rule in alert_manager.rules.values()]

@app.get("/alerts/active")
async def get_active_alerts():
    """Get all active alerts"""
    return [alert.dict() for alert in alert_manager.get_active_alerts()]

# Reporting endpoints
@app.post("/reports/generate")
async def generate_report(
    report_request: ReportRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Generate a report"""
    try:
        # In a real app, you would fetch the actual data here
        report_data = {
            "alerts": [],
            "metrics": {},
            # Add more data as needed
        }
        
        # Generate the report
        report_content = await report_generator.generate_report(
            report_type=report_request.report_type,
            format=report_request.format,
            start_date=report_request.start_date,
            end_date=report_request.end_date,
            title=report_request.title,
            description=report_request.description,
            data=report_data
        )
        
        # Save the report
        filename = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        if report_request.format == ReportFormat.PDF:
            filename += ".pdf"
            content_type = "application/pdf"
        elif report_request.format == ReportFormat.CSV:
            filename += ".csv"
            content_type = "text/csv"
        else:  # Default to HTML
            filename += ".html"
            content_type = "text/html"
        
        filepath = f"reports/{filename}"
        with open(filepath, "wb") as f:
            f.write(report_content)
        
        # Schedule email if recipients are provided
        if report_request.recipients:
            background_tasks.add_task(
                report_scheduler.send_report,
                report_content=report_content,
                recipients=report_request.recipients,
                subject=report_request.title,
                format=report_request.format
            )
        
        # If it's a one-time report, return it
        if not report_request.schedule:
            return FileResponse(
                path=filepath,
                media_type=content_type,
                filename=filename
            )
        else:
            # Schedule the report
            schedule_id = await report_scheduler.schedule_report(
                report_type=report_request.report_type,
                recipients=report_request.recipients,
                schedule=report_request.schedule,
                format=report_request.format,
                **report_request.dict()
            )
            return {"status": "scheduled", "schedule_id": schedule_id}
            
    except Exception as e:
        logging.error(f"Error generating report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Security Tools Integration endpoints
@app.get("/security-tools")
async def list_security_tools():
    """List all configured security tools"""
    return [{"name": name, "type": tool.config.tool_type} 
            for name, tool in security_tool_manager.tools.items()]

@app.post("/security-tools/test-connection")
async def test_security_tool_connection(tool_name: str):
    """Test connection to a security tool"""
    tool = security_tool_manager.get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
    
    try:
        # Simple test - try to get a small amount of data
        if hasattr(tool, 'get_alerts'):
            result = await tool.get_alerts(limit=1)
        elif hasattr(tool, 'get_vulnerabilities'):
            result = await tool.get_vulnerabilities(limit=1)
        else:
            return {"status": "success", "message": f"Connection to {tool_name} successful"}
        
        if result.success:
            return {"status": "success", "message": f"Connection to {tool_name} successful"}
        else:
            return {"status": "error", "message": result.error}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection failed: {str(e)}")

# Query endpoint
@app.post("/query")
async def query_endpoint(
    query_request: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Process a natural language query and return results.
    
    Args:
        query_request: The query request containing the natural language query
        db: Database session
        
    Returns:
        QueryResponse: The results of the query execution
    """
    try:
        # Initialize the agent service
        agent_service = AgentService()
        
        # Process the query
        if query_request.use_agent:
            # Use the agent for more complex queries
            result = await agent_service.process_query(
                query=query_request.query,
                generate_chart=query_request.generate_chart,
                chart_type=query_request.chart_type,
                chart_x=query_request.chart_x,
                chart_y=query_request.chart_y,
                chart_title=query_request.chart_title,
                db=db
            )
        else:
            # Use direct SQL translation for simple queries
            result = await agent_service.direct_sql_query(
                query=query_request.query,
                db=db
            )
            
        # Check for alerts
        alert_manager = AlertManager()
        alert_result = alert_manager.check_alerts(result, query_request.query)
        
        if alert_result["alert_triggered"]:
            result["alert_triggered"] = True
            result["alert_details"] = alert_result["alert_details"]
            
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

# Debug endpoints
@app.get("/debug/schema", response_model=Dict[str, Any])
async def debug_schema():
    """Debug endpoint to inspect database schema and sample data"""
    try:
        agent = SecurityDBAgent()
        schema_info = await agent.inspect_database()
        return schema_info
    except Exception as e:
        logger.error(f"Error inspecting database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
