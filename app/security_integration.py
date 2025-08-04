from typing import Dict, List, Any, Optional, Union, AsyncGenerator
import aiohttp
import json
import logging
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime, timedelta
import asyncio
from typing import List, Dict, Any, Optional, Union

class SecurityToolType(str, Enum):
    SIEM = "siem"
    EDR = "edr"
    VULN_SCANNER = "vulnerability_scanner"
    IDS_IPS = "ids_ips"
    FIREWALL = "firewall"
    IAM = "iam"
    CLOUD_PROVIDER = "cloud_provider"
    THREAT_INTEL = "threat_intelligence"

class SecurityToolAuth(BaseModel):
    """Authentication credentials for a security tool"""
    auth_type: str = "api_key"  # api_key, oauth2, basic, etc.
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    token_url: Optional[HttpUrl] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    scopes: List[str] = []

class SecurityToolConfig(BaseModel):
    """Configuration for a security tool integration"""
    name: str
    tool_type: SecurityToolType
    base_url: HttpUrl
    auth: SecurityToolAuth
    enabled: bool = True
    verify_ssl: bool = True
    timeout: int = 30  # seconds
    rate_limit: int = 10  # requests per second
    metadata: Dict[str, Any] = {}

class SecurityToolResponse(BaseModel):
    """Standardized response from security tools"""
    success: bool
    status_code: int
    data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

class SecurityToolIntegration:
    """Base class for security tool integrations"""
    
    def __init__(self, config: SecurityToolConfig):
        self.config = config
        self.logger = logging.getLogger(f"security_tool.{config.name}")
        self.session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = asyncio.Semaphore(config.rate_limit)
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        """Establish connection to the security tool"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                base_url=str(self.config.base_url),
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                connector=aiohttp.TCPConnector(verify_ssl=self.config.verify_ssl)
            )
            self.logger.info(f"Connected to {self.config.name}")
    
    async def disconnect(self):
        """Close connection to the security tool"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            self.logger.info(f"Disconnected from {self.config.name}")
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> SecurityToolResponse:
        """Make an authenticated request to the security tool"""
        if not self.session or self.session.closed:
            await self.connect()
        
        url = str(self.config.base_url).rstrip("/") + "/" + endpoint.lstrip("/")
        
        # Add authentication headers
        auth_headers = self._get_auth_headers()
        request_headers = {**auth_headers, **(headers or {})}
        
        try:
            async with self._rate_limiter:
                async with self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    headers=request_headers,
                    ssl=None if self.config.verify_ssl else False
                ) as response:
                    response_data = await self._process_response(response)
                    return SecurityToolResponse(
                        success=200 <= response.status < 300,
                        status_code=response.status,
                        data=response_data,
                        metadata={"url": str(response.url)}
                    )
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"Request failed: {str(e)}")
            return SecurityToolResponse(
                success=False,
                status_code=0,
                error=str(e)
            )
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers based on auth type"""
        auth = self.config.auth
        
        if auth.auth_type == "api_key" and auth.api_key:
            return {"Authorization": f"Bearer {auth.api_key}"}
            
        elif auth.auth_type == "basic" and auth.username and auth.password:
            import base64
            credentials = f"{auth.username}:{auth.password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            return {"Authorization": f"Basic {encoded_credentials}"}
            
        # Add more auth types as needed
        return {}
    
    async def _process_response(self, response: aiohttp.ClientResponse) -> Any:
        """Process the API response"""
        content_type = response.headers.get("Content-Type", "")
        
        if "application/json" in content_type:
            try:
                return await response.json()
            except json.JSONDecodeError:
                return await response.text()
        
        return await response.text()
    
    # Common security tool operations
    async def get_alerts(
        self, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> SecurityToolResponse:
        """Get security alerts from the tool"""
        raise NotImplementedError("Subclasses must implement get_alerts")
    
    async def get_vulnerabilities(
        self, 
        asset_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> SecurityToolResponse:
        """Get vulnerability data from the tool"""
        raise NotImplementedError("Subclasses must implement get_vulnerabilities")
    
    async def get_assets(self, **filters) -> SecurityToolResponse:
        """Get asset information from the tool"""
        raise NotImplementedError("Subclasses must implement get_assets")

# Example implementation for a SIEM tool
class SIEMIntegration(SecurityToolIntegration):
    """Integration with a SIEM (Security Information and Event Management) system"""
    
    async def get_alerts(
        self, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> SecurityToolResponse:
        """Get security alerts from SIEM"""
        params = {
            "limit": limit,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "severity": severity
        }
        
        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}
        
        return await self._make_request(
            method="GET",
            endpoint="/api/v1/alerts",
            params=params
        )
    
    async def search_events(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> SecurityToolResponse:
        """Search for events in the SIEM"""
        data = {
            "query": query,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "limit": limit
        }
        
        return await self._make_request(
            method="POST",
            endpoint="/api/v1/events/search",
            json_data=data
        )

# Example implementation for a vulnerability scanner
class VulnerabilityScannerIntegration(SecurityToolIntegration):
    """Integration with a vulnerability scanner"""
    
    async def get_vulnerabilities(
        self, 
        asset_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> SecurityToolResponse:
        """Get vulnerabilities from the scanner"""
        params = {
            "asset_id": asset_id,
            "severity": severity,
            "limit": limit
        }
        
        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}
        
        return await self._make_request(
            method="GET",
            endpoint="/api/v1/vulnerabilities",
            params=params
        )
    
    async def get_asset_vulnerabilities(self, asset_id: str) -> SecurityToolResponse:
        """Get vulnerabilities for a specific asset"""
        return await self._make_request(
            method="GET",
            endpoint=f"/api/v1/assets/{asset_id}/vulnerabilities"
        )

# Example implementation for a cloud provider
class CloudProviderIntegration(SecurityToolIntegration):
    """Integration with a cloud provider (AWS, Azure, GCP)"""
    
    async def get_cloud_resources(self, resource_type: Optional[str] = None) -> SecurityToolResponse:
        """Get cloud resources"""
        endpoint = "/api/v1/resources"
        if resource_type:
            endpoint += f"/{resource_type}"
            
        return await self._make_request(
            method="GET",
            endpoint=endpoint
        )
    
    async def get_security_findings(
        self,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> SecurityToolResponse:
        """Get security findings from the cloud provider"""
        params = {
            "severity": severity,
            "status": status,
            "limit": limit
        }
        
        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}
        
        return await self._make_request(
            method="GET",
            endpoint="/api/v1/security/findings",
            params=params
        )

class SecurityToolManager:
    """Manages multiple security tool integrations"""
    
    def __init__(self):
        self.tools: Dict[str, SecurityToolIntegration] = {}
        self.logger = logging.getLogger("security_tool_manager")
    
    def add_tool(self, tool: SecurityToolIntegration) -> bool:
        """Add a security tool integration"""
        if not tool.config.enabled:
            self.logger.warning(f"Tool {tool.config.name} is disabled")
            return False
            
        self.tools[tool.config.name] = tool
        self.logger.info(f"Added tool: {tool.config.name} ({tool.config.tool_type})")
        return True
    
    def get_tool(self, name: str) -> Optional[SecurityToolIntegration]:
        """Get a security tool by name"""
        return self.tools.get(name)
    
    def get_tools_by_type(self, tool_type: SecurityToolType) -> List[SecurityToolIntegration]:
        """Get all tools of a specific type"""
        return [
            tool for tool in self.tools.values() 
            if tool.config.tool_type == tool_type
        ]
    
    async def collect_alerts(
        self, 
        tool_types: Optional[List[SecurityToolType]] = None,
        **kwargs
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Collect alerts from all tools or specific tool types"""
        alerts = {}
        
        for name, tool in self.tools.items():
            if tool_types and tool.config.tool_type not in tool_types:
                continue
                
            try:
                response = await tool.get_alerts(**kwargs)
                if response.success and response.data:
                    alerts[name] = response.data
            except Exception as e:
                self.logger.error(f"Error collecting alerts from {name}: {str(e)}")
        
        return alerts

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Create a security tool manager
        manager = SecurityToolManager()
        
        # Example SIEM configuration
        siem_config = SecurityToolConfig(
            name="Enterprise SIEM",
            tool_type=SecurityToolType.SIEM,
            base_url="https://siem.example.com",
            auth=SecurityToolAuth(
                auth_type="api_key",
                api_key="your-api-key-here"
            )
        )
        
        # Add SIEM integration
        siem = SIEMIntegration(siem_config)
        manager.add_tool(siem)
        
        # Example vulnerability scanner configuration
        vuln_scanner_config = SecurityToolConfig(
            name="Nexpose Scanner",
            tool_type=SecurityToolType.VULN_SCANNER,
            base_url="https://nexpose.example.com",
            auth=SecurityToolAuth(
                auth_type="basic",
                username="api_user",
                password="api_password"
            )
        )
        
        # Add vulnerability scanner integration
        vuln_scanner = VulnerabilityScannerIntegration(vuln_scanner_config)
        manager.add_tool(vuln_scanner)
        
        # Collect alerts from all tools
        alerts = await manager.collect_alerts(
            limit=10,
            start_time=datetime.utcnow() - timedelta(days=1)
        )
        
        print(f"Collected alerts: {json.dumps(alerts, indent=2)}")
    
    asyncio.run(main())
