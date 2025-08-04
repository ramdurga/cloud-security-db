import asyncio
import os
import json
import time
from typing import Dict, Any, List, Optional, Union
from sqlalchemy.sql import text
from datetime import datetime
from dotenv import load_dotenv

from langchain.agents import AgentExecutor
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.prompt import SQL_PREFIX, SQL_SUFFIX
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage
from langchain_anthropic import ChatAnthropic

# Import visualization module - handle both module and direct execution
try:
    from .visualization import SecurityVisualizer
except ImportError:
    from visualization import SecurityVisualizer

# Load environment variables
load_dotenv()

class AgentService:
    """Service class for handling natural language to SQL queries with agentic capabilities."""
    
    def __init__(self):
        """Initialize the AgentService with required components."""
        self.llm = ChatAnthropic(
            temperature=0,
            model_name="claude-3-opus-20240229"
        )
        self.visualizer = SecurityVisualizer()
    
    async def process_query(
        self,
        query: str,
        generate_chart: bool = False,
        chart_type: Optional[str] = None,
        chart_x: Optional[str] = None,
        chart_y: Optional[str] = None,
        chart_title: Optional[str] = None,
        db_uri: Optional[str] = None,
        db: Any = None
    ) -> Dict[str, Any]:
        """
        Process a natural language query using the agent.
        
        Args:
            query: The natural language query
            generate_chart: Whether to generate a chart
            chart_type: Type of chart to generate
            chart_x: X-axis field for the chart
            chart_y: Y-axis field for the chart
            chart_title: Title for the chart
            db_uri: Database connection URI
            db: Database session (alternative to db_uri)
            
        Returns:
            Dict containing query results and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Initialize the security DB agent
            agent = SecurityDBAgent(db_uri=db_uri, db=db)
            
            # Process the query
            result = await agent.run_query(query)
            
            # Generate chart if requested
            chart_data = None
            if generate_chart and result.get('results'):
                chart_data = self.visualizer.generate_chart(
                    data=result['results'],
                    chart_type=chart_type,
                    x=chart_x,
                    y=chart_y,
                    title=chart_title
                )
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "query": query,
                "sql": result.get('sql', ''),
                "results": result.get('results', []),
                "execution_time": execution_time,
                "chart_data": chart_data,
                "error": None
            }
            
        except Exception as e:
            return {
                "query": query,
                "sql": "",
                "results": [],
                "execution_time": (datetime.utcnow() - start_time).total_seconds(),
                "error": str(e)
            }
    
    async def direct_sql_query(self, query: str, db_uri: str = None, db: Any = None) -> Dict[str, Any]:
        """
        Execute a direct SQL query without using the agent.
        
        Args:
            query: The SQL query to execute
            db_uri: Database connection URI
            db: Database session (alternative to db_uri)
            
        Returns:
            Dict containing query results and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            agent = SecurityDBAgent(db_uri=db_uri, db=db)
            result = await agent.execute_sql(query)
            
            return {
                "query": query,
                "sql": query,
                "results": result.get('results', []),
                "execution_time": (datetime.utcnow() - start_time).total_seconds(),
                "error": None
            }
            
        except Exception as e:
            return {
                "query": query,
                "sql": query,
                "results": [],
                "execution_time": (datetime.utcnow() - start_time).total_seconds(),
                "error": str(e)
            }

class SecurityDBAgent:
    def __init__(self, db_uri: str = None, db: Any = None):
        """
        Initialize the Security Database Agent with a SQLAlchemy database URI or connection.
        
        Args:
            db_uri: SQLAlchemy database URI. If not provided, will use environment variables.
            db: SQLAlchemy database connection object. If provided, will be used to get the engine.
        """
        if db is not None:
            # Use the provided database connection's engine
            self.engine = db.get_bind()
        else:
            # Initialize database connection from URI
            if not db_uri:
                db_uri = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
            from sqlalchemy import create_engine
            self.engine = create_engine(db_uri)
        
        # Initialize Claude with tool use
        self.llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229"),
            temperature=0,
            max_tokens=4000
        )
        
        # Store database metadata
        from sqlalchemy import inspect
        self.inspector = inspect(self.engine)
        
        # Initialize tools
        self.tools = self._setup_tools()
        
        # System prompt for the agent
        self.system_prompt = """You are a security analyst assistant that helps with database queries. 
        You have access to a PostgreSQL database with security-related information. 
        
        DATABASE SCHEMA OVERVIEW:
        - alerts: Contains security alerts with details like severity, status, and timestamps
        - customers: Contains customer organizations with subscription tiers (enterprise, business, basic)
        - rules: Contains detection rules that generate alerts
        - incidents: Groups related alerts into security incidents
        - incident_alerts: Links alerts to incidents in a many-to-many relationship
        
        IMPORTANT NOTES:
        1. Always use proper JOINs to relate tables (e.g., alerts.customer_id = customers.customer_id)
        2. For customer types, use the 'subscription_tier' column in the customers table
        3. Alert severities are stored in UPPERCASE: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO' - always use UPPERCASE in queries
        4. Always include relevant fields in SELECT (e.g., don't use SELECT *)
        5. Include appropriate ORDER BY clauses for time-series data
        6. Use proper date/time formatting in queries
        7. Always include a LIMIT clause for potentially large result sets
        
        When generating SQL:
        1. First examine the schema using the get_schema tool if needed
        2. Write a precise SQL query that answers the question
        3. Include only the necessary fields in the SELECT clause
        4. Use proper JOIN conditions
        5. Add appropriate WHERE clauses for filtering
        6. Include ORDER BY and LIMIT as needed
        """
        
        # Initialize the agent
        self.agent = self._create_agent()
        
    
    def _create_custom_prompt(self):
        """Create a custom prompt for the security database agent"""
        return """You are a security database expert. Your task is to help analyze and query security-related data.
        
        When querying the database:
        1. Always consider security implications of the data being accessed
        2. Be precise with your queries to avoid exposing sensitive information
        3. Pay special attention to security-related fields like:
           - Access logs
           - Authentication attempts
           - Permission changes
           - Security alerts
        4. For subscription tiers, use these exact values (case-sensitive):
           - 'ENTERPRISE'
           - 'BUSINESS'
           - 'STANDARD'
        
        If a query would return sensitive data, consider if it should be masked or summarized.
        """
    
    def _setup_tools(self):
        """Set up tools for the agent to use"""
        from langchain_core.tools import Tool
        
        tools = [
            Tool(
                name="query_database",
                description="Execute a SQL query against the security database. Always use proper JOINs and include only necessary fields in the SELECT clause.",
                func=self._query_database_sync
            ),
            Tool(
                name="get_schema",
                description="Get the database schema information including tables, columns, and relationships",
                func=self._get_schema_sync
            )
        ]
        
        return tools
    
    def _query_database_sync(self, query: str) -> str:
        """Synchronous wrapper for query_database tool"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._execute_sql_tool({"query": query}))
            if result.get('success'):
                import json
                return json.dumps(result.get('results', []), indent=2)
            else:
                return f"Error: {result.get('error', 'Unknown error')}"
        finally:
            loop.close()
    
    def _get_schema_sync(self, _: str = "") -> str:
        """Synchronous wrapper for get_schema tool"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._get_schema_tool({}))
            if result.get('success'):
                import json
                return json.dumps(result.get('schema', {}), indent=2)
            else:
                return f"Error: {result.get('error', 'Unknown error')}"
        finally:
            loop.close()
    
    def _create_agent(self):
        """Create a LangChain agent with Claude's tool use capability"""
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain_core.prompts import PromptTemplate
        
        # Create a ReAct-style prompt template
        prompt = PromptTemplate.from_template("""
{system_prompt}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
        """)
        
        # Fill in the system prompt
        filled_prompt = prompt.partial(system_prompt=self.system_prompt)
        
        # Create the agent with ReAct pattern
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=filled_prompt,
        )
        
        # Create an agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=True
        )
        
        return agent_executor
    
    async def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool and return the result in a standardized format.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            
        Returns:
            Dict containing the tool execution results and metadata
        """
        try:
            print(f"\n=== Executing Tool: {tool_name} ===")
            print(f"Tool Input: {json.dumps(tool_input, indent=2)}")
            
            # Get the appropriate handler
            handler = self._tool_handlers.get(tool_name)
            if not handler:
                error_msg = f"Unknown tool: {tool_name}"
                print(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'results': []
                }
            
            # Execute the handler
            result = await handler(tool_input)
            
            # Log the result (truncate if too large)
            result_str = json.dumps(result, default=str, indent=2)
            print(f"Tool Result: {result_str[:500]}..." if len(result_str) > 500 else f"Tool Result: {result_str}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': error_msg,
                'results': [],
                'traceback': traceback.format_exc()
            }
    
    async def run_query(self, query: str) -> Dict[str, Any]:
        """
        Run a natural language query using an agentic approach with Claude.
        
        Args:
            query: Natural language query
            
        Returns:
            Dict containing the query results and execution details with the following structure:
            {
        """
        print("\n" + "="*80)
        print("=== START AGENTIC QUERY EXECUTION ===")
        print(f"Query: {query}")
        print(f"Database URL: {self.engine.url}")
        start_time = time.time()
        
        try:
            # Get a database connection to verify connectivity
            print("\n--- Testing database connection ---")
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                test_result = result.scalar()
                print(f"Database connection test result: {test_result}")
                if test_result != 1:
                    raise Exception("Unexpected test query result")
            
            # Get table row counts for debugging
            print("\n--- Table row counts ---")
            table_counts = {}
            for table in self.get_available_tables():
                try:
                    with self.engine.connect() as conn:
                        result = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
                        count = result.scalar()
                        table_counts[table] = count
                        print(f"  {table}: {count}")
                except Exception as e:
                    error_msg = f"Error getting row count for {table}: {str(e)}"
                    print(f"  {error_msg}")
            
            # Log database connection info
            print(f"Database URL: {self.engine.url}")
            
            # Get table row counts for debugging
            print("\n--- Table row counts ---")
            table_counts = {}
            for table in self.get_available_tables():
                try:
                    with self.engine.connect() as conn:
                        result = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
                        count = result.scalar()
                        table_counts[table] = count
                        print(f"  {table}: {count}")
                except Exception as e:
                    error_msg = f"Error getting row count for {table}: {str(e)}"
                    print(f"  {error_msg}")
            
            print("\n--- Available Tools ---")
            print(f"Available tools: {[tool.name for tool in self.tools]}")
            
            # Log the system prompt for debugging
            print("\n=== System Prompt ===")
            print(self.system_prompt)
            print("=" * len("=== System Prompt ==="))
            
            # Run the agent
            print("\n--- Running agent ---")
            print(f"Sending query to agent: {query}")
            
            try:
                # Use the agent executor to run the query
                response = await self.agent.ainvoke({"input": query, "chat_history": []})
                print("Agent invocation successful")
                print(f"Response type: {type(response).__name__}")
            except Exception as e:
                print(f"Agent invocation failed: {str(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                raise
            
            # Extract the response content
            # The AgentExecutor returns a dict with 'input', 'output', and possibly intermediate steps
            if isinstance(response, dict):
                content = response.get('output', '')
                print(f"\n--- Agent Response ---")
                print(f"Response type: {type(response).__name__}")
                print(f"Output: {content}")
                
                # Try to extract SQL and results from intermediate steps
                intermediate_steps = response.get('intermediate_steps', [])
                if intermediate_steps:
                    print(f"\n--- Processing Intermediate Steps ({len(intermediate_steps)}) ---")
                    for i, step in enumerate(intermediate_steps):
                        print(f"Step {i+1}: {step}")
                        if isinstance(step, tuple) and len(step) >= 2:
                            action = step[0]
                            output = step[1]
                            
                            # Check if this was a query_database action
                            if hasattr(action, 'tool') and action.tool == 'query_database':
                                if hasattr(action, 'tool_input'):
                                    sql = action.tool_input
                                    print(f"Found SQL from action: {sql}")
                                    
                                # Parse the JSON output to get results
                                if isinstance(output, str):
                                    try:
                                        import json
                                        results = json.loads(output)
                                        print(f"Parsed {len(results)} results from tool output")
                                    except json.JSONDecodeError:
                                        print(f"Could not parse tool output as JSON: {output[:200]}")
            else:
                content = str(response)
                print(f"\n--- Raw Agent Response ---")
                print(f"Response type: {type(response).__name__}")
                print(f"Content: {content}")
            
            # Extract SQL and results if available
            sql = sql if 'sql' in locals() else ""
            results = results if 'results' in locals() else []
            
            # Format the response
            response_data = {
                'query': query,
                'sql': sql,
                'results': results,
                'execution_time': time.time() - start_time,
                'chart_data': self.visualizer.generate_chart_data(results) if hasattr(self, 'visualizer') and results else None,
                'alerts': self.alert_manager.check_alerts(results, query) if hasattr(self, 'alert_manager') and results else []
            }
            
            print("\n--- Final Response ---")
            print(f"Returning {len(results)} results")
            print(f"Execution time: {response_data['execution_time']:.2f} seconds")
            print("\n=== END AGENTIC QUERY EXECUTION ===" + "\n" + "="*80 + "\n")
            
            return response_data
            
        except Exception as e:
            error_msg = f"Error running agentic query: {str(e)}"
            print(f"\n!!! ERROR: {error_msg}")
            import traceback
            print("\n--- Traceback ---")
            print(traceback.format_exc())
            print("-----------------\n")
            
            return {
                'query': query,
                'sql': '',
                'results': [],
                'execution_time': time.time() - start_time,
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
            
            # If we executed any tools, get the final response with explanation
            if tool_use and tool_results and tool_results[-1].get('success', False):
                print("\nGetting final response from Claude...")
                try:
                    # Get the actual results from the last successful tool execution
                    last_result = tool_results[-1]
                    
                    # If we have results from the database, include them in the explanation
                    results_for_explanation = last_result.get('results', [])
                    
                    # Generate a natural language explanation of the results
                    explanation_prompt = f"""
                    Here's a query and its results. Please provide a clear, concise explanation:
                    
                    Query: {query}
                    
                    Results: {json.dumps(results_for_explanation[:3], indent=2) if results_for_explanation else 'No results found'}
                    
                    Provide a 1-2 sentence explanation of what these results mean in the context of the query.
                    """
                    
                    final_response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.llm.invoke([
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": explanation_prompt}
                        ])
                    )
                    
                    # Extract the explanation from the final response
                    if hasattr(final_response, 'content'):
                        explanation = ' '.join([c.text for c in final_response.content if hasattr(c, 'text')])
                        response_data['explanation'] = explanation
                        print(f"Generated explanation: {explanation}")
                    
                except Exception as e:
                    print(f"Error generating explanation: {str(e)}")
                    response_data['explanation'] = "Results were processed successfully, but an error occurred while generating the explanation."
            
            return response_data
            
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"\nError in run_query: {str(e)}\n{error_trace}")
            
            return {
                'results': [],
                'error': f"Error executing query: {str(e)}",
                'traceback': error_trace
            }
    
    def _get_database_schema(self) -> Dict[str, Any]:
        """
        Get detailed database schema information.
        
        Returns:
            Dict containing the database schema with tables, columns, relationships, and sample data
        """
        # First, inspect the actual database schema
        actual_schema = {}
        for table_name in self.inspector.get_table_names():
            columns = []
            for column in self.inspector.get_columns(table_name):
                columns.append({
                    'name': column['name'],
                    'type': str(column['type']),
                    'nullable': column['nullable'],
                    'default': column.get('default')
                })
            actual_schema[table_name] = {
                'columns': columns,
                'primary_key': self.inspector.get_pk_constraint(table_name).get('constrained_columns', [])
            }
            
        print("\n=== ACTUAL DATABASE SCHEMA ===")
        print(json.dumps(actual_schema, indent=2, default=str))
        print("============================\n")
        
        # Define our expected schema for the agent
        schema = {
            'description': 'Security monitoring and alerting database',
            'tables': {
                'alerts': {
                    'description': 'Security alerts generated by monitoring systems',
                    'columns': [
                        {'name': 'alert_id', 'type': 'UUID', 'description': 'Unique identifier for the alert', 'primary_key': True},
                        {'name': 'title', 'type': 'VARCHAR(255)', 'description': 'Short title of the alert'},
                        {'name': 'description', 'type': 'TEXT', 'description': 'Detailed description of the alert'},
                        {'name': 'severity', 'type': 'VARCHAR(20)', 'description': 'Severity level (critical, high, medium, low, info)'},
                        {'name': 'status', 'type': 'VARCHAR(20)', 'description': 'Current status (open, in_progress, resolved, false_positive)'},
                        {'name': 'created_at', 'type': 'TIMESTAMP', 'description': 'When the alert was created'},
                        {'name': 'updated_at', 'type': 'TIMESTAMP', 'description': 'When the alert was last updated'},
                        {'name': 'source', 'type': 'VARCHAR(100)', 'description': 'Source system that generated the alert'},
                        {'name': 'customer_id', 'type': 'UUID', 'description': 'Reference to the affected customer'},
                        {'name': 'rule_id', 'type': 'UUID', 'description': 'Reference to the rule that triggered this alert'}
                    ]
                },
                'customers': {
                    'description': 'Customer organizations being monitored',
                    'columns': [
                        {'name': 'customer_id', 'type': 'UUID', 'description': 'Unique identifier for the customer', 'primary_key': True},
                        {'name': 'name', 'type': 'VARCHAR(255)', 'description': 'Customer organization name'},
                        {'name': 'subscription_tier', 'type': 'VARCHAR(50)', 'description': 'Subscription level (enterprise, business, basic)'},
                        {'name': 'is_active', 'type': 'BOOLEAN', 'description': 'Whether the customer account is active'},
                        {'name': 'created_at', 'type': 'TIMESTAMP', 'description': 'When the customer was onboarded'}
                    ]
                },
                'rules': {
                    'description': 'Alert rules that define when alerts should be generated',
                    'columns': [
                        {'name': 'rule_id', 'type': 'UUID', 'description': 'Unique identifier for the rule', 'primary_key': True},
                        {'name': 'name', 'type': 'VARCHAR(255)', 'description': 'Descriptive name of the rule'},
                        {'name': 'description', 'type': 'TEXT', 'description': 'Detailed description of what the rule detects'},
                        {'name': 'severity', 'type': 'VARCHAR(20)', 'description': 'Default severity for alerts from this rule'},
                        {'name': 'query', 'type': 'TEXT', 'description': 'SQL query or detection logic'},
                        {'name': 'is_active', 'type': 'BOOLEAN', 'description': 'Whether the rule is currently active'}
                    ]
                },
                'incidents': {
                    'description': 'Security incidents composed of one or more related alerts',
                    'columns': [
                        {'name': 'incident_id', 'type': 'UUID', 'description': 'Unique identifier for the incident', 'primary_key': True},
                        {'name': 'title', 'type': 'VARCHAR(255)', 'description': 'Short title of the incident'},
                        {'name': 'description', 'type': 'TEXT', 'description': 'Detailed description of the incident'},
                        {'name': 'status', 'type': 'VARCHAR(20)', 'description': 'Current status (investigating, contained, resolved, closed)'},
                        {'name': 'severity', 'type': 'VARCHAR(20)', 'description': 'Highest severity of related alerts'},
                        {'name': 'created_at', 'type': 'TIMESTAMP', 'description': 'When the incident was created'},
                        {'name': 'updated_at', 'type': 'TIMESTAMP', 'description': 'When the incident was last updated'},
                        {'name': 'customer_id', 'type': 'UUID', 'description': 'Reference to the affected customer'}
                    ]
                },
                'incident_alerts': {
                    'description': 'Many-to-many relationship between incidents and alerts',
                    'columns': [
                        {'name': 'incident_id', 'type': 'UUID', 'description': 'Reference to the incident', 'primary_key': True},
                        {'name': 'alert_id', 'type': 'UUID', 'description': 'Reference to the alert', 'primary_key': True},
                        {'name': 'created_at', 'type': 'TIMESTAMP', 'description': 'When the alert was added to the incident'}
                    ]
                }
            },
            'relationships': [
                {'from': 'alerts.customer_id', 'to': 'customers.customer_id', 'type': 'many-to-one'},
                {'from': 'alerts.rule_id', 'to': 'rules.rule_id', 'type': 'many-to-one'},
                {'from': 'incidents.customer_id', 'to': 'customers.customer_id', 'type': 'many-to-one'},
                {'from': 'incident_alerts.incident_id', 'to': 'incidents.incident_id', 'type': 'many-to-one'},
                {'from': 'incident_alerts.alert_id', 'to': 'alerts.alert_id', 'type': 'many-to-one'}
            ]
        }
        
        # Add sample data descriptions to help with query generation
        schema['sample_data'] = {
            'alerts': 'Contains security alerts with details like severity, status, and timestamps',
            'customers': 'Contains customer organizations with subscription tiers',
            'rules': 'Contains detection rules that generate alerts',
            'incidents': 'Groups related alerts into security incidents',
            'incident_alerts': 'Links alerts to incidents in a many-to-many relationship'
        }
        
        return schema
    
    async def _generate_sql(self, query: str, schema_info: Dict[str, Any]) -> str:
        """Generate SQL query from natural language using the LLM"""
        # Create a prompt for the LLM
        prompt = f"""You are a security analyst working with a PostgreSQL database. 
        Your task is to convert the following natural language query into a SQL query.
        
        Database Schema:
        {json.dumps(schema_info, indent=2)}
        
        User Query: {query}
        
        Provide only the SQL query, without any additional text or explanation.
        Make sure to use proper SQL syntax and include all necessary conditions.
        """
        
        try:
            # Call the LLM to generate SQL
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.invoke(prompt)
            )
            
            # Extract the SQL query from the response
            sql = response.content.strip()
            
            # Clean up the SQL (remove markdown code blocks if present)
            if sql.startswith('```sql'):
                sql = sql[6:]
            if sql.endswith('```'):
                sql = sql[:-3]
                
            return sql.strip()
            
        except Exception as e:
            print(f"Error generating SQL: {str(e)}")
            return ""
    
    async def _execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return the results.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            List of dictionaries representing the query results
            
        Raises:
            Exception: If there's an error executing the query
        """
        print("\n" + "="*80)
        print("=== _execute_sql START ===")
        print(f"Executing SQL query:\n{sql}")
        
        # Log the current working directory and environment
        import os
        print(f"Current working directory: {os.getcwd()}")
        print(f"Environment variables: {os.environ.get('POSTGRES_DB')}")
        
        # Log the database URL (with password masked)
        db_url = str(self.engine.url)
        if '@' in db_url:
            # Mask the password in the URL for logging
            protocol_part = db_url.split('//')[0] + '//'
            auth_part = db_url.split('@')[0].split('//')[1]
            if ':' in auth_part:
                user = auth_part.split(':')[0]
                db_url = f"{protocol_part}{user}:***@{'@'.join(db_url.split('@')[1:])}"
        print(f"Database URL: {db_url}")
        
        # Check if the query is empty
        if not sql or not sql.strip():
            error_msg = "Error: Empty SQL query"
            print(error_msg)
            raise ValueError(error_msg)
            
        try:
            # First, try with the existing sync engine for simplicity
            print("\n--- Attempting to execute with sync engine ---")
            from sqlalchemy import text as sql_text
            
            with self.engine.connect() as conn:
                try:
                    print("\n--- Executing query with sync connection ---")
                    
                    # Log the actual SQL being sent to the database
                    compiled = sql_text(sql).compile(
                        self.engine,
                        compile_kwargs={"literal_binds": True}
                    )
                    print(f"\nCompiled SQL:\n{str(compiled).strip()}")
                    
                    # Execute the query
                    print("\n--- Executing query ---")
                    result = conn.execute(sql_text(sql))
                    print("Query executed successfully with sync connection")
                    
                    # Fetch all rows
                    print("\n--- Fetching results ---")
                    rows = result.fetchall()
                    print(f"Fetched {len(rows)} rows")
                    
                    # Log the first few rows for debugging
                    if rows:
                        print("\n--- First few rows ---")
                        for i, row in enumerate(rows[:3]):
                            print(f"Row {i+1}: {row}")
                    
                    # Convert to list of dicts
                    if not rows:
                        print("\nNo rows returned from query")
                        return []
                    
                    # Get column names
                    columns = result.keys()
                    print(f"\n--- Result Columns ---\n{list(columns)}")
                    
                    print("\n--- Converting rows to dictionaries ---")
                    result_list = []
                    for row in rows:
                        row_dict = {}
                        for i, col in enumerate(columns):
                            value = row[i]
                            # Convert non-serializable types
                            if hasattr(value, 'isoformat'):  # Handle dates and times
                                row_dict[col] = value.isoformat()
                            else:
                                row_dict[col] = str(value) if value is not None else None
                        result_list.append(row_dict)
                    
                    print(f"\nConverted {len(result_list)} rows to dictionaries")
                    if result_list:
                        print("\n--- First row sample ---")
                        print(json.dumps(result_list[0], default=str, indent=2))
                    
                    print("\n=== _execute_sql COMPLETE ===\n" + "="*80 + "\n")
                    return result_list
                    
                except Exception as e:
                    error_msg = f"Error in sync query execution: {str(e)}"
                    print(f"\n!!! ERROR: {error_msg}")
                    import traceback
                    print("\n--- Traceback ---")
                    print(traceback.format_exc())
                    print("-----------------\n")
                    
                    # Fall back to async if sync fails
                    print("Falling back to async execution...")
                    return await self._execute_sql_async(sql)
        
        except Exception as e:
            error_msg = f"Error in _execute_sql: {str(e)}"
            print(f"\n!!! CRITICAL ERROR: {error_msg}")
            import traceback
            print("\n--- Full Traceback ---")
            print(traceback.format_exc())
            print("----------------------\n")
            raise Exception(error_msg)
        finally:
            print("=== _execute_sql END ===\n")
    
    def _execute_sql_async(self, sql: str):
        """
        Execute a SQL query and return the results.
        This is a synchronous implementation for consistency.
        """
        print("\n--- Using SQL execution ---")
        print(f"Executing SQL query: {sql[:200]}...")
        
        try:
            with self.engine.connect() as conn:
                try:
                    # Execute the query
                    result = conn.execute(text(sql))
                    conn.commit()  # Explicitly commit the transaction
                    print("Query executed successfully")
                    
                    # Fetch all rows
                    rows = result.fetchall()
                    print(f"Fetched {len(rows)} rows")
                    
                    # Convert to list of dicts
                    if not rows:
                        print("No rows returned from query")
                        return []
                    
                    columns = list(rows[0]._mapping.keys())
                    print(f"Columns in result: {columns}")
                    
                    result_list = []
                    for row in rows:
                        row_dict = {}
                        for col in columns:
                            value = row._mapping[col]
                            # Convert non-serializable types
                            if hasattr(value, 'isoformat'):  # Handle dates and times
                                row_dict[col] = value.isoformat()
                            else:
                                row_dict[col] = value
                        result_list.append(row_dict)
                    
                    print(f"Converted {len(result_list)} rows to dicts")
                    if result_list:
                        print(f"First row sample: {json.dumps(result_list[0], default=str, indent=2)}")
                    
                    return result_list
                    
                except Exception as e:
                    error_msg = f"Error in query execution: {str(e)}"
                    print(error_msg)
                    import traceback
                    print(traceback.format_exc())
                    raise Exception(error_msg)
        
        except Exception as e:
            error_msg = f"Error in _execute_sql: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            raise
        
    async def _execute_sql_tool(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle the query_database tool execution.
        
        Args:
            tool_input: Dictionary containing the tool input with 'query' key
            
        Returns:
            Dictionary with query results or error information
        """
        query = tool_input.get('query', '')
        print("\n" + "="*80)
        print("=== _execute_sql_tool START ===")
        print(f"Executing SQL query:\n{query}")
        
        try:
            # Log the database URL (with password masked)
            db_url = str(self.engine.url)
            if '@' in db_url:
                # Mask the password in the URL for logging
                protocol_part = db_url.split('//')[0] + '//'
                auth_part = db_url.split('@')[0].split('//')[1]
                if ':' in auth_part:
                    user = auth_part.split(':')[0]
                    db_url = f"{protocol_part}{user}:***@{'@'.join(db_url.split('@')[1:])}"
            print(f"Database URL: {db_url}")
            
            # Execute the query
            print("\n--- Executing SQL query ---")
            results = await self._execute_sql(query)
            
            # Log the results
            print(f"\n--- Query Results ---")
            print(f"Query returned {len(results)} rows")
            
            if results:
                # Print column headers
                if results:
                    columns = list(results[0].keys())
                    print("Columns:", ", ".join(columns))
                
                # Print first 3 rows
                print("\nFirst 3 rows:")
                for i, row in enumerate(results[:3]):
                    row_str = ", ".join(f"{k}: {v}" for k, v in row.items())
                    print(f"  Row {i+1}: {row_str}")
            else:
                print("No results returned from query")
                
                # For debugging, try to explain why no results were returned
                print("\n--- Debugging empty result set ---")
                print("Checking if tables exist and have data...")
                
                # Check if the query contains specific conditions that might cause no results
                if "WHERE" in query.upper():
                    print("\nQuery contains WHERE clause. Checking conditions...")
                    # Simple check for common conditions that might not match any data
                    if "severity = 'high'" in query.lower():
                        print("  - Query filters for severity = 'high' (note: case-sensitive comparison)")
                        print("    Try using UPPERCASE 'HIGH' instead of 'high'")
                    
                    if "customers" in query.lower() and "subscription_tier = 'enterprise'" in query.lower():
                        print("  - Query filters for subscription_tier = 'enterprise'")
                        print("    Checking if any customers have this subscription tier...")
                        try:
                            from sqlalchemy import text
                            with self.engine.connect() as conn:
                                result = conn.execute(text("SELECT COUNT(*) FROM customers WHERE subscription_tier = 'enterprise'"))
                                count = result.scalar()
                                print(f"    Found {count} customers with subscription_tier = 'enterprise'")
                        except Exception as e:
                            print(f"    Error checking customers: {str(e)}")
                
                print("\n--- End Debugging ---")
            
            print("\n=== _execute_sql_tool COMPLETE ===" + "\n" + "="*80 + "\n")
            
            return {
                'success': True,
                'results': results,
                'row_count': len(results)
            }
            
        except Exception as e:
            error_msg = f"Error executing SQL: {str(e)}"
            print(f"\n!!! ERROR: {error_msg}")
            import traceback
            print("\n--- Traceback ---")
            print(traceback.format_exc())
            print("-----------------\n")
            
            return {
                'success': False,
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
    
    async def _get_schema_tool(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle the get_schema tool execution.
        
        Returns:
            Dict containing the database schema information
        """
        try:
            print("\n=== Getting Database Schema ===")
            schema = self._get_database_schema()
            return {
                'success': True,
                'schema': schema
            }
        except Exception as e:
            error_msg = f"Error getting database schema: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            return {
                'success': False,
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
    
    def get_available_tables(self) -> List[str]:
        """Get list of available tables in the database"""
        return self.inspector.get_table_names()
    
    async def inspect_database(self) -> Dict[str, Any]:
        """Inspect database schema and sample data for debugging"""
        from sqlalchemy import text, inspect
        
        print("\n=== Database Inspection ===")
        result = {'tables': {}}
        
        # Get all tables
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        print(f"Tables in database: {tables}")
        
        # For each table, get columns and sample data
        for table in tables:
            print(f"\n--- Table: {table} ---")
            table_info = {'columns': [], 'sample_data': []}
            
            try:
                # Get columns with metadata
                columns = inspector.get_columns(table)
                column_names = [col['name'] for col in columns]
                table_info['columns'] = [{
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col.get('nullable', True),
                    'default': str(col.get('default', 'NULL'))
                } for col in columns]
                
                print(f"Columns: {column_names}")
                
                # Get primary keys
                pk_constraint = inspector.get_pk_constraint(table)
                pk_columns = pk_constraint.get('constrained_columns', [])
                if pk_columns:
                    print(f"Primary keys: {pk_columns}")
                    table_info['primary_keys'] = pk_columns
                
                # Get foreign keys
                fks = inspector.get_foreign_keys(table)
                if fks:
                    print("Foreign keys:")
                    for fk in fks:
                        print(f"  {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")
                
                # Get sample data (first 2 rows)
                with self.engine.connect() as conn:
                    # Use a simple query to get row count
                    count_result = conn.execute(text(f"SELECT COUNT(*) as count FROM \"{table}\""))
                    row_count = count_result.scalar()
                    print(f"Total rows: {row_count}")
                    
                    if row_count > 0:
                        # Get sample data
                        sample_result = conn.execute(text(f"SELECT * FROM \"{table}\" LIMIT 2"))
                        rows = sample_result.mappings().all()
                        print(f"Sample data ({len(rows)} rows):")
                        
                        for i, row in enumerate(rows):
                            row_data = {}
                            for key, value in row.items():
                                # Handle different data types for JSON serialization
                                if hasattr(value, 'isoformat'):  # Handle dates and times
                                    row_data[key] = value.isoformat()
                                else:
                                    row_data[key] = str(value) if value is not None else None
                            
                            print(f"  Row {i+1}: {row_data}")
                            table_info['sample_data'].append(row_data)
                
            except Exception as e:
                error_msg = f"Error inspecting table {table}: {str(e)}"
                print(error_msg)
                table_info['error'] = str(e)
                import traceback
                table_info['traceback'] = traceback.format_exc()
            
            result['tables'][table] = table_info
                
        print("\n=== End Database Inspection ===\n")
        return result
    
    def get_table_info(self, table_names: List[str] = None) -> str:
        """
        Get information about tables in the database
        
        Args:
            table_names: List of table names to get info for. If None, returns all tables.
            
        Returns:
            String with table information
        """
        if table_names is None:
            table_names = self.get_available_tables()
            
        info = []
        for table_name in table_names:
            if table_name not in self.get_available_tables():
                continue
                
            # Get columns
            columns = self.inspector.get_columns(table_name)
            pks = self.inspector.get_pk_constraint(table_name)
            fks = self.inspector.get_foreign_keys(table_name)
            
            # Format column info
            col_info = []
            for col in columns:
                col_desc = f"- {col['name']} ({col['type']})"
                if col.get('primary_key', False) or col['name'] in pks.get('constrained_columns', []):
                    col_desc += " [PK]"
                if col.get('foreign_keys'):
                    col_desc += " [FK]"
                if col.get('nullable') is False:
                    col_desc += " NOT NULL"
                if col.get('default') is not None:
                    col_desc += f" DEFAULT {col['default']}"
                col_info.append(col_desc)
            
            # Format foreign key info
            fk_info = []
            for fk in fks:
                fk_info.append(
                    f"- {', '.join(fk['constrained_columns'])} -> "
                    f"{fk['referred_table']}({', '.join(fk['referred_columns'])})"
                )
            
            # Combine table info
            table_info = [
                f"Table: {table_name}",
                "Columns:",
                *[f"  {line}" for line in col_info]
            ]
            
            if fk_info:
                table_info.extend(["Foreign Keys:", *[f"  {line}" for line in fk_info]])
                
            info.append("\n".join(table_info))
            
        return "\n\n".join(info)

