import os
import json
from typing import Dict, Any
import httpx
from pydantic import BaseModel, Field

class LLMQueryService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Please set ANTHROPIC_API_KEY environment variable.")
        
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
    async def _make_request(self, messages: list, max_tokens: int = 4000) -> str:
        """Make a request to the Anthropic API"""
        # Prepare the messages in the correct format
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
                continue
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        
        # Add system prompt if present
        if 'system_prompt' in locals():
            payload["system"] = system_prompt
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the response content
            if "content" in response_data and len(response_data["content"]) > 0:
                if "text" in response_data["content"][0]:
                    return response_data["content"][0]["text"]
                elif "text" in response_data["content"][0]:
                    return response_data["content"][0]["text"]
            
            # If we can't find the expected response format, return the raw response
            return str(response_data)
    
    async def generate_sql(self, user_query: str, schema: Dict[str, Any]) -> str:
        """Generate SQL query from natural language"""
        try:
            schema_str = json.dumps(schema, indent=2)
            
            system_prompt = """You are a PostgreSQL expert. Your task is to convert the user's natural language query into a valid SQL query.
            
            Instructions:
            1. Analyze the database schema to understand table relationships
            2. Generate a valid PostgreSQL query that answers the user's question
            3. Only return the SQL query, no additional text or explanation
            4. If the query requires a table or column that doesn't exist, return an error message starting with 'Error:'
            5. Use proper JOINs and WHERE clauses as needed
            6. Format the SQL for readability
            
            IMPORTANT NOTES:
            - For subscription_tier values, use these exact values (case-sensitive):
              * 'ENTERPRISE' (not 'enterprise' or 'Enterprise')
              * 'BUSINESS' (not 'business' or 'Business')
              * 'STANDARD' (not 'standard' or 'Standard')
            
            VERY IMPORTANT: Your response MUST be a valid SQL query wrapped in a code block like this:
            ```sql
            SELECT * FROM table_name WHERE condition;
            ```
            
            Do NOT include any other text in your response.
            """
            
            user_message = f"""Database Schema:
            {schema_str}
            
            User Query: {user_query}
            
            SQL Query:"""
            
            # Format the messages for the API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Make the async request
            response = await self._make_request(messages)
            
            # Extract SQL from code block if present
            if "```sql" in response:
                response = response.split("```sql")[1].split("```")[0].strip()
            
            return response.strip()
            
        except Exception as e:
            return f"Error: Failed to generate SQL - {str(e)}"
