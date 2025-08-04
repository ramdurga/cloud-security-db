# Cloud Security Database Query Tool

A PostgreSQL-based security monitoring database with natural language query capabilities powered by AI (Claude via LangChain).

## ✨ Features

- **AI-Powered Natural Language Queries**: Use Claude to convert natural language to SQL
- **Agent-Based Query Processing**: Intelligent query understanding and execution
- **Security Monitoring**: Track alerts, incidents, and customer security data
- **Visualization Support**: Generate charts and visual reports
- **RESTful API**: FastAPI-based endpoints for easy integration
- **Real-time Schema Inspection**: Automatic database schema discovery

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL database (or Docker)
- Anthropic API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd cloud-security-db
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```bash
   # Database Configuration
   POSTGRES_USER=security_admin
   POSTGRES_PASSWORD=secure_password_123
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=cloud_security
   
   # Anthropic API Configuration
   ANTHROPIC_API_KEY=your_api_key_here
   ANTHROPIC_MODEL=claude-3-opus-20240229
   ```

### Database Setup

1. **Start PostgreSQL with Docker:**
   ```bash
   docker-compose up -d
   ```

2. **Initialize the database schema:**
   ```bash
   # The database will be automatically initialized with migrations
   # Check if data exists:
   python check_data.py
   
   # Or manually insert test data:
   python insert_test_data.py
   ```

### Running the Application

1. **Start the FastAPI server:**
   ```bash
   # From the project root directory
   python app/main.py
   
   # Or using uvicorn directly
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Verify the server is running:**
   ```bash
   curl http://localhost:8000/health
   ```

## 📚 API Documentation

### Interactive Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Core Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Get Database Schema
```bash
curl http://localhost:8000/schema
```

#### Natural Language Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find all high severity alerts for enterprise customers"
  }'
```

#### Agent-Based Query (Intelligent Processing)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find all high severity alerts for enterprise customers",
    "use_agent": true
  }'
```

#### Direct SQL Query
```bash
curl -X POST http://localhost:8000/sql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "SELECT * FROM alerts WHERE severity = '\''HIGH'\'' LIMIT 10"
  }'
```

## 💡 Example Queries

### Security Alert Queries

1. **Find high severity alerts for enterprise customers:**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Find all high severity alerts for enterprise customers",
       "use_agent": true
     }'
   ```

2. **Get recent critical alerts:**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Show me critical alerts from the last 7 days"
     }'
   ```

3. **Count alerts by severity:**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Count alerts grouped by severity level"
     }'
   ```

### Customer Queries

4. **List enterprise customers:**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Show me all active enterprise customers"
     }'
   ```

5. **Customer alert summary:**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Give me a summary of alerts per customer with their subscription tier"
     }'
   ```

### Advanced Queries with Visualization

6. **Generate alert trend chart:**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Show alert trends over time",
       "generate_chart": true,
       "chart_type": "line",
       "chart_x": "created_at",
       "chart_y": "count"
     }'
   ```

## 🛠️ Development

### Project Structure
```
cloud-security-db/
├── app/
│   ├── main.py              # FastAPI application
│   ├── agent_service.py     # LangChain agent implementation
│   ├── database.py          # Database connection and models
│   ├── llm_service.py       # LLM query service
│   ├── visualization.py     # Chart generation
│   ├── alerts.py            # Alert management
│   └── reporting.py         # Report generation
├── db-migrations/           # SQL migration files
├── docker-compose.yml       # PostgreSQL container setup
├── requirements.txt         # Python dependencies
└── .env                    # Environment variables
```

### Database Schema

- **alerts**: Security alerts with severity, status, and timestamps
- **customers**: Customer organizations with subscription tiers
- **rules**: Detection rules that generate alerts
- **incidents**: Groups of related alerts
- **incident_alerts**: Many-to-many relationship between incidents and alerts

### Troubleshooting

1. **Database connection issues:**
   ```bash
   # Check if PostgreSQL is running
   docker ps
   
   # Check database logs
   docker logs cloud_security_db
   
   # Test database connection
   python check_data.py
   ```

2. **Agent not returning results:**
   - Ensure ANTHROPIC_API_KEY is set correctly
   - Check that severity values are uppercase (HIGH, CRITICAL, etc.)
   - Verify subscription tiers are lowercase (enterprise, business, etc.)

3. **Import errors when running directly:**
   ```bash
   # Run from the app directory
   cd app && python main.py
   
   # Or use module execution
   python -m app.main
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [Claude](https://www.anthropic.com/claude) via [LangChain](https://www.langchain.com/)
- Database: [PostgreSQL](https://www.postgresql.org/)