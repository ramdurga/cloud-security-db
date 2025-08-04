# Cloud Security Database Query Tool

This application provides a natural language interface for querying a cloud security database. It uses Anthropic's Claude to convert natural language queries into SQL, executes them against a PostgreSQL database, and returns structured results.

## ✨ Features

- Natural language to SQL conversion
- Secure database querying
- Real-time schema inspection
- Support for complex queries with joins and aggregations
- RESTful API interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL database
- Anthropic API key

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd cloud-security-db
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Update the `.env` file with your database credentials and Anthropic API key.

### Running the Application

Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Endpoints

- `GET /health`: Health check endpoint
  ```bash
  curl http://localhost:8000/health
  ```

- `GET /schema`: Get the database schema
  ```bash
  curl http://localhost:8000/schema
  ```

- `POST /query`: Submit a natural language query
  ```bash
  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "your query here"}'
  ```

## 💡 Example Queries

### Basic Queries

1. **Get all customers**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "get all users"}'
   ```

2. **Find active enterprise customers**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Show me all active enterprise customers"}'
   ```

### Advanced Queries

3. **Get active enterprise customers and their assets**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Show me all active enterprise customers and their assets, including the asset type and region, sorted by customer name"}'
   ```

4. **Get all enterprise customers (including inactive) and their assets**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Show me all ENTERPRISE customers and their assets, including the asset type and region, sorted by customer name"}'
   ```

5. **Find high severity alerts**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Find all high severity alerts for assets that belong to inactive customers"}'
   ```

4. **Asset distribution by type and region**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Show me a count of assets by type and region, ordered by count"}'
   ```

5. **Find unencrypted assets with alerts**
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Find all assets that are not encrypted and have high or critical alerts"}'
   ```

## 🛠️ Development

### Database Schema

The application works with the following tables:
- `customers`: Customer information and subscription details
- `assets`: Cloud assets belonging to customers
- `alerts`: Security alerts related to assets

### Environment Variables

- `POSTGRES_*`: Database connection settings
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `ANTHROPIC_MODEL`: (Optional) Defaults to `claude-3-haiku-20240307`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
   curl http://localhost:8000/schema
   ```

2. Submit a natural language query:
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "Show me all tables in the database"}'
   ```

## Development

- The database schema is automatically loaded from `db-migrations/` directory
- The application uses SQLAlchemy for database operations
- LLM integration is handled through LangChain

## License

MIT
