from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker
import os

router = APIRouter()

def get_db_engine():
    db_uri = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    return create_engine(db_uri)

@router.get("/debug/tables")
async def debug_tables():
    """Debug endpoint to list all tables and their row counts"""
    engine = get_db_engine()
    inspector = engine.dialect.inspector(engine)
    
    tables = inspector.get_table_names()
    result = {}
    
    with engine.connect() as conn:
        for table in tables:
            try:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                result[table] = {
                    "row_count": count,
                    "columns": [col["name"] for col in inspector.get_columns(table)]
                }
            except Exception as e:
                result[table] = {"error": str(e)}
    
    return result

@router.get("/debug/sample/{table_name}")
async def debug_table_sample(table_name: str, limit: int = 5):
    """Debug endpoint to get sample data from a table"""
    engine = get_db_engine()
    
    with engine.connect() as conn:
        try:
            result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT {limit}"))
            columns = result.keys()
            rows = [dict(zip(columns, row)) for row in result]
            return {"table": table_name, "data": rows}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
