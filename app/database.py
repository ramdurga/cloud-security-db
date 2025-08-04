from typing import Dict, List, Any, Generator, Optional
import logging
import os
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, MetaData, Table, Column, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database connection settings with environment variable fallbacks
DB_CONFIG = {
    'user': os.getenv('POSTGRES_USER', 'security_admin'),
    'password': os.getenv('POSTGRES_PASSWORD', 'securepassword123'),
    'dbname': os.getenv('POSTGRES_DB', 'cloud_security'),
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'pool_size': int(os.getenv('DB_POOL_SIZE', '5')),
    'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '10')),
    'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
    'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', '3600')),
}

# Create database URL
DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"

# Configure engine with connection pooling
engine: Engine = create_engine(
    DATABASE_URL,
    pool_size=DB_CONFIG['pool_size'],
    max_overflow=DB_CONFIG['max_overflow'],
    pool_timeout=DB_CONFIG['pool_timeout'],
    pool_recycle=DB_CONFIG['pool_recycle'],
    pool_pre_ping=True  # Enable connection health checks
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency function that yields a database session.
    
    Yields:
        Session: A SQLAlchemy database session
        
    Example:
        ```python
        def get_data():
            with get_db() as db:
                return db.query(MyModel).all()
        ```
    """
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for database sessions"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error in context manager: {str(e)}")
        raise
    finally:
        db.close()

def get_table_names() -> List[str]:
    """
    Get all table names from the database
    
    Returns:
        List[str]: List of table names
    """
    try:
        inspector = inspect(engine)
        return inspector.get_table_names()
    except Exception as e:
        logger.error(f"Error getting table names: {str(e)}")
        raise

def get_table_schema(table_name: str) -> List[Dict[str, Any]]:
    """
    Get schema information for a specific table
    
    Args:
        table_name: Name of the table to inspect
        
    Returns:
        List[Dict]: List of column information dictionaries
    """
    try:
        inspector = inspect(engine)
        columns = []
        
        # Get column information
        for column in inspector.get_columns(table_name):
            columns.append({
                'name': column['name'],
                'type': str(column['type']),
                'nullable': column['nullable'],
                'default': str(column.get('default', '')),
                'primary_key': column.get('primary_key', False),
                'autoincrement': column.get('autoincrement', False),
                'comment': column.get('comment', '')
            })
            
        # Get foreign key information
        foreign_keys = []
        for fk in inspector.get_foreign_keys(table_name):
            foreign_keys.append({
                'name': fk.get('name', ''),
                'constrained_columns': fk['constrained_columns'],
                'referred_table': fk['referred_table'],
                'referred_columns': fk['referred_columns'],
                'options': fk.get('options', {})
            })
            
        # Get index information
        indexes = []
        for idx in inspector.get_indexes(table_name):
            indexes.append({
                'name': idx['name'],
                'column_names': idx['column_names'],
                'unique': idx['unique'],
                'type': idx.get('type', '')
            })
            
        return {
            'columns': columns,
            'foreign_keys': foreign_keys,
            'indexes': indexes
        }
        
    except Exception as e:
        logger.error(f"Error getting schema for table {table_name}: {str(e)}")
        raise

def get_table_relationships() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get relationships between tables using foreign keys
    
    Returns:
        Dict: Dictionary with table relationships
    """
    try:
        inspector = inspect(engine)
        relationships = {}
        
        for table_name in inspector.get_table_names():
            fks = inspector.get_foreign_keys(table_name)
            if fks:
                relationships[table_name] = []
                for fk in fks:
                    relationships[table_name].append({
                        'source_table': table_name,
                        'source_columns': fk['constrained_columns'],
                        'target_table': fk['referred_table'],
                        'target_columns': fk['referred_columns'],
                        'name': fk.get('name', 'fk_' + '_'.join(fk['constrained_columns']))
                    })
        
        return relationships
    except Exception as e:
        logger.error(f"Error getting table relationships: {str(e)}")
        raise

def get_database_schema() -> Dict[str, Any]:
    """
    Get complete database schema including tables, columns, and relationships
    
    Returns:
        Dict: Complete database schema
    """
    try:
        inspector = inspect(engine)
        schema = {
            'tables': {},
            'relationships': get_table_relationships()
        }
        
        for table_name in inspector.get_table_names():
            schema['tables'][table_name] = get_table_schema(table_name)
            
        return schema
        
    except Exception as e:
        logger.error(f"Error getting database schema: {str(e)}")
        raise

def get_db_schema():
    """
    Alias for get_database_schema to maintain backward compatibility.
    
    Returns:
        Dict: Complete database schema
    """
    return get_database_schema()

def execute_query(query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Execute a raw SQL query and return results as dictionaries
    
    Args:
        query: SQL query string
        params: Optional query parameters
        
    Returns:
        List[Dict]: List of result rows as dictionaries
    """
    with get_db() as session:
        try:
            result = session.execute(text(query), params or {})
            return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}\nQuery: {query}")
            raise
