from app.database import get_db, get_database_schema
from sqlalchemy import text

def check_data():
    db = next(get_db())
    
    # Check customers
    print("\n=== Customers ===")
    result = db.execute(text("SELECT * FROM customers")).fetchall()
    for row in result:
        print(dict(row._mapping))
    
    # Check assets
    print("\n=== Assets ===")
    result = db.execute(text("SELECT * FROM assets")).fetchall()
    for row in result:
        print(dict(row._mapping))
    
    # Check the specific query
    print("\n=== Enterprise Customers with Assets ===")
    query = """
    SELECT c.customer_id, c.name, c.subscription_tier, 
           a.asset_id, a.asset_name, a.asset_type, a.region
    FROM customers c
    LEFT JOIN assets a ON c.customer_id = a.customer_id
    WHERE c.subscription_tier = 'ENTERPRISE'
    ORDER BY c.name, a.asset_name
    """
    result = db.execute(text(query)).fetchall()
    for row in result:
        print(dict(row._mapping))

if __name__ == "__main__":
    check_data()
