#!/usr/bin/env python3
"""
Script to insert test data into the cloud security database.
"""
import os
import random
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import json

# Load environment variables
load_dotenv()

def get_db_connection():
    """Create and return a database connection."""
    db_uri = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    return create_engine(db_uri)

def insert_test_data():
    """Insert test data into the database."""
    engine = get_db_connection()
    
    # Test customer data - simplified
    customers = [
        {"name": "Acme Corp", "email": "security@acme.com", "subscription_tier": "enterprise", "is_active": True},
        {"name": "Initech", "email": "security@initech.com", "subscription_tier": "business", "is_active": True}
    ]
    
    # Test asset data - simplified
    asset_types = ["ec2", "s3", "rds"]
    cloud_providers = ["aws"]
    regions = ["us-east-1"]
    
    # Test alert data - using only valid values from database constraints
    alert_types = ["unauthorized_access", "misconfiguration"]
    severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]  # Must match database CHECK constraint
    statuses = ["OPEN", "IN_PROGRESS", "RESOLVED"]  # Must match database CHECK constraint
    
    with engine.connect() as conn:
        try:
            # Clear existing data
            conn.execute(text("TRUNCATE TABLE alerts, assets, customers CASCADE"))
            conn.commit()
            
            # Insert customers
            customer_ids = []
            for customer in customers:
                result = conn.execute(
                    text("""
                    INSERT INTO customers (name, email, subscription_tier, is_active, created_at)
                    VALUES (:name, :email, :subscription_tier, :is_active, NOW())
                    RETURNING customer_id
                    """),
                    customer
                )
                customer_ids.append(result.scalar())
            conn.commit()
            
            # Insert assets
            asset_ids = []
            for customer_id in customer_ids:
                # Each customer gets 2 assets
                for i in range(2):
                    asset = {
                        "customer_id": customer_id,
                        "asset_name": f"asset-{customer_id}-{i+1}",
                        "asset_type": random.choice(asset_types),
                        "cloud_provider": random.choice(cloud_providers),
                        "region": random.choice(regions),
                        "is_encrypted": True,
                        "last_scan_time": datetime.now(timezone.utc) - timedelta(days=random.randint(1, 30))
                    }
                    result = conn.execute(
                        text("""
                        INSERT INTO assets 
                        (customer_id, asset_name, asset_type, cloud_provider, region, is_encrypted, last_scan_time, created_at)
                        VALUES 
                        (:customer_id, :asset_name, :asset_type, :cloud_provider, :region, :is_encrypted, :last_scan_time, NOW())
                        RETURNING asset_id
                        """),
                        asset
                    )
                    asset_ids.append(result.scalar())
            conn.commit()
            
            # Insert alerts - simplified with guaranteed valid values
            for asset_id in asset_ids:
                # Each asset gets 2 alerts
                for i in range(2):
                    alert_time = datetime.now(timezone.utc) - timedelta(days=random.randint(0, 30))
                    
                    # Simple round-robin through statuses to ensure we get all types
                    status = statuses[i % len(statuses)]
                    severity = severities[i % len(severities)]
                    
                    resolved_time = None
                    if status == "RESOLVED":
                        resolved_time = alert_time + timedelta(hours=1)
                    
                    alert = {
                        "asset_id": asset_id,
                        "severity": severity,
                        "alert_type": alert_types[i % len(alert_types)],
                        "description": f"Test alert {i+1} for asset {asset_id}",
                        "status": status,
                        "created_at": alert_time,
                        "resolved_at": resolved_time,
                        "metadata": json.dumps({
                            "source": "test_script",
                            "confidence": 0.8,
                            "tags": ["test"]
                        })
                    }
                    
                    # Get the customer_id for this asset
                    result = conn.execute(
                        text("SELECT customer_id FROM assets WHERE asset_id = :asset_id"),
                        {"asset_id": asset_id}
                    )
                    customer_id = result.scalar()
                    
                    # Insert the alert with all required fields
                    conn.execute(
                        text("""
                        INSERT INTO alerts 
                        (customer_id, asset_id, severity, alert_type, description, status, created_at, resolved_at, metadata)
                        VALUES 
                        (:customer_id, :asset_id, :severity, :alert_type, :description, :status, :created_at, :resolved_at, CAST(:metadata AS JSONB))
                        """),
                        {
                            "customer_id": customer_id,
                            "asset_id": alert["asset_id"],
                            "severity": alert["severity"],
                            "alert_type": alert["alert_type"],
                            "description": alert["description"],
                            "status": alert["status"],
                            "created_at": alert["created_at"],
                            "resolved_at": alert["resolved_at"],
                            "metadata": json.dumps(alert["metadata"])
                        }
                    )
            
            conn.commit()
            print("Successfully inserted test data!")
            
            # Print summary
            result = conn.execute(text("SELECT COUNT(*) FROM customers"))
            print(f"Inserted {result.scalar()} customers")
            
            result = conn.execute(text("SELECT COUNT(*) FROM assets"))
            print(f"Inserted {result.scalar()} assets")
            
            result = conn.execute(text("SELECT COUNT(*) FROM alerts"))
            print(f"Inserted {result.scalar()} alerts")
            
            # Print alert summary by severity
            result = conn.execute(text("SELECT severity, COUNT(*) FROM alerts GROUP BY severity"))
            print("\nAlerts by severity:")
            for row in result:
                print(f"  {row[0]}: {row[1]}")
                
        except Exception as e:
            print(f"Error inserting test data: {e}")
            raise

if __name__ == "__main__":
    insert_test_data()
