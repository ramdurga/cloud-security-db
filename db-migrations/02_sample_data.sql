-- Insert sample customers
INSERT INTO customers (name, email, subscription_tier, is_active) VALUES
('Acme Corp', 'security@acme.com', 'ENTERPRISE', true),
('Globex Corporation', 'security@globex.com', 'BUSINESS', true),
('Initech', 'infosec@initech.com', 'STANDARD', true),
('Umbrella Corp', 'security@umbrella.com', 'ENTERPRISE', false);

-- Insert sample assets
-- Acme Corp assets
INSERT INTO assets (customer_id, asset_name, asset_type, cloud_provider, region, is_encrypted, last_scan_time) VALUES
(1, 'acme-prod-db-01', 'RDS', 'AWS', 'us-west-2', true, NOW() - INTERVAL '1 day'),
(1, 'acme-prod-ec2-01', 'EC2', 'AWS', 'us-west-2', false, NOW() - INTERVAL '2 hours'),
(1, 'acme-dev-s3', 'S3', 'AWS', 'us-east-1', true, NOW() - INTERVAL '6 hours'),

-- Globex Corporation assets
(2, 'globex-prod-db-01', 'Cloud SQL', 'GCP', 'us-central1', true, NOW() - INTERVAL '3 hours'),
(2, 'globex-backup-bucket', 'Cloud Storage', 'GCP', 'us-central1', true, NOW() - INTERVAL '1 day'),

-- Initech assets
(3, 'initech-vm-01', 'VM', 'Azure', 'eastus', false, NOW() - INTERVAL '12 hours'),
(3, 'initech-sql-01', 'SQL Database', 'Azure', 'eastus', true, NOW() - INTERVAL '4 hours'),

-- Umbrella Corp assets (inactive customer)
(4, 'umbrella-prod-db-01', 'RDS', 'AWS', 'us-east-1', true, NOW() - INTERVAL '7 days');

-- Insert sample alerts
-- Acme Corp alerts
INSERT INTO alerts (customer_id, asset_id, severity, alert_type, description, status, created_at, metadata) VALUES
(1, 1, 'HIGH', 'UNAUTHORIZED_ACCESS', 'Multiple failed login attempts detected', 'OPEN', NOW() - INTERVAL '2 hours', '{"attempts": 15, "source_ip": "192.168.1.100"}'),
(1, 2, 'MEDIUM', 'VULNERABILITY_SCAN', 'Critical vulnerability found in system packages', 'OPEN', NOW() - INTERVAL '5 hours', '{"cve": "CVE-2023-1234", "package": "openssl"}),
(1, 3, 'LOW', 'CONFIGURATION_ISSUE', 'S3 bucket has public read access', 'OPEN', NOW() - INTERVAL '1 day', '{"permission": "public-read"}'),

-- Globex Corporation alerts
(2, 4, 'CRITICAL', 'MALWARE_DETECTED', 'Ransomware signature detected', 'IN_PROGRESS', NOW() - INTERVAL '3 hours', '{"malware_type": "ransomware", "file_path": "/tmp/suspicious.exe"}'),
(2, 5, 'MEDIUM', 'ENCRYPTION_OFF', 'Storage bucket has encryption disabled', 'OPEN', NOW() - INTERVAL '2 days', '{"bucket": "globex-backup-bucket"}'),

-- Initech alerts
(3, 6, 'HIGH', 'BRUTE_FORCE', 'SSH brute force attack detected', 'RESOLVED', NOW() - INTERVAL '3 days', '{"port": 22, "source_ips": ["45.33.1.1", "45.33.1.2"]}'),
(3, 7, 'LOW', 'CERTIFICATE_EXPIRING', 'SSL certificate expiring in 15 days', 'OPEN', NOW() - INTERVAL '1 day', '{"expiry_date": "2023-12-31"}'),

-- Umbrella Corp alerts (inactive customer)
(4, 8, 'HIGH', 'UNAUTHORIZED_ACCESS', 'Suspicious login from new location', 'OPEN', NOW() - INTERVAL '10 days', '{"location": "Unknown", "ip": "198.51.100.1"}');
