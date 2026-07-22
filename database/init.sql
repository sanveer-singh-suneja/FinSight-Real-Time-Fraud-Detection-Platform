-- FinSight PostgreSQL Initialisation Script
-- Runs once when the container first starts

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- MLflow schema (used by the MLflow tracking server)
CREATE SCHEMA IF NOT EXISTS mlflow;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE finsight TO finsight;
GRANT ALL PRIVILEGES ON SCHEMA public TO finsight;
