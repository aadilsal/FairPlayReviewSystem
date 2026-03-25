from supabase import create_client, Client
from API.core.config import settings

# Initialize Supabase client
supabase_client: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

def get_supabase():
    """Get Supabase client instance"""
    return supabase_client

# Table names constants
USERS_TABLE = "users"
MATCHES_TABLE = "matches"
REVIEWS_TABLE = "reviews"
NOTIFICATIONS_TABLE = "notifications"

# Helper functions for common operations
async def insert_record(table: str, data: dict):
    """Insert a record into a table"""
    response = supabase_client.table(table).insert(data).execute()
    if response.data:
        return response.data[0]
    return None

async def get_record(table: str, id: int):
    """Get a record by ID"""
    response = supabase_client.table(table).select("*").eq("id", id).execute()
    if response.data:
        return response.data[0]
    return None

async def get_all_records(table: str):
    """Get all records from a table"""
    response = supabase_client.table(table).select("*").execute()
    return response.data

async def update_record(table: str, id: int, data: dict):
    """Update a record"""
    response = supabase_client.table(table).update(data).eq("id", id).execute()
    if response.data:
        return response.data[0]
    return None

async def delete_record(table: str, id: int):
    """Delete a record"""
    response = supabase_client.table(table).delete().eq("id", id).execute()
    return True

async def query_records(table: str, filter_key: str, filter_value):
    """Query records with a filter"""
    response = supabase_client.table(table).select("*").eq(filter_key, filter_value).execute()
    return response.data
