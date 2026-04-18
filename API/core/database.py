from supabase import create_client, Client
from API.core.config import settings

# Initialize Supabase client
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# For backwards compatibility with SQLAlchemy code
Base = None
SessionLocal = None

