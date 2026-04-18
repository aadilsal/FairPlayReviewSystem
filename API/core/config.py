import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Supabase Configuration
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    
    # JWT Configuration
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "supersecretkey")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))
    
    # Database (kept for compatibility)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "supabase")

    # Media tooling
    FFMPEG_BINARY: str = os.getenv("FFMPEG_BINARY", "")

    # Snick detection tuning
    SNICK_AUDIO_SAMPLE_RATE: int = int(os.getenv("SNICK_AUDIO_SAMPLE_RATE", 16000))
    SNICK_LOW_HZ: int = int(os.getenv("SNICK_LOW_HZ", 1200))
    SNICK_HIGH_HZ: int = int(os.getenv("SNICK_HIGH_HZ", 6500))
    SNICK_PEAK_PROMINENCE: float = float(os.getenv("SNICK_PEAK_PROMINENCE", 2.5))
    SNICK_ALIGN_WINDOW_MS: int = int(os.getenv("SNICK_ALIGN_WINDOW_MS", 80))
    SNICK_VISUAL_WEIGHT: float = float(os.getenv("SNICK_VISUAL_WEIGHT", 0.45))
    SNICK_AUDIO_WEIGHT: float = float(os.getenv("SNICK_AUDIO_WEIGHT", 0.55))
    SNICK_DETECT_THRESHOLD: float = float(os.getenv("SNICK_DETECT_THRESHOLD", 0.62))
    SNICK_LOW_THRESHOLD: float = float(os.getenv("SNICK_LOW_THRESHOLD", 0.30))

settings = Settings()
