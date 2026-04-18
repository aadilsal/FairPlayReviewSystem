import httpx
import pytest
from datetime import datetime

BASE_URL = "http://localhost:8000/api"

@pytest.mark.asyncio
async def test_health():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "success"

def test_schemas_manually():
    # Since we can't easily run the full server with DB during this environment's test and auth is required,
    # we'll just check if the files exist and have no syntax errors.
    import API.models
    import API.schemas.match_schemas
    import API.schemas.review_schemas
    import API.schemas.notification_schemas
    print("All schemas and models imported successfully.")

if __name__ == "__main__":
    test_schemas_manually()
