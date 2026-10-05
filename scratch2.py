import asyncio
import traceback
from fastapi.testclient import TestClient

# Mock models before importing main
import sys
from unittest.mock import MagicMock
sys.modules['specialist_models.vision_vqa'] = None
sys.modules['specialist_models'] = MagicMock()

from ai_agent.main import app

client = TestClient(app)

def test():
    print("Sending request...")
    try:
        response = client.post(
            "/api/v1/query-with-image",
            data={"query": "Are there any open green spaces or water channels visible within this dense urban fabric?"},
            files={"file": ("dummy.jpg", b"dummy content")}
        )
        print("Status:", response.status_code)
        print("Response:", response.text)
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    test()
