import json
import sys
from pathlib import Path

# Validate the support app can import and answer a policy question.
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from app.main import app

client = __import__('fastapi.testclient').testclient.TestClient(app)
response = client.post('/ask', json={'query': 'What is the delivery fee for orders below INR 149?'})
print('STATUS', response.status_code)
print(json.dumps(response.json(), ensure_ascii=False))
assert response.status_code == 200
assert 'INR 25' in response.json()['answer']
print('VERIFY_PASS')
