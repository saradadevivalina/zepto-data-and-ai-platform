import os
import sys

print('START')
print('CWD', os.getcwd())
sys.path.insert(0, '.')
import app.main
print('IMPORT_OK', app.main.app.title)
from fastapi.testclient import TestClient
client = TestClient(app.main.app)
health = client.get('/health')
print('HEALTH', health.status_code, health.json())
response = client.post('/ask', json={'query': 'What is the delivery fee for orders below INR 149?'})
print('ASK', response.status_code)
print(response.json())
