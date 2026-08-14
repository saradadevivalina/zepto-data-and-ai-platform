import sys
sys.path.insert(0, '.')

print("Step 1: Import FastAPI")
from fastapi import FastAPI, HTTPException
print("✓ FastAPI imported")

print("Step 2: Import corpus_loader")
from app.corpus_loader import initialize_vector_store
print("✓ corpus_loader imported")

print("Step 3: Import graph")
from app.graph import build_support_graph
print("✓ graph imported")

print("Step 4: Import schemas")
from app.schemas import AskRequest, SupportResponse
print("✓ schemas imported")

print("Step 5: Create FastAPI app")
app = FastAPI(title="Test")
print("✓ FastAPI app created")

print("All imports successful!")
