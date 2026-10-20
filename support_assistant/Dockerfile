# Dockerfile for the Zepto Support Assistant (FastAPI + LangGraph + ChromaDB)
#
# Build and run locally (from inside support_assistant/):
#   docker build -t zepto-support-assistant .
#   docker run -p 7860:7860 zepto-support-assistant
# Then call POST http://127.0.0.1:7860/ask  (or open /docs)

FROM python:3.10-slim

# Run as a non-root user (uid 1000), as Hugging Face Spaces expects
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# 1) Install the CPU-only build of PyTorch first. sentence-transformers needs torch,
#    and the default build includes large GPU libraries this service does not use.
# 2) Then install the remaining dependencies.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user "torch==2.8.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

# Copy the application code and the 8 policy documents
COPY --chown=user app ./app
COPY --chown=user docs ./docs

# Graded baseline: fully offline mock mode unless MOCK_LLM=0 is set at run time
ENV MOCK_LLM=1

# Download all-MiniLM-L6-v2 and build the ChromaDB collection during the build,
# so the container starts quickly and needs no download at run time
RUN python -c "from app.corpus_loader import get_collection; print('Documents embedded:', get_collection().count())"

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]