# Step 1 
FROM python:3.11-slim

# Step 2 — Set working directory inside container
WORKDIR /app

# Step 3 — Copy requirements first (before code)
COPY requirements.txt .

# Step 4 — Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5 — Download NLTK data inside container
RUN python -m nltk.downloader stopwords wordnet \
    averaged_perceptron_tagger averaged_perceptron_tagger_eng punkt punkt_tab

# Step 6 — Copy all project files into container
COPY . .

# Step 7 — Expose port 8000
EXPOSE 8000

# Step 8 — Command to run when container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
