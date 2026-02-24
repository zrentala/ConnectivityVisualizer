FROM python:3.11

# Create non-root user (HF requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app
COPY --chown=user . /app

# Hugging Face requires port 7860
ENV PORT=7860

CMD ["python", "app.py"]