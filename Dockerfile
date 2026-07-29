# Runtime image for SentryQuery: the Streamlit UI by default, with the ingest
# mode and the MCP server reachable by overriding the command (see
# docker-compose.yml).
#
# No secret is ever baked in. There is no ARG or ENV for a key, .env is excluded
# by .dockerignore, and keys are injected at run time only (compose uses
# env_file). Layers are cached so a source edit does not reinstall deps.
FROM python:3.11-slim

# PYTHONDONTWRITEBYTECODE keeps .pyc files out of the image; PYTHONUNBUFFERED
# makes logs appear immediately rather than sitting in a buffer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, in their own layer. Because requirements.txt is copied and
# installed before any source, editing a .py file reuses this layer instead of
# triggering a full reinstall. Runtime deps only: requirements-dev.txt (ruff,
# pytest) is intentionally not installed here.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Source last, so it is the only layer that rebuilds on a code change.
COPY *.py ./

# Run as a non-root user. Created after the install so pip still writes to
# system site-packages, and given ownership of /app only.
RUN useradd --create-home --shell /usr/sbin/nologin sentry \
    && chown -R sentry:sentry /app
USER sentry

EXPOSE 8501

# Streamlit publishes its own health endpoint. A failing check means the UI is
# up but not serving, which is what we want to catch.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

# Default: the Streamlit UI. Bound to 0.0.0.0 so the published port reaches it
# from outside the container.
CMD ["python", "-m", "streamlit", "run", "sentry_query.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
