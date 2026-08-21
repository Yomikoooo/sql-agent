# Packaging: someone else takes the repository and it runs.
#
# A container is a harsher version of a clean clone. It has none of your home
# directory, no virtualenv, no .env and no Ollama already running. Working
# locally is not the same as working here, and the difference is exactly the
# set of things you did not notice you depended on.
FROM python:3.13-slim

WORKDIR /app

# Dependencies first, sources second. The order is deliberate: editing a line
# of code does not invalidate the layer above, which turns a rebuild from
# minutes into seconds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install the package itself. Without this the modules under src/ are not
# importable, and the console scripts below do not exist.
RUN pip install --no-cache-dir --no-deps -e .

# data/*.db is excluded by .dockerignore and has to be rebuilt here. seed_db
# uses a fixed seed, so the database in the image matches the one on your
# machine. check_tasks runs too, so a broken task set fails the build instead
# of surfacing at run time as bad data inside a shipped image.
RUN sql-agent-seed && sql-agent-check-tasks

# Do not run as root. Under an injection or an RCE, an ordinary user can do far
# less.
RUN useradd --create-home app && chown -R app:app /app
USER app

# 0.0.0.0 is required: binding 127.0.0.1 inside a container means only the
# container can connect, and everyone outside sees a connection refused that
# reads like an unmapped port.
ENV HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000

# The health check only hits /health, which does not call the model. Otherwise
# it burns money every 30 seconds, and a wobble at the provider gets the
# container declared unhealthy and restarted.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["sql-agent-serve"]
