FROM python:3.12-slim

RUN pip install --no-cache-dir \
    pytest==8.3.4 \
    pytest-asyncio==0.24.0 \
    pydantic==2.10.6 \
    requests==2.32.3 \
    httpx==0.28.1 \
    ruff==0.9.7

WORKDIR /workspace

CMD ["tail", "-f", "/dev/null"]
