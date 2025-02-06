FROM python:3.10-slim

WORKDIR /tests/

COPY ./*.py .

RUN pip install pytest
RUN pip install requests

CMD ["pytest", "--maxfail=1", "--disable-warnings", "--tb=short"]
