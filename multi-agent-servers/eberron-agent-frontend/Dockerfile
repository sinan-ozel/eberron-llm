FROM python:3.11-slim

RUN apt update
RUN apt install curl -y

# Clean up apt
RUN rm -rf /var/cache/apk/*
RUN apt-get autoremove -y
RUN apt-get clean

WORKDIR /app

RUN pip install streamlit==1.42.2

COPY src/streamlit_app.py .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py"]