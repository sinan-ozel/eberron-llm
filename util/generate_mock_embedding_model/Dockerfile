# Use lightweight Python image
FROM python:3.10-slim

RUN rm -rf /var/cache/apk/*
RUN apt update
RUN apt install -y git
RUN apt-get autoremove -y
RUN apt-get clean


# Install PyTorch (for saving the .bin file)
RUN pip install torch==2.4.1
RUN pip install git+https://github.com/huggingface/transformers@919220dab1e29f4d04eacd61a197a45a4fec2613
RUN pip install langchain==0.3.15
RUN pip install langchain-community==0.3.15
RUN pip install langchain-huggingface==0.1.2

# Set environment variable for Hugging Face cache (inside container)
ENV HF_HOME=/hf

# Create a working directory
WORKDIR /app

# Copy the script into the container
COPY generate_mock_embedding_model.py .

# Run the script when the container starts
CMD ["python", "generate_mock_embedding_model.py"]
