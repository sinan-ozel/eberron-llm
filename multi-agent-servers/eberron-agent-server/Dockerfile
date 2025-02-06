FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

RUN apt update
RUN apt install -y curl
RUN apt install -y nodejs
RUN apt install -y npm
RUN apt install -y python3.10
RUN apt install -y python3-pip

RUN apt install -y wget
RUN apt install -y ca-certificates
RUN apt install -y apt-transport-https


# Clean up apt
RUN rm -rf /var/cache/apk/*
RUN apt-get autoremove -y
RUN apt-get clean

RUN pip3 install torch==2.4.1
RUN pip3 install tensorflow==2.18.0
RUN pip3 install datasets==3.0.0

RUN pip3 install git+https://github.com/huggingface/transformers@919220dab1e29f4d04eacd61a197a45a4fec2613
RUN pip3 install diffusers==0.30.0
RUN pip3 install accelerate==0.34.2
RUN pip3 install bitsandbytes==0.43.3
RUN pip3 install flash-attn --no-build-isolation
RUN pip3 install git+https://github.com/huggingface/peft.git

RUN pip3 install ir-measures==0.3.6
RUN pip3 install rank-bm25==0.2.2
RUN pip3 install gliner==0.2.11
RUN pip3 install rerankers[all]==0.5.3

RUN pip3 install tf-keras==2.18.0
RUN pip3 install langchain==0.3.15
RUN pip3 install langchain-community==0.3.15
RUN pip3 install langchain-experimental==0.3.4
RUN pip3 install langchain-huggingface==0.1.2

RUN pip3 install pylance==0.22.0
RUN pip3 install lancedb==0.18.0
RUN pip3 install agents-on-langchain==0.1.0.dev202502010517

RUN pip3 install git+https://github.com/huggingface/huggingface_hub@3618a38f80ae3073b95ea0fba26f363d2ec4ece8

EXPOSE 80
WORKDIR /app
COPY app/*.py /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]