FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

RUN apt update
RUN apt install -y curl
RUN apt install -y nodejs
RUN apt install -y npm
RUN apt install -y python3.10
RUN apt install -y python3-pip

RUN pip install jupyterlab==4.2.5
RUN pip install ipywidgets==8.1.5
RUN pip install jupyter_contrib_nbextensions==0.7.0
RUN jupyter labextension enable @jupyterlab/toc

RUN pip3 install torch==2.4.1
RUN pip3 install tensorflow==2.17.0
RUN pip3 install datasets==3.0.0

RUN pip3 install git+https://github.com/huggingface/transformers
RUN pip3 install diffusers==0.30.0
# RUN pip3 install git+https://github.com/huggingface/diffusers
RUN pip3 install accelerate==0.34.2
RUN pip3 install bitsandbytes==0.43.3
RUN pip3 install flash-attn --no-build-isolation
RUN pip3 install git+https://github.com/huggingface/peft.git

RUN pip3 install lancedb==0.13.0

RUN pip3 install ir-measures==0.3.3
RUN pip3 install rank-bm25==0.2.2
RUN pip3 install RAGatouille==0.0.8.post4
RUN pip3 install gliner==0.2.11
RUN pip3 install rerankers[all]==0.5.3

RUN pip3 install nltk==3.9.1
RUN pip3 install sentence_transformers==2.5.1

RUN pip3 install pypdf==4.3.1

RUN pip3 install git+https://github.com/huggingface/huggingface_hub


EXPOSE 8888
WORKDIR /jupyterlab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir='/jupyterlab/notebooks'", "--allow-root"]