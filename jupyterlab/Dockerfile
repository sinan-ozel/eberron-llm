FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

ARG TESSERACT_VERSION="main"
ARG TESSERACT_URL="https://api.github.com/repos/tesseract-ocr/tesseract/tarball/${TESSERACT_VERSION}"

RUN apt update
RUN apt install -y curl
RUN apt install -y nodejs
RUN apt install -y npm
RUN apt install -y python3.10
RUN apt install -y python3-pip

RUN apt install -y wget
RUN apt install -y ca-certificates
RUN apt install -y apt-transport-https
RUN apt install -y g++
RUN apt install -y make
RUN apt install -y libleptonica-dev
RUN apt install -y libtool
RUN apt install -y pkg-config

WORKDIR /src
RUN wget -qO tesseract.tar.gz $TESSERACT_URL
RUN tar -xzf tesseract.tar.gz
RUN mv tesseract-* tesseract

WORKDIR /src/tesseract

RUN ./autogen.sh
RUN ./configure
RUN make
RUN make install
RUN ldconfig

RUN rm -rf /src

WORKDIR /usr/local/share/tessdata/

COPY jupyterlab/tesseract/get-languages.sh .
COPY jupyterlab/tesseract/languages.txt .
RUN chmod +x ./get-languages.sh
RUN ./get-languages.sh
RUN rm ./get-languages.sh
RUN rm ./languages.txt

RUN apt install unzip zip -y
RUN apt install rsync -y

# Clean up apt
RUN rm -rf /var/cache/apk/*
RUN apt-get autoremove -y
RUN apt-get clean

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
RUN pip3 install openparse==0.7.0

RUN pip3 install fastembed==0.3.3
RUN pip3 install flashrank==0.2.8

RUN pip3 install langchain-community==0.2.6
RUN pip3 install langchain-experimental==0.0.62
RUN pip3 install langchain-groq==0.1.6
RUN pip3 install langchain-qdrant==0.1.1
RUN pip3 install langchain==0.2.6
RUN pip3 install langchain-mistralai==0.1.12

RUN pip3 install lancedb==0.16.0

RUN pip3 install git+https://github.com/huggingface/huggingface_hub

RUN pip3 install mangoCR==0.1.4

EXPOSE 8888
WORKDIR /jupyterlab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir='/jupyterlab/notebooks'", "--allow-root"]