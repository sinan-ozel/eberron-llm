import os
import pickle

import torch
import lancedb
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_core.vectorstores.base import VectorStore


ARTEFACT_VERSION = os.getenv('ARTEFACT_VERSION', '03')
ARTEFACT_ROOT_FOLDER = os.environ.get('ARTEFACT_ROOT_FOLDER', '/artefact')
ARTEFACT_FOLDER = os.path.join(ARTEFACT_ROOT_FOLDER, 'eberron', f'v{ARTEFACT_VERSION}')
HF_HOME = os.environ.get('HF_HOME')
MODEL_NAME = os.getenv("MODEL_NAME", "MockLLM")
MODEL_ORG = os.getenv("MODEL_ORG", "test")
MODEL_COMMIT_HASH = os.getenv("COMMIT_HASH", "mockhash")


with open(os.path.join(ARTEFACT_FOLDER, 'model_metadata.pkl'), 'rb') as f:
    model_metadata = pickle.load(f)

if model_metadata['embedding_format'] == 'pickle':
    with open(os.path.join(ARTEFACT_FOLDER, 'embeddings.pkl'), 'rb') as f:
        embeddings = pickle.load(f)
elif model_metadata['embedding_format'] == 'lancedb':
    embeddings_folder = os.path.join(ARTEFACT_FOLDER, 'embeddings')
    db = lancedb.connect(embeddings_folder)
    table = db.open_table('documents')


# Load the embedding model
embedding_model_org, embedding_model_name = model_metadata['embedding_model']['name'].split('/')
embedding_model_path = os.path.join(HF_HOME,
                                    'hub',
                                    f'models--{embedding_model_org}--{embedding_model_name}',
                                    'snapshots',
                                    model_metadata['embedding_model']['revision'])

embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path,
                                   model_kwargs={'device': 'cpu',
                                                 'trust_remote_code': True},
                                   encode_kwargs={'normalize_embeddings': True})


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


# model_provider = 'mistralai'
# model_name = 'Mistral-7B-Instruct-v0.3'
# model_revision = 'e0bc86c23ce5aae1db576c8cca6f06f1f73af2db'
model_path = os.path.join(HF_HOME,
                          'hub',
                          f'models--{MODEL_ORG}--{MODEL_NAME}',
                          'snapshots',
                          MODEL_COMMIT_HASH)
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.bfloat16,
                                             quantization_config=bnb_config,
                                             device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_path)

vector_store = LanceDB(connection=db,
                       table_name='documents',
                       embedding=embeddings)
