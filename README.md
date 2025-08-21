# Eberron DM Helper Agent

In January 2025, I created a Multi-Agent Platform for the aptly-named bootcamp [Build Multi-Agent Applications](https://maven.com/aggregate-intellect/llm-systems).

This repo contains the prototype in the notebooks and server code.

# How To Run

If you want to run this, you will need a basic knowledge of AWS systems, Infrastructure as Code and deployment tools and methods, LLMs, and of course, D&D and the Eberron campaign setting.

## Requirements
1. An AWS Account
2. docker-compose installed locally
3. A corpus of Eberron-themed books in PDF files. (This is going to work with pretty much any other PDF corpus. It is a general chat-with-your-documents tool.)

## Infrastructure

Another repo, [https://github.com/sinan-ozel/jupyterlab-on-kubernetes] contains the infrastructure. 
This repo uses JupyterLab to deploy and install... another JupyterLab, but this time on AWS and one a GPU-powered node. 
This gives the user the high GPU availability that is usually not found in local machines. 
Do this first, run the JupyterLab on AWS.

## RAG & LLM

Now upload the documents from the current repo, from inside the /jupyterlab/notebooks/eberron folder to the AWS JupyterLab.

Now upload your corpus and put it into an appropriate folder in the cloud JupyterLab.

Go through the notebooks. They are numbered, so it should be relatively easy to run in sequence.
Once you get to the fourth notebook, you should be able to ask the hosted model (`Mistral-7B-Instruct-0.3`, unless you change it) questions about the documents you uploaded.




