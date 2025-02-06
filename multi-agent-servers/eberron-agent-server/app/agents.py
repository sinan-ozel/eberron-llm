from typing import List, Tuple
from transformers import pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.vectorstores.base import VectorStore

from agents_on_langchain.base_agent import BaseAgent

from models import model, tokenizer, vector_store, embeddings


class CanonicalSummaryAgent(BaseAgent):
    version = '01'
    base_llm = HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.1,
            max_new_tokens=768,
            do_sample=True
        )
    )

    def listen(self, context: str) -> bool:
        """This agent does not listen to any contextual information"""
        return False

    def __init__(self, vector_store: VectorStore, search_type: str, search_kwargs: dict):
        self.retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def _retrieve(self, q: str) -> List[Tuple[str, dict]]:
        self._last_retrieved_docs = self.retriever.invoke(q)

        return [(d.page_content, d.metadata) for d in self._last_retrieved_docs]

    def _prompt(self, q: str):
        retrieved_documents = [d[0] for d in self._retrieve(q)]

        retrieved_text = '\n\n'.join(retrieved_documents)

        self._last_prompt = dedent(f"""[INST]
        Use the following information (until the final cutoff =====) to answer the user query Q below.
        Prefer information closer to the top.

        {retrieved_text}

        =====

        Q:
        {q}

        A:
        [/INST]""")
        return self._last_prompt

    def respond(self, q: str):
        # empty_chunk_count = 0
        self._last_response = ""
        prompt = self._prompt(q)
        for chunk in self.base_llm.bind(skip_prompt=True).stream(prompt):
            self._last_response += chunk
            yield chunk
