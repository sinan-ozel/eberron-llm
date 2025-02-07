from typing import List, Tuple, Iterable
from textwrap import dedent

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

    def __init__(self, search_type: str, search_kwargs: dict):
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

    def run() -> None:
        pass


class ImprovisorAgent(BaseAgent):
    version = '01'
    base_llm = HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.7,
            max_new_tokens=768,
            do_sample=True
        )
    )

    def listen(self, context: str) -> bool:
        """This agent does not listen to any contextual information."""
        return False

    def __init__(self, search_type: str, search_kwargs: dict):
        self.retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def _retrieve(self, q: str) -> List[Tuple[str, dict]]:
        self._last_retrieved_docs = self.retriever.invoke(q)

        return [(d.page_content, d.metadata) for d in self._last_retrieved_docs]

    def _prompt(self, q: str):
        retrieved_documents = [d[0] for d in self._retrieve(q)]

        retrieved_text = '\n\n'.join(retrieved_documents)

        self._last_prompt = dedent(f"""[INST]
        Use the following information as inspiration (until the final cutoff =====) to answer the user query Q below.
        Be creative.
        Come up with interesting information.


        {retrieved_text}

        =====

        Q:
        {q}

        A:
        [/INST]""")
        return self._last_prompt

    def respond(self, q: str) -> Iterable[str]:
        prompt = self._prompt(q)
        self._last_response = ""
        for chunk in self.base_llm.bind(skip_prompt=True).stream(prompt):
            self._last_response += chunk
            yield chunk

    def run() -> None:
        pass


class CharacterPrompterAgent(BaseAgent):
    version = '01'
    base_llm = HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.9,
            max_new_tokens=512,
            do_sample=True
        )
    )

    def listen(self, context: str) -> bool:
        """This agent does not listen to any contextual information"""
        return False

    def __init__(self, search_type: str, search_kwargs: dict):
        self.retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def _retrieve(self, q: str) -> List[Tuple[str, dict]]:
        self._last_retrieved_docs = self.retriever.invoke(q)

        return [(d.page_content, d.metadata) for d in self._last_retrieved_docs]

    def _prompt(self, q: str):
        retrieved_documents = [d[0] for d in self._retrieve(q)]

        retrieved_text = '\n\n'.join(retrieved_documents)

        self._last_prompt = dedent(f"""[INST]
        Use the following information as inspiration (until the final cutoff =====) to come up with a character concept that fits the user query Q.
        A character concept includes one paragraph on backstory, one paragraph on long-term goals, and one paragraph of immediate wants and needs.
        Sugggest a few persoanlity traits and secrets.
        Come up with relevant D&D 5e skill checks for players to notice or figure out certain quirks or secrets of the character.


        {retrieved_text}

        =====

        Q:
        {q}

        A:
        [/INST]""")
        return self._last_prompt

    def respond(self, q: str) -> Iterable[str]:
        prompt = self._prompt(q)
        self._last_response = ""
        for chunk in self.base_llm.bind(skip_prompt=True).stream(prompt):
            self._last_response += chunk
            yield chunk

    def run() -> None:
        pass


class CharacterGeneratorAgent(BaseAgent):
    version = '01'
    base_llm = HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.2,
            max_new_tokens=768,
            do_sample=True
        )
    )

    def listen(self, context: str) -> bool:
        """This agent does not listen to any contextual information"""
        return False

    # def __init__(self, vector_store: VectorStore, search_type: str, search_kwargs: dict):
    #     self.retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def __init__(self):
        pass

    def _retrieve(self, q: str) -> List[Tuple[str, dict]]:
        # TODO: Change this retrieve information based on the class...
        # self._last_retrieved_docs = self.retriever.invoke(q)

        # return [(d.page_content, d.metadata) for d in self._last_retrieved_docs]
        return []

    def _prompt(self, q: str):
        retrieved_documents = [d[0] for d in self._retrieve(q)]

        retrieved_text = '\n\n'.join(retrieved_documents)

        # Use the prompt P to create a complete character sheet based on the 5th Edition rules for the world of Eberron.
        # A complete character sheet should include name, race, class, level, attributes, proficiencies based on the class and fifth edition rules,

        self._last_prompt = dedent(f"""[INST]
        Use the following character description to create a complete character sheet based on the 5th Edition rules for the world of Eberron.
        Make sure that the class fits the description, and make sure that the name is the same as in the description.

        Character Description:
        {q}
        ========

        Create the character sheet below.
        A complete character sheet should include race, class, level, attributes, proficiencies based on the class and fifth edition rules,
        proficiency bonus based on class and level, saving throws bonus based on class, feats and proficiency , hit points based on class and level,
        weapons and equipment, attack bonus for each weapon based on abilities, any race or class based features, class and level, attack roll,
        and equipment, armor class based on dexterity and armor, and if the character has spellcasting ability, spells based on class and level,
        as well as the number of spell slots.
        Include an inventory based on their level, class, and race, with a few personal touches in the items.

        {retrieved_text}

        Character Sheet:
        [/INST]""")
        return self._last_prompt

    def respond(self, q: str) -> Iterable[str]:
        # empty_chunk_count = 0
        prompt = self._prompt(q)
        self._last_response = ""
        for chunk in self.base_llm.bind(skip_prompt=True).stream(prompt):
            self._last_response += chunk
            yield chunk

    def run() -> None:
        pass


class RequestClassifierAgent(BaseAgent):
    version = '01'
    base_llm = HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.1,
            max_new_tokens=64,
            do_sample=True
        )
    )

    # TODO: Move this evaluation & classification to another agent.
    q_and_a = [
        ("Tell me about the languages of Eberron.", "lookup"),
        ("War veteran", "character"),
        ("Create a House Cannith item.", "original"),
        ("Create the details of a town on the border between Zilargo and Breland.", "original"),
        ("Groggy comic relief", "character"),
        ("Tell me about fashion in the five nations.", "lookup"),
        ("Find for me a magic lipstick.", "lookup"),
    ]

    def listen(self, context: str) -> bool:
        """This agent does not listen to any contextual information"""
        return False

    def __init__(self):
        pass

    def _retrieve(self, q: str) -> List[Tuple[str, dict]]:
        return []

    def _prompt(self, q: str):
        return dedent(f"""[INST]
        User is making a request. Classify the request into one of the three categories. Response with only one word.
        If the user is asking to create some original content, respond with the word "original".
        If this is a request to create a character or NPC, or it looks like it is decribing a D&D character, respond with the word "character".
        If the user is making a request to find out information, respond with the word "lookup".

        Q:
        {q}

        A:
        [/INST]""")

    def respond(self, q: str) -> Iterable[str]:
        if q.startswith("/character"):
            yield 'character'
            return

        category = self.base_llm.bind(skip_prompt=True).invoke(self._prompt(q))

        yield category.strip()

    def run() -> None:
        pass
