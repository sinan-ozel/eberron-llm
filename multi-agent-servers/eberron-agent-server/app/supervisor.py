from typing import Iterable
import re

from agents_on_langchain.base_agent import BaseAgent
from agents import (CanonicalSummaryAgent,
                    ImprovisorAgent,
                    CharacterPrompterAgent,
                    CharacterGeneratorAgent,
                    RequestClassifierAgent)


canonical_summary_agent = CanonicalSummaryAgent("similarity", {'k': 5})
improvisor_agent = ImprovisorAgent("similarity", {'k': 5})
character_prompter_agent = CharacterPrompterAgent("similarity", {'k': 10})
character_generator_agent = CharacterGeneratorAgent()
request_classifier = RequestClassifierAgent()


class SupervisorAgent(BaseAgent):
    version = '01'
    base_llm = None

    def listen(self, context: str) -> bool:
        """This agent does not listen to any contextual information"""
        return False

    def _retrieve(self, q: str):
        """This agent does not retrieve any contextual information."""
        return []

    def _prompt(self, q: str):
        """This agent does not require a prompt."""
        return ""

    def respond(self, q: str) -> Iterable[str]:

        category_classification = self.ask(request_classifier, q)
        category = self.receive(category_classification)

        q = re.sub(r'^[a-zA-Z0-9]+ +', '', q)

        if category == 'character':
            yield from self.ask(character_prompter_agent, q)
            yield from self.ask(character_generator_agent, self.received_response)
        elif category == 'lookup':
            yield from self.ask(canonical_summary_agent, q)
        elif category == 'original':
            yield from self.ask(improvisor_agent, q)

    def run() -> None:
        pass

supervisor = SupervisorAgent()