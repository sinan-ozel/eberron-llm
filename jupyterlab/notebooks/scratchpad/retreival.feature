Feature: LLM Agent
    Scenario: Retrieving Documents
        Given an embedding model and document embeddings
        When I query fashion in Eberron
        Then one of the top 5 retrieved documents should include fashion in their title

    Scenario: Querying The LLM
        Given an embedding model
        When I query the refugee district in Sharn
        Then the response should contain High Walls