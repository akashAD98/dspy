import lancedb
import numpy as np
import openai
import dspy
from typing import List, Union, Optional


class LancedbRM(dspy.Retrieve):
    """
    A retrieval module that uses LanceDB for vector similarity searches.

    Args:
        table_name (str): Name of the LanceDB table containing vector data.
        uri (str): URI for connecting to LanceDB.
        openai_embed_model (str, optional): OpenAI embedding model. Defaults to "text-embedding-ada-002".
        openai_api_key (str, optional): API key for OpenAI. If not provided, should be set in environment.
        k (int, optional): Number of top results to retrieve. Defaults to 3.

    Examples:
        retriever_model = LancedbRM('my_table', 'data/sample-lancedb')
        dspy.settings.configure(rm=retriever_model)
        result = retriever_model("your query")
    """

    def __init__(
        self,
        table_name: str,
        uri: str,
        openai_embed_model: str = "text-embedding-ada-002",
        openai_api_key: Optional[str] = None,
        k: int = 3,
    ):
        if openai_api_key:
            openai.api_key = openai_api_key
        self._openai_embed_model = openai_embed_model
        self.db = lancedb.connect(uri)
        self.table = self.db[table_name]
        super().__init__(k=k)

    def _get_embeddings(self, queries: List[str]) -> List[np.ndarray]:
        """Return query vectors after creating embeddings using OpenAI."""
        embedding = openai.Embedding.create(
            input=queries, model=self._openai_embed_model)
        return [np.array(embedding["data"][i]["embedding"]) for i in range(len(queries))]

    def forward(self, query_or_queries: Union[str, List[str]]) -> dspy.Prediction:
        """Perform a similarity search in LanceDB for the given query."""
        queries = [query_or_queries] if isinstance(
            query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]  # Filter empty queries
        embeddings = self._get_embeddings(queries)

        results = []
        for query_embedding in embeddings:
            search_result = self.table.search(
                query_embedding.tolist(), k=self.k)
            # Assuming 'item' contains the desired text
            passages = [result["item"] for result in search_result]
            results.extend(passages)

        return dspy.Prediction(passages=results)

# Example usage:
# retriever_model = LancedbRM('my_table', 'data/sample-lancedb')
# result = retriever_model("What did the president say about Ketanji Brown Jackson")
