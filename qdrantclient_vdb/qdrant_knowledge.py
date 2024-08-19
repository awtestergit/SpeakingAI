# -*- coding: utf-8 -*-
# 
# @author awtestergit
# @description provide some simple Qdrantclient functionalities: add knowledge to vector db, and query vector db
# 

from qdrant_client import QdrantClient
from qdrant_client.http import models
from interface.interface_model import IEmbeddingModel

class KnowledgeWarehouse():
    def __init__(self, client:QdrantClient, collection_name:str, embedding_model:IEmbeddingModel) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding_model

    def add_knowledge(self, text:str):
        """
        add knowledge to vector database
        text: the raw text as meta data to be stored
        """
        texts = [text]
        vectors = [self.embedding_model.encode(text)]
        return self.add_knowledge_in_bulk(texts, vectors)

    def add_knowledge_in_bulk(self, texts:list):
        """
        add knowledge to vector database
        texts: a list of the raw text as meta data to be stored
        """
        #
        # this is a simple way of get point ids, assuming no record will be deleted so total count is used as base for point id
        #
        # record count in the vector db
        record_count = self.client.count(self.collection_name)
        record_count = record_count.count
        # number of texts to be added
        size = len(texts)
        #get point ids, starting at record_count in the length of size
        point_ids = [i+record_count for i in range(size)]

        # get vectors, payloads
        payloads = [{'meta': text} for text in texts]
        embedding_vectors = [self.embedding_model.encode(text) for text in texts]

        result = self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=point_ids,
                vectors=embedding_vectors,
                payloads=payloads,
            )
        )

    def query_knowledge(self, query:str, top_k=3, threshold=0.8)->list:
        """
        query the knowledge warehouse
        query: the query string
        top_k: top k results to be returned
        threshold: the threshold score, only higher than this score can be returned
        Output: a list of texts for this query, total number of the texts could be 0 ~ top_k
        """
        query_vector = self.embedding_model.encode(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            score_threshold=threshold,
        )
        # outputs
        outputs = [result.payload for result in results]

        return outputs