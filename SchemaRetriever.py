from sentence_transformers import SentenceTransformer
import faiss
from annoy import AnnoyIndex
import numpy as np
from rank_bm25 import BM25Okapi  
import re
class getSchema:
  # Load an embedding model
  def __init__(self):
    self.model = SentenceTransformer("all-MiniLM-L6-v2")
  def get_embedded_schemas(self,schemas):
    self.schema_embeddings = self.model.encode(schemas)

    # Store embeddings in FAISS index
    dimension = self.schema_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(self.schema_embeddings))
    return index
  
  def retrieve_schema_faiss(self,schemas,query):
    query_embedding = self.model.encode([query])
    index = self.get_embedded_schemas(schemas)
    distances, indices = index.search(query_embedding, 1)  # Retrieve top match
    best_match = schemas[indices[0][0]]
    return best_match

  def retrieve_schema_annoy(self,schemas,query):
    dimension = 384  # Adjust based on embedding model
    annoy_index = AnnoyIndex(dimension, 'angular')
    self.schema_embeddings = self.model.encode(schemas)
    for i, emb in enumerate(self.schema_embeddings):
        annoy_index.add_item(i, emb)
    annoy_index.build(10)  # Build with 10 trees
    query_embedding = self.model.encode(query)
    index = annoy_index.get_nns_by_vector(query_embedding, 1)[0]
    return schemas[index]

  def retrieve_schema_bm25(self,schemas,query):
    def tokenize(text):
        return re.findall(r'\w+', text.lower())
    tokenized_schemas = [tokenize(schema.lower()) for schema in schemas]  
    bm25 = BM25Okapi(tokenized_schemas)  
    query_tokens = tokenize(query.lower())  
    scores = bm25.get_scores(query_tokens)  
    best_match = schemas[scores.argmax()]  
    return best_match  


  





# # Example question
# query = "Which schema stores information about financial transactions?"
# get_schema_object = getSchema()

# print("Best matching schema:\n", get_schema_object.retrieve_schema_annoy(schemas,query))
# print("Best matching schema:\n", get_schema_object.retrieve_schema_bm25(schemas,query))
# print("Best matching schema:\n", get_schema_object.retrieve_schema_faiss(schemas,query))
