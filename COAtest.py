from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Initialize the LLM (ensure you have set up OpenAI API key in your environment)
llm = OpenAI(model_name="gpt-3.5-turbo")

# Define a prompt template for query sketch generation
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    Given the graph schema:
    Node properties:
    - **Movie**: `title`: STRING, `votes`: INTEGER, `tagline`: STRING, `released`: INTEGER
    - **Person**: `born`: INTEGER, `name`: STRING
    Relationships:
    - (:Person)-[:ACTED_IN]->(:Movie)
    - (:Person)-[:DIRECTED]->(:Movie)
    - (:Person)-[:PRODUCED]->(:Movie)
    - (:Person)-[:WROTE]->(:Movie)
    - (:Person)-[:FOLLOWS]->(:Person)
    - (:Person)-[:REVIEWED]->(:Movie)
    
    Convert the following natural language question into a query sketch without filling in specific details:
    
    Question: {question}
    Query Sketch:
    """
)

def generate_query_sketch(question):
    formatted_prompt = prompt.format(question=question)
    response = llm(formatted_prompt)
    return response

question = "List all movies with a tagline containing the word 'Real World'."
query_sketch = generate_query_sketch(question)
print(query_sketch)

# TODO
# load the question datasets

