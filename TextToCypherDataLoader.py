from datasets import load_dataset 
from torch.utils.data import Dataset



instruction = """Given an input question, convert it to a Cypher query.
        To translate a question into a Cypher query, please follow these steps:

        1. Carefully analyze the provided graph schema to understand what nodes, relationships, and properties are available. Pay attention to the node labels, relationship types, and property keys.
        2. Identify the key entities and relationships mentioned in the natural language question. Map these to the corresponding node labels, relationship types, and properties in the graph schema.
        3. Think through how to construct a Cypher query to retrieve the requested information step-by-step. Focus on:
        - Identifying the starting node(s) 
        - Traversing the necessary relationships
        - Filtering based on property values
        - Returning the requested information
        Feel free to use multiple MATCH, WHERE, and RETURN clauses as needed.
        4. Explain how your Cypher query will retrieve the necessary information from the graph to answer the original question. Provide this explanation inside <explanation> tags.
        5. Once you have finished explaining, construct the Cypher query inside triple backticks ```cypher```.

        Remember, the goal is to construct a Cypher query that will retrieve the relevant information to answer the question based on the given graph schema.
        Carefully map the entities and relationships in the question to the nodes, relationships, and properties in the schema.
        Additional instructions:
        1. **Array Length**: Always use `size(array)` instead of `length(array)` to get the number of elements in an array.
        2. **Implicit aggregations**: Always use intermediate WITH clause when performing aggregations
        3. **Target Neo4j version is 5**: Use Cypher syntax for Neo4j version 5 and above. Do not use any deprecated syntax.

        Based on the Neo4j graph schema below, write a Cypher query that would answer the user's question:
        {schema}

        Question: {question}
        """

class Text2CypherDataset(Dataset):
    def __init__(self, dataset_split):
        dataset_list = dataset_split.to_dict()
        self.questions = dataset_list["question"]
        self.schemas = dataset_list["schema"]
        self.cypher_queries = dataset_list["cypher"]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        try:
            return {
                    "input": instruction.format(schema=self.schemas[idx], question=self.questions[idx]),
                    "output": self.cypher_queries[idx]
                }
        except Exception as e:
            print(f"Error at idx {idx}")
            raise e
    

class Text2CypherCollator:
    def __init__(self, tokenizer, max_input_length=512, max_output_length=512):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, batch):
        input_texts = [item["input"] for item in batch]
        output_texts = [item["output"] for item in batch]

        inputs = self.tokenizer(
            input_texts,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        outputs = self.tokenizer(
            output_texts,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = outputs["input_ids"].clone()
        labels[labels == self.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels
        }
    

if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import T5Tokenizer
    from torch.utils.data import DataLoader
    dataset = load_dataset("neo4j/text2cypher-2024v1")

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    train_dataset = Text2CypherDataset(dataset["train"])
    collator = Text2CypherCollator(tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collator)

    # Sample batch
    for batch in train_loader:
        print(batch)
        break
    
    