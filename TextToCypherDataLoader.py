from datasets import load_dataset 
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import copy

IGNORE_INDEX = -100

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
    

class Text2CypherCasualCollator:
    def __init__(self, tokenizer, source_max_length=1024, target_max_lenght=256, train_on_source=False):
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_lenght
        self.train_on_source = train_on_source
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, batch):
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in batch]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in batch]

        tokenized_sources = self.tokenizer(
            sources, max_length=self.source_max_length, truncation=True, add_special_tokens=False
        )

        tokenized_targets = self.tokenizer(
            targets, max_length=self.target_max_length, truncation=True, add_special_tokens=False
        )

        input_ids, labels = [], []

        for src_ids, tgt_ids in zip(tokenized_sources["input_ids"], tokenized_targets["input_ids"]):
            full_input = src_ids + tgt_ids
            input_ids.append(torch.tensor(full_input))

            if self.train_on_source:
                full_label = copy.deepcopy(full_input)
            else:
                full_label = [IGNORE_INDEX] * len(src_ids) + copy.deepcopy(tgt_ids)

            labels.append(torch.tensor(full_label))
        
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.pad_token_id),
            "labels": labels
        }

# Use for generation of baseline    
class GenerationCollator:
    def __init__(self, tokenizer, source_max_len=1024):
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, batch):
        input_texts = [item["input"] for item in batch]

        tokenized = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.source_max_len,
            return_tensors="pt"
        )

        return tokenized


    

if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import T5Tokenizer
    from torch.utils.data import DataLoader
    dataset = load_dataset("neo4j/text2cypher-2024v1")

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    train_dataset = Text2CypherDataset(dataset["train"])
    collator = Text2CypherCasualCollator(tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collator)
    baseline_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=GenerationCollator(tokenizer))

    # Sample batch
    for batch in train_loader:
        print("Training batch:")
        print(batch)
        break

    # Sample baseline batch
    for batch in baseline_loader:
        print("Baseline batch:")
        print(batch)
        break

    # Validation during training
    # trainer.evaluate(eval_dataset=val_dataset)

    # # Test after training
    # trainer.predict(test_dataset=test_dataset)

    
    