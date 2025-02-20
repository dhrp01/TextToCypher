from torch.utils.data import Dataset

class Text2CypherDataset(Dataset):
    def __init__(self, dataset_split, tokenizer, max_length=512):
        self.dataset = dataset_split
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_point = self.dataset[idx]
        question = data_point["question"]  # User's natural language question
        schema = data_point["schema"]  # Database schema details
        database_reference_alias = data_point["database_reference_alias"] # Database alias name, might be useful in subgraph or cross-domain.
        cypher_query = data_point["cypher"]  # Target Cypher query

        # Combine question and schema as input
        input_text = f"Question: {question} Schema: {schema} Database Refenerce Alias: {database_reference_alias}"

        # Tokenize input (question + schema) and output (cypher query)
        inputs = self.tokenizer(input_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        outputs = self.tokenizer(cypher_query, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": outputs["input_ids"].squeeze(0),
        }
    

if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import T5Tokenizer
    from torch.utils.data import DataLoader
    dataset = load_dataset("neo4j/text2cypher-2024v1")

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    train_dataset = Text2CypherDataset(dataset["train"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Sample batch
    for batch in train_loader:
        print(batch)
        break
    
    