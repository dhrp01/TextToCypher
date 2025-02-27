import torch
from torch.utils.data import Dataset

class Text2CypherDataset(Dataset):
    def __init__(self, dataset_split, tokenizer, max_length=512, use_cuda=True):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.use_cuda = use_cuda

        self.questions = [data["question"] for data in dataset_split]
        self.schemas = [data["schema"] for data in dataset_split]
        self.cypher_queries = [data["cypher"] for data in dataset_split]

        self.tokenized_inputs = self.tokenizer(
            [f"Question: {q} Schema: {s}" for q, s in zip(self.questions, self.schemas)],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        self.tokenized_outputs = self.tokenizer(
            self.cypher_queries,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        if use_cuda and torch.cuda.is_available():
            self.tokenized_inputs = {k: v.to("cuda") for k, v in self.tokenized_inputs.items()}
            self.tokenized_outputs = {k: v.to("cuda") for k, v in self.tokenized_outputs.items()}


    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
       return {
            "question": self.questions[idx],
            "schema": self.schemas[idx],
            "cypher": self.cypher_queries[idx],
            "input_ids": self.tokenized_inputs["input_ids"][idx],
            "attention_mask": self.tokenized_inputs["attention_mask"][idx],
            "labels": self.tokenized_outputs["input_ids"][idx]
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
    
    