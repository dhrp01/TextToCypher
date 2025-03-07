import re
import json
import logging
import neo4j
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CypherGenerator:
    def __init__(
        self,
        schema: Dict,
        neo4j_uri: Optional[str] = None,
        neo4j_auth: Optional[tuple] = None,
        openai_api_key: Optional[str] = None,
        llm_model: str = "gpt-4 "
    ):
        self.schema = schema
        self.llm = OpenAI(api_key=openai_api_key)
        self.llm_model = llm_model
        self.neo4j_driver = (
            GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
            if neo4j_uri and neo4j_auth
            else None
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((json.JSONDecodeError, ValueError)),
        reraise=True
    )

    # WHAT IS THIS FUNCTION
    def _llm_chat_completion(self, prompt: str) -> Dict:
        """Execute LLM call with structured output handling"""
        try:
            response = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a Cypher query expert. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM API error: {str(e)}")
            raise
    
    # SCHEM GEN STEP
    def _get_schema_context(self) -> str:
        """Generate schema context for LLM prompts"""
        node_context = "\n".join(
            f"Label: {n['label']}\nProperties: {', '.join(n['properties'])}"
            for n in self.schema['nodes']
        )
        rel_context = "\n".join(
            f"Type: {r['type']}\nFrom: {r['from']}\nTo: {r['to']}"
            for r in self.schema['relationships']
        )
        return f"""Graph Schema:
            Nodes:
            {node_context}

            Relationships:
            {rel_context}"""


    def extract_entities(self, question: str) -> Dict:
        """Extract entities with relationship direction detection"""
        schema_context = self._get_schema_context()
        prompt = f"""Analyze this question and extract:
            1. Node labels with properties
            2. Relationship types with direction
            3. Filters/conditions
            4. Return properties

            {schema_context}

            Question: {question}

            Respond with JSON format:
            {{
                "nodes": [
                    {{"label": "Label", "variable": "var", "properties": {{"property": "value"}}}},
                ],
                "relationships": [
                    {{"type": "TYPE", "direction": "→"|"←"|"↔", "from_var": "var", "to_var": "var"}},
                ],
                "conditions": [
                    {{"variable": "var", "property": "prop", "operator": "=", "value": "val"}},
                ],
                "returns": ["var.property", ...]
            }}"""

        response = self._llm_chat_completion(prompt)
        
        # Validate extracted entities against schema
        self._validate_extracted_entities(response)
        return response

    def _validate_extracted_entities(self, entities: Dict):
        """Validate extracted entities against schema"""
        valid_labels = {n["label"] for n in self.schema["nodes"]}
        valid_rels = {r["type"] for r in self.schema["relationships"]}

        for node in entities.get("nodes", []):
            if node["label"] not in valid_labels:
                raise ValueError(f"Invalid node label: {node['label']}")

        for rel in entities.get("relationships", []):
            if rel["type"] not in valid_rels:
                raise ValueError(f"Invalid relationship type: {rel['type']}")

    def generate_cypher(self, entities: Dict) -> str:
        """Generate complete Cypher query"""
        query_parts = ["MATCH"]
        
        # Build MATCH clauses
        for rel in entities["relationships"]:
            direction = {
                "→": "-[:{}]->",
                "←": "<-[:{}]-",
                "↔": "-[:{}]-"
            }[rel["direction"]]
            
            match_clause = f"({rel['from_var']}:{rel['from_label']}){direction.format(rel['type'])}({rel['to_var']}:{rel['to_label']})"
            query_parts.append(match_clause)

        # Add WHERE clauses
        if entities.get("conditions"):
            query_parts.append("WHERE")
            conditions = []
            for cond in entities["conditions"]:
                value = f"'{cond['value']}'" if isinstance(cond['value'], str) else cond['value']
                conditions.append(f"{cond['variable']}.{cond['property']} {cond['operator']} {value}")
            query_parts.append(" AND ".join(conditions))

        # Add RETURN clause
        return_clause = "RETURN " + ", ".join(entities["returns"])
        query_parts.append(return_clause)

        query = "\n".join(query_parts)
        
        # Add final validation
        self._validate_cypher(query)
        return query

    def _validate_cypher(self, query: str):
        """Validate query syntax and schema compliance"""
        # Basic syntax validation
        if not re.search(r'MATCH.*?RETURN', query, re.DOTALL | re.IGNORECASE):
            raise ValueError("Invalid Cypher structure")

        # Schema validation using Neo4j
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    result = session.run(f"EXPLAIN {query}")
                    # If we get here, the query is at least syntactically valid
            except Exception as e:
                raise ValueError(f"Invalid Cypher query: {str(e)}")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=5)
    )
    def full_pipeline(self, question: str) -> str:
        """End-to-end query generation with fallback"""
        try:
            entities = self.extract_entities(question)
            return self.generate_cypher(entities)
        except Exception as e:
            logger.warning(f"Pipeline failed: {str(e)}. Attempting fallback...")
            return self._fallback_generation(question)

    def _fallback_generation(self, question: str) -> str:
        """Fallback method for complex queries"""
        prompt = f"""Generate a Cypher query directly for this question.
Schema:
{self._get_schema_context()}

Question: {question}

Return only the Cypher query with no additional text."""
        
        response = self._llm_chat_completion(prompt)
        self._validate_cypher(response["query"])
        return response["query"]

# Example Usage
if __name__ == "__main__":
    SCHEMA = {
        "nodes": [
            {"label": "Person", "properties": ["name", "born"]},
            {"label": "Movie", "properties": ["title", "released"]}
        ],
        "relationships": [
            {"type": "ACTED_IN", "from": "Person", "to": "Movie"},
            {"type": "DIRECTED", "from": "Person", "to": "Movie"}
        ]
    }

    generator = CypherGenerator(
        schema=SCHEMA,
        openai_api_key="sk-your-key-here",
        neo4j_uri="bolt://localhost:7687",
        neo4j_auth=("neo4j", "password")
    )

    questions = [
        "Which movies did Tom Hanks act in?",
        "Find movies released after 2010 directed by Christopher Nolan",
        "Who directed The Matrix?"
    ]

    for q in questions:
        try:
            print(f"Question: {q}")
            cypher = generator.full_pipeline(q)
            print(f"Cypher:\n{cypher}\n")
        except Exception as e:
            print(f"Failed to generate query: {str(e)}")