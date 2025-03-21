import neo4j
from neo4j_connector import Neo4jConnector
from neo4j import Query, GraphDatabase
import evaluate
from neo4j_graph import neo4jGraph
from utils import convert_dict_to_str

def exact_match(pred_cypher: str,
               target_cypher: str,
               neo4j_connector: Neo4jConnector,
               timeout: int = 120) -> float:
    """Whether the predicted Cypher query is executable"""
    if pred_cypher == target_cypher:
        return 1.0
    try:
        generated_result = neo4j_connector.run_query(pred_cypher, timeout=timeout, convert_to_str=True)


        # with GraphDatabase.driver(neo4j_connector.URI, 
        #                                 auth=(neo4j_connector.username, neo4j_connector.username),
        #                                 connection_timeout=120,
        #                                 notifications_min_severity='OFF',  # or 'WARNING' to enable entirely
        #                                 # notifications_disabled_classifications=['HINT', 'GENERIC'],
        #                                 ) as con:
        #     con.verify_connectivity()
        # generated_result = con.execute_query(
        #     Query(pred_cypher, timeout=timeout),
        #     database_=neo4j_connector.username,
        #     )
        # generated_result = convert_dict_to_str(generated_result)
    except (
            neo4j.exceptions.CypherSyntaxError,
            neo4j.exceptions.DatabaseError,
            neo4j.exceptions.CypherTypeError,
            neo4j.exceptions.ClientError,
    ) as e:
        return 0.0
    except Exception as e:
        print(f"Warning: Exception {e} occurred while executing the predicted Cypher query {pred_cypher}")
        return 0.0
    # con.close()
    target_result = neo4j_connector.run_query(target_cypher, timeout=timeout, convert_to_str=True)
    exact_match = evaluate.load("exact_match")
    try: 
        output = exact_match.compute(predictions=[generated_result], references=[target_result])
    except Exception as e:
        return 0.0
    return output['exact_match']
    # if generated_result == target_result:
    #     return 1.0
    # else:
    #     return 0.0