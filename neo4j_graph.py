from neo4j import GraphDatabase, Query
from typing import Literal

class neo4jGraph:

    def __init__(self, URI, username, password):
        self.driver = GraphDatabase.driver(URI, 
                                           auth=(username, password),
                                           )
        self.username = username
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.driver.close()

    # def run_query(self, query, timeout=10,  convert_func: Literal['data', 'graph'] = 'data'):
    #     cypher = Query(query, timeout=timeout)
    #     result =  self.driver.execute_query(
    #         cypher,
    #         database_=self.username,
    #         )

    #     return result
    def run_query(self, cypher, timeout=None, convert_func: Literal['data', 'graph'] = 'data'):
        # if self.debug:
        #     t0 = time.time()
        #     print(f'Running Cypher:\n```\n{cypher}\n```')

        with self.driver.session(database=self.username) as session:
            query = Query(cypher, timeout=timeout)
            result = session.run(query)
            if convert_func == 'data':
                result = result.data()
            elif convert_func == 'graph':
                result = result.graph()
            else:
                raise ValueError(f"Invalid convert_func: {convert_func}")

        # if self.debug:
        #     print(f'Cypher finished in {time.time() - t0:.2f}s')
        return result
        
    def get_num_entities(self):
        return self.driver.execute_query(
            "MATCH (n) RETURN count(n) as num",
            database_= self.username,
            )[0][0]['num']

    def get_num_relations(self):
        return self.driver.execute_query(
            "MATCH ()-[r]->() RETURN count(r) as num",
            database_= self.username,
            )[0][0]['num']