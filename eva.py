import argparse
import copy
import json
import os
import math
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from metrics import *
from neo4j_connector import Neo4jConnector
from neo4j_graph import neo4jGraph
from neo4j import GraphDatabase


RETURN_PATTERN_MAPPING = {
    "n_name": "n_name",
    "n_prop": "n_prop_combined",
    "n_name_prop": "n_prop_combined",
    "n_prop_distinct": "n_prop_combined",
    "n_prop_array_distinct": "n_prop_combined",
    "n_order_by": "n_order_by",
    "n_argmax": "n_argmax",
    "n_where": "n_where",
    "n_agg": "n_agg",
    "n_group_by": "n_group_by"
}

METRIC_FUNC_MAPPING = {
    'execution_accuracy': execution_accuracy,
    'psjs': provenance_subgraph_jaccard_similarity,
    'executable': executable,
    'google_bleu': google_BLEU,
}

# item: csv row
def compute_metrics(item, metrics, driver):
    item = copy.deepcopy(item)
    for m in metrics:
        pred_cypher = item['generated_cypher']
        ref_cypher = item['cypher']
        if pred_cypher.endswith('<end_of_turn>'):
            pred_cypher = pred_cypher[:-len('<end_of_turn>')].strip()
        if m == 'google_bleu':
            result = METRIC_FUNC_MAPPING[m](
                pred_cypher=pred_cypher,
                target_cypher=ref_cypher,
            )
        else:
            result = METRIC_FUNC_MAPPING[m](
                pred_cypher=pred_cypher,
                target_cypher=ref_cypher,
                neo4j_connector=driver,
            )
        item[m] = result
    return item


def avg_and_round(nums: list[float], n: int = 4):
    return round(sum(nums) / len(nums), n) if nums else math.nan


def aggregate(results: list[tuple[str, float]]):
    res = {}
    for key, value in results:
        if key not in res:
            res[key] = []
        res[key].append(value)
    for key, values in res.items():
        res[key] = avg_and_round(values)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_threads', type=int, default=8)
    # parser.add_argument('--metrics', nargs='+', default=['executable', 'execution_accuracy', 'psjs'])
    parser.add_argument('--metrics', nargs='+', default=['executable', 'execution_accuracy'])
    parser.add_argument('--metric_for_agg', default='execution_accuracy')
    parser.add_argument('--URI', default='neo4j+s://demo.neo4jlabs.com:7687')


    args = parser.parse_args()
    print(args)
    print()
    # open csv file
    absolute_path = '/work/pi_wenlongzhao_umass_edu/9/'
    NEO4J_DATASET = os.path.join(absolute_path, 'bloomberg_project/generate_cypher.csv')

    print("CSV Path:", NEO4J_DATASET)
    if os.path.exists(NEO4J_DATASET):
        df = pd.read_csv(NEO4J_DATASET)
        df = df.sort_values(by='database_reference_alias', ascending=True)
    else:
        print(f"Error: File not found - {NEO4J_DATASET}")

    df = df.sort_values(by='database_reference_alias', ascending=True)
    unique_graphs = df['database_reference_alias'].unique()

    result_dfs = []
    for grpah in unique_graphs:
        print(f"Processing grpah: {grpah}")
        df_category = df[df['database_reference_alias'] == grpah].copy()
        graph_name = grpah.replace("neo4jlabs_demo_db_", "")
        # AUTH = (graph_name, graph_name)
        try:
            with neo4jGraph(args.URI, graph_name, graph_name) as connector:
                connector.driver.verify_connectivity()
                print(f"Connection established in {graph_name} graph.")
                        # Use ThreadPoolExecutor for multithreading
            results = []
            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # with ThreadPoolExecutor(max_workers=1) as executor:
                futures = [executor.submit(compute_metrics, row, args.metrics, connector) for _, row in df_category.iterrows()]
                for future in tqdm(as_completed(futures), total=len(df_category)):
                    results.append(future.result())
            df_results = pd.DataFrame(results)
            result_dfs.append(df_results)

        except Exception as e:
            print(f"Fail to connect to {graph_name} graph: {e}")
            continue
    if result_dfs:
        final_df = pd.concat(result_dfs, ignore_index=True)
        final_df.to_csv(os.path.join(absolute_path, 'zek/results.csv'), index=False)
        print("Results saved to results.csv")
        # Convert results back to a DataFrame and update 'age' column
        # df_category = pd.DataFrame(results)
        # result_dfs.append(df_category)

    # Combine all processed DataFrames
    # df_final = pd.concat(result_dfs, ignore_index=True)

    # # Save the modified DataFrame back to a CSV
    # output_path = os.path.join(absolute_path, 'zek/generate_cypher_with_age.csv')
    # df_final.to_csv(output_path, index=False)

    # aggregated = {}
    # aggregated['overall'] = {m: avg_and_round([item.metrics[m] for item in result_with_metrics]) for m in args.metrics}

    # metric_for_agg = args.metric_for_agg
    # aggregated['by_graph'] = aggregate([(item.graph, item.metrics[metric_for_agg]) for item in result_with_metrics])
    # aggregated['by_match'] = aggregate([(item.from_template.match_category, item.metrics[metric_for_agg])
    #                                     for item in result_with_metrics])
    # aggregated['by_return'] = aggregate(
    #     [(RETURN_PATTERN_MAPPING[item.from_template.return_pattern_id], item.metrics[metric_for_agg])
    #      for item in result_with_metrics if item.from_template.return_pattern_id in RETURN_PATTERN_MAPPING]
    # )

    # output_path = os.path.join(args.result_dir, f'result_with_metrics.json')
    # with open(output_path, 'w') as fout:
    #     json.dump([item.model_dump(mode='json') for item in result_with_metrics], fout, indent=2)
    # print(f'Saved result with metrics to {output_path}')

    # output_path = os.path.join(args.result_dir, f'aggregated_metrics.json')
    # with open(output_path, 'w') as fout:
    #     json.dump(aggregated, fout, indent=2)
    # print(f'Saved aggregated metrics to {output_path}')

    # print()
    # print('Aggregated metrics:')
    # print(json.dumps(aggregated, indent=2))


if __name__ == '__main__':
    main()
