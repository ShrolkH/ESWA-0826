from sklearn.metrics import roc_auc_score, average_precision_score,  precision_score, recall_score, f1_score
import os
import pandas as pd
import igraph as ig
import math
import dgl
from dgl.data.utils import save_graphs
import numpy as np
from rag.rag_model_py import rag_chat
from contextlib import redirect_stdout
from rag.self_chroma import Self_Chroma
import scipy.sparse as sp
import networkx as nx
import io
import json
import torch
from datetime import datetime
import pytz
import random
from graph_predict.graph_predict import LinkPredictor

def save_graph(graph, save_dir, filename="without_zeredegree_hetero_graph.bin"):
    """
    Save the heterograph to a local path.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    dgl.save_graphs(save_path, [graph])
    print(f"Graph saved to {save_path}")
    return 0

# Load dataset (corresponding timestamps: 2014-12-31 23:59:59 —— 1420041599, 2007-12-31 23:59:59 —— 1199116799)
def load_dataset(dataname="large",radio_edge=0):
    test_ratio = 0.01
    if dataname == "large":  
        graph_path = "/hy-tmp/cot-prompt-words/preprocessing/large_disease.bin"
        csv_path = "/hy-tmp/data/updated_disease_descriptions.csv"
        timestamp = 1420041599
        test_ratio = 0.0001
    elif dataname == "small":
        # Read GML graph
        graph_path = "/hy-tmp/cot-prompt-words/preprocessing/small_disease.bin"
        csv_path = "/hy-tmp/cot-prompt-words/preprocessing/small_node.csv"
        timestamp = 1199116799
    elif dataname == "KBS":
        graph_path = "/hy-tmp/llm_lp-main/data/kbs_data/without_zeredegree_hetero_graph.bin"
        csv_path = "/hy-tmp/llm_lp-main/data/kbs_data/without_zero_nodes.csv"
        timestamp = 1746108218
    g = load_graph(graph_path)
    node = pd.read_csv(csv_path)
    graph_predictor = LinkPredictor(dataname)
    print("Graph information")
    print(g)
    if radio_edge == 0:
        train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = split_edges(g,test_ratio)
    else:
        test_pos_g = None
        test_neg_g = None
        train_g, train_pos_g, train_neg_g = splite_edges_all(g)
    return graph_predictor,train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g, node, timestamp 



def split_edges(g, test_ratio=0.01):
    """Split edge data into training set and test set"""
    u, v = g.edges()
    eids = np.random.permutation(g.num_edges())
    test_size = int(len(eids) * test_ratio)


    # Positive sample split
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Negative sample generation
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), 
                       shape=(g.num_nodes(), g.num_nodes()))
    adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    
    neg_eids = np.random.choice(len(neg_u), g.num_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    # Create subgraphs
    train_g = dgl.remove_edges(g, eids[:test_size])
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.num_nodes())
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.num_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.num_nodes())

    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g

def splite_edges_all(g):
    u, v = g.edges()
    eids = np.random.permutation(g.num_edges())
    
    # Positive sample split
    train_pos_u, train_pos_v = u[eids], v[eids]

    # Negative sample generation
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), 
                       shape=(g.num_nodes(), g.num_nodes()))
    adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    
    neg_eids = np.random.choice(len(neg_u), g.num_edges())
    train_neg_u, train_neg_v = neg_u[neg_eids], neg_v[neg_eids]

    # Create subgraphs
    train_g = dgl.remove_edges(g, eids)
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.num_nodes())

    return train_g, train_pos_g, train_neg_g


# Build and save small DGL graph
def build_and_save_heterograph(save_path):
    # Read GML graph file
    G = ig.Graph.Read_GML('/hy-tmp/llm_lp-main/data/human-disease.gml')

    # Create node DataFrame
    node = pd.DataFrame({
        attribute: G.vs[attribute] for attribute in G.vertex_attributes()
    })[['name', 'label']]

    # Create edge DataFrame
    edges_df = pd.DataFrame({
        attribute: G.es[attribute] for attribute in G.edge_attributes()
    })

    # Add source node and target node information
    edges_df['src'] = [G.vs[edge.source]['name'] if 'name' in G.vertex_attributes() else edge.source for edge in G.es]
    edges_df['dsc'] = [G.vs[edge.target]['name'] if 'name' in G.vertex_attributes() else edge.target for edge in G.es]

    all_nodes = np.union1d(edges_df['src'].values, edges_df['dsc'].values)
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(all_nodes)}
    edges_df['src'] = edges_df['src'].map(node_mapping)
    edges_df['dsc'] = edges_df['dsc'].map(node_mapping)

    # Create DGL graph
    g = dgl.graph((edges_df['src'].values, edges_df['dsc'].values), num_nodes=len(all_nodes))

    # Save graph data
    os.makedirs(save_path, exist_ok=True)
    save_graphs(os.path.join(save_path, 'small_disease.bin'), [g])

    print(f"Graph saved to {save_path}/small_disease.bin")
    return 0

def save_error_response(text_response,pos):
    # Save erroneous responses to file
    error_file_path = "/hy-tmp/llm_lp-main/rag/result/error_json_change.json"
    os.makedirs(os.path.dirname(error_file_path), exist_ok=True)
    
    # Add timestamp and error data
    current_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S")
    error_data = {
        "timestamp": current_time,
        "error_response": text_response,
        "pos": pos
    }
    
    # If the file exists, read existing content first
    existing_data = []
    if os.path.exists(error_file_path) and os.path.getsize(error_file_path) > 0:
        try:
            with open(error_file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except json.JSONDecodeError:
            # If file format is incorrect, create a new list
            existing_data = []
            
    # Add new error data
    existing_data.append(error_data)
    
    # Save complete data back to file
    with open(error_file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
    return 0

# Load heterogeneous graph
def load_graph(load_path):
    """
    Load heterograph from local file.
    """
    graphs, _ = dgl.load_graphs(load_path)
    graph = graphs[0]
    return graph

def compute_metrics(pos_test_scores, neg_test_scores, threshold=0.5):
    """
    Calculate AUC, Average Precision Score, F1 Score, Precision, and Recall, and save the results to a dictionary.

    Parameters:
    - pos_test_scores (array-like): Predicted scores for positive class.
    - neg_test_scores (array-like): Predicted scores for negative class.
    - threshold (float): Threshold used to classify scores as positive class, default is 0.5.

    Returns:
    - result (dict): Dictionary containing various metrics.
    """
    result = {}

    # Create StringIO object to capture output
    f = io.StringIO()
    with redirect_stdout(f):
        # Calculate AUC
        test_auc = roc_auc_score(
            np.concatenate([np.ones(len(pos_test_scores)), np.zeros(len(neg_test_scores))]),
            np.concatenate([pos_test_scores, neg_test_scores])
        )
        print("Test AUC:", test_auc)

        # Calculate Average Precision Score
        ap_score = average_precision_score(
            np.concatenate([np.ones(len(pos_test_scores)), np.zeros(len(neg_test_scores))]),
            np.concatenate([pos_test_scores, neg_test_scores])
        )
        print("Average Precision Score:", ap_score)

        # Combine all scores and labels
        scores = np.concatenate([pos_test_scores, neg_test_scores])
        labels = np.concatenate([np.ones(len(pos_test_scores)), np.zeros(len(neg_test_scores))])  # True labels

        # Convert scores to binary predictions based on threshold
        predictions = (scores >= threshold).astype(int)

        # Calculate F1 Score
        f1 = f1_score(labels, predictions)
        print("F1 Score:", f1)

        # Calculate Precision
        precision = precision_score(labels, predictions)
        print("Precision:", precision)

        # Calculate Recall
        recall = recall_score(labels, predictions)
        print("Recall:", recall)

    # Get captured output and parse into dictionary
    output = f.getvalue().strip().split("\n")
    for line in output:
        key, value = line.split(": ")
        result[key.strip()] = float(value.strip())

    return result

def query_gpt(config,chroma,query,graph_score):
    response = rag_chat(config,chroma,query,graph_score)
    text_response = response.strip()
    # print("Large model output return")
    # print(text_response)
    try:
        response_data = json.loads(text_response)
        if "1" in response_data['result']:
            return 1
        elif "0" in response_data['result']:
            return 0
    except json.JSONDecodeError:
        print("thinking_response format error:", text_response)
        if '"result": "1"' or '"result": "exists"' in text_response:
            print("Value recorded as 1")
            return 1
        print("Value recorded as 0")
        return save_error_response(text_response,config['pos'])

    # Default or error handling case
    print(f"Unexpected response: {text_response}")
    return None  # Return None or consider raising an exception or logging an errorr

# Get other nodes connected to the source node and target node respectively
def get_reference_nodes(graph,src_id,dst_id):

    # Get all target nodes connected to the source node
    src_reference_nodes = graph.successors(src_id)
    dst_reference_nodes = graph.successors(dst_id)

    src_list = src_reference_nodes.tolist()
    dst_list = dst_reference_nodes.tolist()
    src_list = src_list.remove(dst_id)
    dst_list = dst_list.remove(src_id)
    return 0

# Get global information of the graph
def get_graph_info(graph):
    # print("Global information when extracting graph information")
    # print(graph)
    # 1. Convert DGL graph to NetworkX (multigraph)
    nx_graph = dgl.to_networkx(graph, node_attrs=None, edge_attrs=None)

    # 2. Convert to undirected graph
    nx_graph = nx_graph.to_undirected()

    # 3. Simplify to simple graph (remove multiple edges)
    simple_graph = nx.Graph(nx_graph)

    # 4. Calculate degree list
    degrees = np.array([deg for _, deg in simple_graph.degree()])

    # 5. Average degree
    average_degree = degrees.mean()

    # 6. Global average clustering coefficient
    avg_clustering = nx.average_clustering(simple_graph)

    # 7. Degree assortativity coefficient
    degree_assortativity = nx.degree_assortativity_coefficient(simple_graph)

    # 8. Graph density
    graph_density = nx.density(simple_graph)

    # 9. Median degree
    median_degree = np.median(degrees)

    return {
        "average_degree": f"{average_degree:.4f}",
        "avg_clustering": f"{avg_clustering:.4f}",
        "degree_assortativity": f"{degree_assortativity:.4f}",
        "graph_density": f"{graph_density:.4f}",
        "median_degree": f"{median_degree:.0f}"
    }


# Generate node description
def generate_description(config,graph, src, dst, df):
#     output_templates = {
#     'zero_shot': "Is there a potential relationship between {src_name} and {dst_name}? {src_name} is {src_dec}, {dst_name} is {dst_dec}. ",
#     'few_shot': ("For example: There are links in the following diseases: Node ID: {train_src_id}, Disease: {train_src} has relationship with Node ID: {train_dst_id}, Disease: {train_dst}. "
#                  "Is there a potential relationship between {src_name} and {dst_name}? {src_name} is {src_dec}, {dst_name} is {dst_dec}. Evaluate and respond with '1' for a strong link and '0' for a weak or no link. Do Not provide your reasoning."),
#     'chain_of_thought': ("Step-by-step, analyze whether there is a potential relationship between {src_name} and {dst_name}. "
#                          "That indicates one might lead to or be associated with the other. "
#                          "Evaluate and respond with '1' for a strong link and '0' for a weak or no link. Do Not provide your reasoning."),
#     'full_graph_analysis': ("In a disease network, Disease {src_name} has {src_degree} connections, Disease {dst_name} has {dst_degree} connections. "
#                             "They share {common_count} common diseases.The names of the common neighbors are {common_neighbors} Is there a potential relationship between {src_name} and {dst_name}? "
#                             "That indicates one might lead to or be associated with the other. ")
# }
    # Fetching node information
    # print("Error output check")
    # print(df)
    src_name = df[df['Id'] == src]['Disease Term'].values[0]
    src_dec = df[df['Id'] == src]['Description'].values[0]
    dst_name = df[df['Id'] == dst]['Disease Term'].values[0]
    dst_dec = df[df['Id'] == dst]['Description'].values[0]

    dst_nodes_info = ''

    # print("Current graph is")
    # print(graph)
    # if output_type=="semantic_graph":
    # # Get all node IDs connected to the source node and remove the target node ID
    #     dst_nodes = graph.successors(src).tolist()
    #     # print(f'Source node ID: {src}, Target node ID: {dst}, All nodes connected to the source node: {dst_nodes}')
    #     if dst in dst_nodes:
    #         dst_nodes.remove(dst)
    #     if len(dst_nodes)>0:  
    #         for item in dst_nodes:
    #             relate_name = df[df['Id'] == item]['Disease Term'].values[0]
    #             relate_dec = df[df['Id'] == item]['Description'].values[0]
    #             dst_nodes_info += f'Disease Name: {relate_name}, It is defined as {relate_dec}\n'
            # print(f'Related diseases: {dst_nodes_info}')
    # return 0
    # Calculating graph metrics
    successors_src = graph.successors(src).tolist()
    successors_dst = graph.successors(dst).tolist()
    common_neighbors = set(successors_src).intersection(set(successors_dst))
    common_neighbors_names = df[df['Id'].isin(common_neighbors)]['Disease Term'].tolist()
    common_neighbors_decs = df[df['Id'].isin(common_neighbors)]['Description'].tolist()
    common_neighbors_names_text = ""  
    for i in range(min(10, len(common_neighbors_names))):
        common_neighbors_names_text += f'Node "{common_neighbors_names[i]}" is defined as "{common_neighbors_decs[i]}" '

    common_count = len(common_neighbors_names)

    src_degree = graph.out_degrees(src) + graph.in_degrees(src)
    dst_degree = graph.out_degrees(dst) + graph.in_degrees(dst)

    # Jaccard's Coefficient
    jaccard_score = common_count / (src_degree + dst_degree - common_count)

    # Adamic-Adar Index
    adamic_adar_score = 0

    # Resource Allocation Index
    resource_score = 0
    for item in common_neighbors:
        item_degree = graph.out_degrees(item) + graph.in_degrees(item)
        resource_score += 1 / item_degree
        adamic_adar_score += 1 / math.log(item_degree)
    # Preferential Attachment
    preferential_attachment = src_degree * dst_degree



    query = {
        "src_name":src_name,
        "src_dec":src_dec,
        "dst_name":dst_name,
        "dst_dec":dst_dec,
        "dst_nodes_info":dst_nodes_info,
        "common_neighbors_names_text":common_neighbors_names_text+"\n",
        "common_count":common_count,
        "src_degree":src_degree,
        "dst_degree":dst_degree,
        "relation_score":{
            "jaccard_score":jaccard_score,
            "adamic_adar_score":adamic_adar_score,
            "resource_score":resource_score,
            "preferential_attachment":preferential_attachment
        }
    }

    # # Determine if path exists
    # try:
    #     shortest_path = dgl.shortest_dist(graph, src, dst)
    #     path_exists = "Yes"
    # except:
    #     shortest_path = 'N/A'
    #     path_exists = "No"
    
    # Selecting the appropriate template
    # template = output_templates[output_type]
    # query = template.format(
    #     src_name=src_name, src_dec=src_dec, dst_name=dst_name, dst_dec=dst_dec,
    #     common_count=len(common_neighbors), common_neighbors=', '.join(common_neighbors_names),
    #     src_degree=src_degree, dst_degree=dst_degree, path_exists=path_exists, shortest_path=shortest_path
    # )
    return query

# Print the first 10 positive samples (edges that actually exist)
def print_first_ten_edges_with_disease(graph, graph_name, df):
    # Extract source and destination node IDs
    src, dst = graph.edges()

    # Convert to lists for easier handling
    src_list = src.tolist()
    dst_list = dst.tolist()

    # Print the first ten edges with disease terms
    print(f"There are links in the following diseases:")
    for i in range(min(10, len(src_list))):  # Ensure we do not go out of bounds
        src_id = src_list[i]
        dst_id = dst_list[i]
        src_name = df[df['Id'] == src_id]['Disease Term'].values[0]  # Lookup source disease term
        dst_name = df[df['Id'] == dst_id]['Disease Term'].values[0]  # Lookup destination disease term
        print(f"Node ID: {src_id}, Disease: {src_name} has relationship with Node ID: {dst_id}, Disease: {dst_name}")
    return 0

# Use code to determine if there is an association between the four indicators
def judge_is_connect(jc_value,aa_value,ra_value,pa_value,query):
    score = query['relation_score']["jaccard_score"]
    if (score > jc_value[1]) or (score >= jc_value[2] and score <= jc_value[3]):
        return 1
    score = query['relation_score']["adamic_adar_score"]
    if (score > aa_value[1]) or (score >= aa_value[2] and score <= aa_value[3]):
        return 1
    score = query['relation_score']["resource_score"]
    if (score > ra_value[1]) or (score >= ra_value[2] and score <= ra_value[3]):
        return 1
    score = query['relation_score']["preferential_attachment"]
    if (score > pa_value[1]) or (score >= pa_value[2] and score <= pa_value[3]):
        return 1
    return 0

# Print the first 10 negative samples (edges that do not exist in the data)
def print_first_ten_neg_edges_with_disease(graph, graph_name, df):
    # Extract source and destination node IDs
    src, dst = graph.edges()

    # Convert to lists for easier handling
    src_list = src.tolist()
    dst_list = dst.tolist()

    # Print the first ten edges with disease terms
    print("There are no links in the following diseases:")
    for i in range(min(10, len(src_list))):  # Ensure we do not go out of bounds
        src_id = src_list[i]
        dst_id = dst_list[i]
        src_name = df[df['Id'] == src_id]['Disease Term'].values[0]  # Lookup source disease term
        dst_name = df[df['Id'] == dst_id]['Disease Term'].values[0]  # Lookup destination disease term
        print(f"Node ID: {src_id}, Disease: {src_name} has no relationship with Node ID: {dst_id}, Disease: {dst_name}")
    return 0

# Get the list of split edge nodes
def get_src_dst_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        src_list = data['src']
        dst_list = data['dst']
    return src_list,dst_list


def compute_scores(graph_predictor,config,chroma,timestamp,graph, df):
    """Compute scores and print predictions for all edges in a graph, ensuring binary responses."""
    src, dst = graph.edges()
    scores = []
    predictions = []  # List to store predictions
    print("Information of the currently processed graph:")
    print(graph)
    # format_text = "## Basic information about the graph in which the node is located: The Jaccard Coefficient exhibits a global mean of {jc_mean}, median of {jc_median}, and interquartile range [{jc_q25}, {jc_q75}]; the Adamic–Adar Index shows a mean of {aa_mean}, median of {aa_median}, and IQR [{aa_q25}, {aa_q75}]; the Resource Allocation Index has a mean of {ra_mean}, median of {ra_median}, and IQR [{ra_q25}, {ra_q75}]; and the Preferential Attachment score’s mean is {pa_mean}, median {pa_median}, with IQR [{pa_q25}, {pa_q75}]. These summary statistics define each metric’s typical level and variability in the network, enabling a clear global benchmark for assessing the target pair’s structural similarity."
    format_text = ""
    if config['dataset_name'] == 'small':
        # Array values are [mean, median, q25, q75]
        jc_value = [0.310184,0.266667,0.111111,0.500000]
        aa_value = [1.764156,1.753363,0.910239,2.485340]
        ra_value = [0.479271,0.518929,0.250000,0.671429]
        pa_value = [85.744949,36.000000,16.000000,99.000000]
    #     graph_info_text = '## Basic information about the graph in which the node is located: Average degree is 4.6047; Average clustering is 0.6358; Degree assortativity is 0.0666; Graph density is 0.0089; Median degree is 4;\n'
    elif config['dataset_name'] == 'large':
        jc_value = [0.253537,0.248132,0.164706,0.335938]
        aa_value = [24.346344,20.905256,12.101393,32.929743]
        ra_value = [0.482226,0.376865,0.212909,0.630406]
        pa_value = [108595.278440,84591.0,42717.25,148410.0]
    graph_info_text = format_text.format(
        jc_mean = jc_value[0],
        jc_median = jc_value[1],
        jc_q25 = jc_value[2],
        jc_q75 = jc_value[3],
        aa_mean = aa_value[0],
        aa_median = aa_value[1],
        aa_q25 = aa_value[2],
        aa_q75 = aa_value[3],
        ra_mean = ra_value[0],
        ra_median = ra_value[1],
        ra_q25 = ra_value[2],
        ra_q75 = ra_value[3],
        pa_mean = pa_value[0],
        pa_median = pa_value[1],
        pa_q25 = pa_value[2],
        pa_q75 = pa_value[3]
    )
    if config['radio_edge'] != 0:
        src_list = src.tolist()
        dst_list = dst.tolist()
        total_edges = len(src_list)
        print(f"Total number of edges is {total_edges}")
        sample_size = int(total_edges * config['radio_edge'] / 100)
        print(f"Number of sampled edges is {sample_size}")
        perm = torch.randperm(total_edges)
        sample_indices = perm[:sample_size]

        for i in sample_indices:
            s = src_list[i]
            d = dst_list[i]
            query = generate_description(config,graph, s, d, df)
            query['graph_info_text'] = graph_info_text
            prob, graph_score = graph_predictor.predict_edge(s,d)
            score = query_gpt(config,chroma,query,graph_score)
            query["src_id"] = s
            query["dst_id"] = d
            if score is not None and score in [0, 1]:  # Ensure score is strictly binary
                scores.append(score)
                prediction_text = f"Prediction: {score}"
            else:
                prediction_text = f"Prediction: Error - non-binary response received"
            predictions.append(prediction_text)
            print(prediction_text) 
    else:
        for s, d in zip(src.tolist(), dst.tolist()):
            query = generate_description(config,graph, s, d, df)
            query['graph_info_text'] = graph_info_text
            score = query_gpt(config,chroma,query)
            if score is not None and score in [0, 1]:  # Ensure score is strictly binary
                scores.append(score)
                prediction_text = f"Prediction: {score}"
            else:
                prediction_text = f"Prediction: Error - non-binary response received"

            predictions.append(prediction_text)
            print(prediction_text)  # Print each prediction
    if config['pos']:
        print("Predictions on Positive Samples:")
    else:
        print("Predictions on Negative Samples:")
    for prediction in predictions:
        print(prediction)

    return np.array(scores), predictions  # Return scores and predictions

# Store experimental results
def storage_input_and_output(file_path,data):

    # Ensure the folder exists; create it if it does not
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # If the file exists, first read the existing data
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
                # Convert to list if existing data is not a list
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # Add the new data to the existing data
    existing_data.append(data)

    # Save the updated data back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print(f"Data has been successfully saved to {file_path}")
    return 0


def main(config,chroma):
    graph_predictor,train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g, node, timestamp= load_dataset(config['dataset_name'],radio_edge=config['radio_edge'])
    print_first_ten_edges_with_disease(train_pos_g, "Positive Training Graph", node)
    print_first_ten_neg_edges_with_disease(train_neg_g, "NEGATIVE Training Graph", node)
    if config['is_compare_think']:
        for config['is_add_think'] in [False]:
    # Example usage
            if config['is_train']:
                config['pos'] = True
                pos_test_scores, pos_predictions = compute_scores(graph_predictor,config,chroma,timestamp,train_pos_g, node)
                config['pos'] = False
                neg_test_scores, neg_predictions = compute_scores(graph_predictor,config,chroma,timestamp,train_neg_g, node)
            else:
                config['pos'] = True
                pos_test_scores, pos_predictions = compute_scores(graph_predictor,config,chroma,timestamp,test_pos_g, node)
                config['pos'] = False
                neg_test_scores, neg_predictions = compute_scores(graph_predictor,config,chroma,timestamp,test_neg_g, node)

            metrics_result = compute_metrics(pos_test_scores,neg_test_scores)

            print(metrics_result)

            # print(metrics_result)
            metrics_result['data_name'] = config['dataset_name']
            metrics_result['model_name'] = config['model_name']
            metrics_result['prompt_templete'] = config['prompt_templete']
            metrics_result['is_RAG'] = config['is_RAG']
            metrics_result['is_add_think'] = config['is_add_think']
            metrics_result['is_compare_think'] = config['is_compare_think']
            metrics_result['is_train'] = config['is_train']
            storage_input_and_output(config['result_file_path'],metrics_result)
    return 0

# Continue processing based on already processed data
def get_dealed_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# Data executed on the evening of 5-27: 20250526-163736.json
if __name__ == "__main__":
    # Get current date and time in the format YYYYMMDD-HHMMSS
    # Set timezone to UTC+8 (Asia/Shanghai)
    tz = pytz.timezone('Asia/Shanghai')
    # Get current time in UTC+8 timezone
    current_datetime = datetime.now(tz).strftime("%Y%m%d-%H%M%S")
    # Construct file save path
    result_file_path = f"/hy-tmp/llm_lp-main/rag/result/exp_rusult{current_datetime}.json"
    
    model_list = ['qwen-plus-latest']
    config = {
        "is_compare_think":True,
        "is_RAG":True,
        "is_add_think":False,
        "is_train":True,
        "is_few_shot":False,
        "prompt_templete":'full_graph_analysis',
        "dataset_name":'large',
        "model_name":'qwen-plus-latest',
        "result_file_path":result_file_path, # Value range: 1-100 (percentage)
        "think_score": 8.5,
        "current_date" : current_datetime
    }
    self_chroma = Self_Chroma()
    config["dataset_name"] = 'large'
    for config["dataset_name"] in ['large']:
        print(f'Current dataset is {config["dataset_name"]}')
        if config["dataset_name"] == 'large':
            config["last_time"] = "2014/12/31"
            config["radio_edge"] = 1
            config["timestamp_threshold"] = 1420041599
        else:
            config["last_time"] = "2007/12/31"
            config["radio_edge"] = 10
            config["timestamp_threshold"] = 1199116799
        main(config,self_chroma)
    print("Execution completed")