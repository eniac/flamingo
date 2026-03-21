from Cryptodome.Cipher import AES, ChaCha20
from Cryptodome.Random import get_random_bytes
import numpy as np
import pandas as pd
import math

# System parameters
vector_len = 16000
vector_type = 'uint32'
committee_size = 60
fraction = 1/3
fixed_key = b"abcd"

# Waiting time
# Set according to a target dropout rate (e.g., 1%) 
# and message lantecy (see model/LatencyModel.py)
wt_flamingo_report = pd.Timedelta('10s')
wt_flamingo_crosscheck = pd.Timedelta('3s')
wt_flamingo_reconstruction = pd.Timedelta('3s')

wt_google_adkey = pd.Timedelta('10s')
wt_google_graph = pd.Timedelta('10s')
wt_google_share = pd.Timedelta('30s')    # ensure all user_choice received messages
wt_google_collection = pd.Timedelta('10s')
wt_google_crosscheck = pd.Timedelta('3s')
wt_google_reconstruction = pd.Timedelta('2s') 

# WARNING: 
# this should be a random seed from beacon service;
# we use a fixed one for simplicity
root_seed = get_random_bytes(32) 
nonce = b'\x00\x00\x00\x00\x00\x00\x00\x00'

def assert_power_of_two(x):
    return (math.ceil(math.log2(x)) == math.floor(math.log2(x)))
    
# choose committee members
def choose_committee(root_seed, committee_size, num_clients):
    
    prg_committee_holder = ChaCha20.new(key=root_seed, nonce=nonce)

    data = b"secr" * committee_size * 128 
    prg_committee_bytes = prg_committee_holder.encrypt(data)
    committee_numbers = np.frombuffer(prg_committee_bytes, dtype=vector_type)
        
    user_committee = set()
    cnt = 0
    while (len(user_committee) < committee_size):
        sampled_id = committee_numbers[cnt] % num_clients
        (user_committee).add(sampled_id)
        cnt += 1

    return user_committee

# Cache for the full neighbor graph: {iteration -> {client_id -> set(neighbors)}}
# In a real deployment, each client computes its own neighbors locally.
# In the simulation, we compute the full graph once and let each client look up its entry.
_graph_cache = {}

def compute_full_graph(root_seed, current_iteration, num_clients, neighborhood_size):
    """Compute the full neighbor graph for all clients in one pass.

    Returns a dict: {client_id: set(neighbor_ids)}.
    """
    cache_key = (current_iteration, num_clients, neighborhood_size)
    if cache_key in _graph_cache:
        return _graph_cache[cache_key]

    # compute PRF(root, iter_num), output a seed
    prf = ChaCha20.new(key=root_seed, nonce=nonce)
    current_seed = prf.encrypt(current_iteration.to_bytes(32, 'big'))

    # compute PRG(seed), a binary string
    prg = ChaCha20.new(key=current_seed, nonce=nonce)

    # compute number of bytes we need for a graph
    num_choose = math.ceil(math.log2(num_clients)) * neighborhood_size
    bytes_per_client = math.ceil(math.log2(num_clients) / 8)
    bits_per_client = math.ceil(math.log2(num_clients))
    segment_len = num_choose * bytes_per_client
    num_rand_bytes = segment_len * num_clients
    graph_string = prg.encrypt(b"a" * num_rand_bytes)

    # Parse all segments into forward-edge lists
    forward_edges = {}  # client_id -> set of chosen neighbors
    for cid in range(num_clients):
        seg = graph_string[cid * segment_len: (cid + 1) * segment_len]
        chosen = set()
        for i in range(num_choose):
            cur_bytes = seg[i * bytes_per_client: (i + 1) * bytes_per_client]
            neighbor = int.from_bytes(cur_bytes, 'big') & ((1 << bits_per_client) - 1)
            if neighbor != cid:
                chosen.add(neighbor)
        forward_edges[cid] = chosen

    # Build the full bidirectional neighbor graph
    full_graph = {cid: set(forward_edges[cid]) for cid in range(num_clients)}
    for cid in range(num_clients):
        for neighbor in forward_edges[cid]:
            full_graph[neighbor].add(cid)  # reverse edge

    _graph_cache[cache_key] = full_graph
    return full_graph


def findNeighbors(root_seed, current_iteration, num_clients, id, neighborhood_size):
    """Look up neighbors for a single client from the cached full graph."""
    full_graph = compute_full_graph(root_seed, current_iteration, num_clients, neighborhood_size)
    return full_graph[id]


def parse_segment_to_list(segment, num_choose, bits_per_client, bytes_per_client):
    cur_ls = set()
    for i in range(num_choose):
        cur_bytes = segment[i * bytes_per_client: (i + 1) * bytes_per_client]
        cur_no = int.from_bytes(cur_bytes, 'big') & ((1 << bits_per_client) - 1)
        cur_ls.add(cur_no)
    return cur_ls