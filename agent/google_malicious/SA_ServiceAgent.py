# malicious version
from multiprocessing.sharedctypes import Value

from matplotlib.pyplot import axis
from scipy.fftpack import idstn
from util.benchmark import bench
from agent.Agent import Agent
from message.Message import Message

import logging
import time
import dill

import math
import nacl.secret
import nacl.utils

import numpy as np
import pandas as pd
import random

from util import param
from util.crypto import ecchash
from util.crypto.secretsharing import secret_int_to_points, points_to_secret_int

from Cryptodome.PublicKey import ECC
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes


# Secret sharing for each client and the server
# import Crypto.Protocol.SecretSharing as shamir

# NEW MESSAGES: CLIENT_WEIGHTS : weights, SHARED_WEIGHTS : weights

# The PPFL_ServiceAgent class inherits from the base Agent class.  It provides
# the simple shared service necessary for model combination under secure
# federated learning.


class SA_ServiceAgent(Agent):

    def __init__(self, id, name, type,
                 random_state=None,
                 msg_fwd_delay=1000000,
                 round_time=pd.Timedelta("10s"),
                 field_size=-1,
                 iterations=4,
                 num_clients=10,
                 num_neighbors=-1,
                 neighbor_threshold=-1,
                 users={},
                 debug_mode=0,
                 max_input=10000,
                 wait_mode='fixed',
                 wait_threshold=0.9,
                 vector_len=None):

        # Base class init.
        super().__init__(id, name, type, random_state)

        self.logger = logging.getLogger("Log")
        self.logger.setLevel(logging.INFO)
        if debug_mode:
            logging.basicConfig()

        # Set parameters
        self.num_clients = num_clients
        self.vector_len = vector_len if vector_len else param.vector_len
        self.vector_dtype = 'uint32'
        self.vec_sum_partial = np.zeros(self.vector_len, dtype=self.vector_dtype)
        self.prime = ecchash.n

        self.system_sk = None
        hdr = 'pki_files/system_pk.pem'
        f = open(hdr, "rt")
        key = ECC.import_key(f.read())
        f.close()
        self.system_sk = key.d

        # How long does it take us to forward a peer-to-peer client relay message?
        self.msg_fwd_delay = msg_fwd_delay
        self.round_time = round_time

        # Agent accumulation of elapsed times by category of task.
        self.elapsed_time = {'ADKEY': pd.Timedelta(0),
                             'GRAPH': pd.Timedelta(0),
                             'SHARE': pd.Timedelta(0),
                             'COLLECTION': pd.Timedelta(0),
                             'CROSSCHECK': pd.Timedelta(0),
                             'RECONSTRUCTION': pd.Timedelta(0),
                             }

        # How many iterations of the protocol should be run?
        self.no_of_iterations = iterations


        # The list of all users id
        self.users = users 

        self.neighbors = {} 

        self.user_vectors = {}
        self.recv_user_vectors = {}
    
        self.recv_user_pubkeys = {}
        self.user_pubkeys = {}

        self.user_choice = {}
        self.recv_user_choice = {}

        self.backup_shares_ai = {}
        self.recv_backup_shares_ai = {}

        self.backup_shares_mi = {}
        self.recv_backup_shares_mi = {}
        
        self.online_set = set()
        self.offline_set = set()

        self.ack = {}
        self.recv_ack = {}

        self.recon_shares_mi = {}
        self.recv_recon_shares_mi = {}
        self.recon_shares_ai = {}
        self.recv_recon_shares_ai = {}
        self.recon_index = {}
        self.recv_recon_index = {}
        self._recv_shares_count = 0  # total SHARES messages received (incl. empty)
        
       
        # Track the current iteration and round of the protocol.
        self.current_iteration = 1
        self.current_hash = 0
        self.current_round = 0

        # Waiting mode: 'fixed' = preconfigured timeouts, 'adaptive' = wait for enough messages
        # Rounds 1-3 (setup): always wait for ALL clients (these are all-to-all rounds)
        # Rounds 4-6 (aggregation): use wait_threshold fraction
        self.wait_mode = wait_mode
        self.wait_threshold = wait_threshold
        # Track the expected wakeup time for the current round.
        # When a wakeup fires, compare its delivery time to this.
        # If they don't match, it's a stale wakeup from a previous round.
        self._expected_wakeup_time = None

        # Mapping the message processing functions
        self.aggProcessingMap = {
            0: self.init_func,
            1: self.advertise_keys,
            2: self.establish_graph,
            3: self.forward_shares,
            4: self.collection,
            5: self.check_alive,
            6: self.reconstruction,
        }

        self.namedict = {
            0: "init_func",
            1: "advertise_keys",
            2: "establish_graph",
            3: "forward_shares",
            4: "collection",
            5: "check_alive",
            6: "reconstruction",
        }
    # Simulation lifecycle messages.

    def kernelStarting(self, startTime):
        # self.kernel is set in Agent.kernelInitializing()

        # Initialize custom state properties into which we will accumulate results later.
        self.kernel.custom_state['srv_adkey'] = pd.Timedelta(0)
        self.kernel.custom_state['srv_graph'] = pd.Timedelta(0)
        self.kernel.custom_state['srv_share'] = pd.Timedelta(0)

        self.kernel.custom_state['srv_collection'] = pd.Timedelta(0)
        self.kernel.custom_state['srv_crosscheck'] = pd.Timedelta(0)
        self.kernel.custom_state['srv_reconstruction'] = pd.Timedelta(0)

        # This agent should have negligible (or no) computation delay until otherwise specified.
        self.setComputationDelay(0)

        # Request a wake-up call as in the base Agent.
        super().kernelStarting(startTime)

    def kernelStopping(self):
        # Add the server time components to the custom state in the Kernel, for output to the config.
        # Note that times which should be reported in the mean per iteration are already so computed.
        
        self.kernel.custom_state['srv_adkey'] += (
            self.elapsed_time['ADKEY'] / self.no_of_iterations)
        self.kernel.custom_state['srv_graph'] += (
            self.elapsed_time['GRAPH'] / self.no_of_iterations)
        self.kernel.custom_state['srv_share'] += (
            self.elapsed_time['SHARE'] / self.no_of_iterations)
        
        self.kernel.custom_state['srv_collection'] += (
            self.elapsed_time['COLLECTION'] / self.no_of_iterations)
        self.kernel.custom_state['srv_crosscheck'] += (
            self.elapsed_time['CROSSCHECK'] / self.no_of_iterations)
        self.kernel.custom_state['srv_reconstruction'] += (
            self.elapsed_time['RECONSTRUCTION'] / self.no_of_iterations)

        # Allow the base class to perform stopping activities.
        super().kernelStopping()

    # Simulation participation messages.

    # The service agent wakeup at the end of each round
    # More specifically, it stores the messages on receiving the msgs;
    # When the timing out happens, or it collects enough number of msgs,
    # (i.e., from all clients it is waiting for),
    # it starts processing and replying the messages.

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        if self.current_iteration > self.no_of_iterations:
            return
        # Skip stale wakeups: a stale wakeup fires BEFORE the expected time
        # (it was scheduled for a previous round). Legitimate wakeups fire
        # AT or AFTER the expected time (possibly delayed by computation delay).
        if self._expected_wakeup_time is not None and currentTime < self._expected_wakeup_time:
            return
        print(f"[Server] wakeup iter={self.current_iteration} step={self.namedict[self.current_round]} simtime={currentTime}")
        self.aggProcessingMap[self.current_round](currentTime)

    def _check_adaptive_trigger(self, currentTime):
        """In adaptive mode, check if enough messages arrived to trigger processing early."""
        if self.wait_mode != 'adaptive':
            return
        # Only trigger if we haven't already scheduled an early wakeup for this round.
        # Check: if expected time != the safety timeout time, an adaptive trigger already fired.
        # But simpler: just check if expected time is in the past (already processing).
        if self._expected_wakeup_time is not None and self._expected_wakeup_time <= currentTime:
            # Already triggered or processing — don't trigger again
            return

        triggered = False
        # Rounds 1-3 (setup): wait for ALL participating clients
        if self.current_round == 1:  # advertise_keys: wait for all PUBKEY
            if len(self.recv_user_pubkeys) >= self.num_clients and self.num_clients > 0:
                print(f"[Server] adaptive trigger: {len(self.recv_user_pubkeys)}/{self.num_clients} pubkeys (all)")
                triggered = True
        elif self.current_round == 2:  # establish_graph: wait for all CHOICE
            expected = len(self.user_pubkeys)
            if expected > 0 and len(self.recv_user_choice) >= expected:
                print(f"[Server] adaptive trigger: {len(self.recv_user_choice)}/{expected} choices (all)")
                triggered = True
        elif self.current_round == 3:  # forward_shares: wait for all BACKUP
            expected = len(self.user_choice)
            if expected > 0 and len(self.recv_backup_shares_ai) >= expected:
                print(f"[Server] adaptive trigger: {len(self.recv_backup_shares_ai)}/{expected} backups (all)")
                triggered = True

        # Round 4 (collection): use wait_threshold
        elif self.current_round == 4:
            expected = len(self.user_choice)
            needed = int(expected * self.wait_threshold)
            if expected > 0 and len(self.recv_user_vectors) >= needed:
                print(f"[Server] adaptive trigger: {len(self.recv_user_vectors)}/{expected} vectors (threshold={needed})")
                triggered = True
        # Round 5 (check_alive): wait for ALL online clients' ACKs
        elif self.current_round == 5:
            expected = len(self.user_vectors)
            if expected > 0 and len(self.recv_ack) >= expected:
                print(f"[Server] adaptive trigger: {len(self.recv_ack)}/{expected} acks (all)")
                triggered = True
        # Round 6 (reconstruction): wait for ALL online clients' shares
        elif self.current_round == 6:
            expected = len(self.user_vectors)
            if expected > 0 and self._recv_shares_count >= expected:
                print(f"[Server] adaptive trigger: {self._recv_shares_count}/{expected} recon shares (all)")
                triggered = True

        if triggered:
            # Override the expected wakeup time to this earlier time.
            # The old safety timeout will fire later with a different time → skipped.
            self._expected_wakeup_time = currentTime
            self.setWakeup(currentTime)

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

        sender_id = msg.body['sender']

        if msg.body['msg'] == "PUBKEY":
            dt_protocol_start = pd.Timestamp('now')

            if msg.body['iteration'] == self.current_iteration:
         
                self.recv_user_pubkeys[sender_id] = msg.body['pubkey']
                bench.set_iteration(self.current_iteration)
                bench.record_arrival("pubkeys", currentTime, expected_total=self.num_clients)
               
            else:
                print("Server receives PUBKEY from iteration", msg.body['iteration'],
                      " client ", msg.body['sender'])

        elif msg.body['msg'] == "CHOICE":
            dt_protocol_start = pd.Timestamp('now')

            if msg.body['iteration'] == self.current_iteration:
                
                # store who chooses each client
                self.recv_user_choice[sender_id] = msg.body['choice']
                bench.set_iteration(self.current_iteration)
                bench.record_arrival("choices", currentTime, expected_total=self.num_clients)
                 
            else:
                print("Server receives CHOICE from iteration", msg.body['iteration'],
                      " client ", msg.body['sender'])

        elif msg.body['msg'] == "BACKUP":
            
            if msg.body['iteration'] == self.current_iteration:
                
                # store who chooses each client
                self.recv_backup_shares_ai[sender_id] = msg.body['backup_shares_ai']
                self.recv_backup_shares_mi[sender_id] = msg.body['backup_shares_mi']
                bench.set_iteration(self.current_iteration)
                bench.record_arrival("backups", currentTime, expected_total=self.num_clients)
                
            else:
                print("Server receives BACKUP from iteration", msg.body['iteration'],
                      " client ", msg.body['sender'])

        elif msg.body['msg'] == "VECTOR":

            dt_protocol_start = pd.Timestamp('now')

            if msg.body['iteration'] == self.current_iteration:
                
                # Store the vectors
                self.recv_user_vectors[sender_id] = msg.body['vector']
                bench.set_iteration(self.current_iteration)
                bench.record_arrival("vectors", currentTime, expected_total=self.num_clients)
                
            else:
                print("Server receives VECTORS from iteration", msg.body['iteration'],
                      " client ", msg.body['sender'])

            # Accumulate into offline setup.
            self.recordTime(dt_protocol_start, "COLLECTION")
        
        elif msg.body['msg'] == "ACK":
            dt_protocol_start = pd.Timestamp('now')
            
            if msg.body['iteration'] == self.current_iteration:
                self.recv_ack[sender_id] = msg.body['ack']
                bench.set_iteration(self.current_iteration)
                bench.record_arrival("acks", currentTime, expected_total=self.num_clients)
            else:
                print("Server receives ACK from iteration", msg.body['iteration'],
                      " client ", msg.body['sender'])

        elif msg.body['msg'] == "SHARES":

            dt_protocol_start = pd.Timestamp('now')

            if msg.body['iteration'] == self.current_iteration:
                self._recv_shares_count += 1
                bench.set_iteration(self.current_iteration)
                bench.record_arrival("recon_shares", currentTime, expected_total=self.num_clients)
                if len(msg.body['shares_of_mi']) != 0:
                    self.recv_recon_shares_mi[sender_id] = msg.body['shares_of_mi']
                if len(msg.body['shares_of_ai']) != 0:
                    self.recv_recon_shares_ai[sender_id] = msg.body['shares_of_ai']

            else:
                print("Server receives SHARED_RESULT from iteration", msg.body['iteration'],
                      " client ", msg.body['sender'])

            
            self.recordTime(dt_protocol_start, "RECONSTRUCTION")

        # In adaptive mode, check if we have enough to process early
        self._check_adaptive_trigger(currentTime)

    # NOTE: the currentTime is the 'start' of the function

    # Processing and replying the messages.
    def _next_round(self, currentTime, next_round, comp_delay, fixed_wait):
        """Advance to next round with appropriate wakeup."""
        self.current_round = next_round
        # Don't add comp_delay — recordTime already calls setComputationDelay
        # which the kernel applies separately. Adding both double-counts.
        wakeup_time = currentTime + fixed_wait
        self._expected_wakeup_time = wakeup_time
        self.setWakeup(wakeup_time)

    def init_func(self, currentTime):
        bench.set_iteration(self.current_iteration)
        bench.record_timeline("iter_start", currentTime)
        dt_protocol_start = pd.Timestamp('now')
        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        self._next_round(currentTime, 1, server_comp_delay, pd.Timedelta('3s'))

    def advertise_keys_read_from_pool(self):
        self.user_pubkeys = self.recv_user_pubkeys
        self.recv_user_pubkeys = {}
    
    def advertise_keys_clear_pool(self):
        # Empty the pool for upcoming messages 
        # before server sends requests
        self.recv_user_choice = {}

    def advertise_keys(self, currentTime):
        bench.record_timeline("server starts processing advertise_keys", currentTime)
        dt_protocol_start = pd.Timestamp('now')
        self.advertise_keys_read_from_pool()

        print("Server collected #pubkeys =", len(self.user_pubkeys))

        self.advertise_keys_clear_pool()

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        bench.record_timeline("server finishes advertise_keys, sends graph requests",
                              currentTime + server_comp_delay)

        # send to users who send their pubkeys
        for id in self.user_pubkeys:
            self.sendMessage(id,
                             Message({"msg": "REQ_CHOICE",
                                      "iteration": self.current_iteration,
                                      "sender": 0,
                                      "active_pubkey_clients": list(self.user_pubkeys.keys()),
                                      }),
                             tag="comm_active_clients")
        
        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        print("Server time for advertise_keys:", server_comp_delay)

        self.recordTime(dt_protocol_start, "ADKEY")

        # print serialization size
        if __debug__:
            tmp_pubkeys = {}
            for i in self.user_pubkeys:
                tmp_pubkeys[i] = (int(self.user_pubkeys[i].x), int(self.user_pubkeys[i].y))
            self.logger.info(f"Server pubkey comm cost: {len(dill.dumps(tmp_pubkeys))}")

        self._next_round(currentTime, 2, server_comp_delay, param.wt_google_graph)

    def establish_graph_read_from_pool(self):
        # send who chose id to client id
        self.user_choice = self.recv_user_choice
        self.recv_user_choice = {}

    def establish_graph_clear_pool(self):
        return

    def establish_graph(self, currentTime):
        bench.record_timeline("server starts processing graph", currentTime)
        dt_protocol_start = pd.Timestamp('now')
        self.establish_graph_read_from_pool()
        # Server should know a complete graph, who is whose neighbors
      
        print("Server collected #graph choice =", len(self.recv_user_choice))

        # if set(self.user_choice.keys()) != set(self.user_pubkeys.keys()):
        #     print("user pubkeys:", set(self.user_pubkeys.keys()))
        #     print("user choice:", set(self.user_choice.keys()))
        #     print("The waiting time is not enough: not all required users send their choices.")
        #     exit(1)

        # an optimized heuristic version:
        # for those who does not send choice, remove those from a client's neighbors_out

        neighbors_in = {}

        # store neighbors of each client.
        # neighbors[i] is the neighbors of client i.
        self.neighbors = {}  

        for i in self.user_choice:
            tmp = set()
            # find who chose i
            for j in self.user_choice:
                if i in self.user_choice[j] and j != i:
                    tmp.add(j)
            neighbors_in[i] = tmp

            # remove those who are not in user_choice.keys() from user_choice[i]
            # find those who are in user_choice but not in user_pubkeys

            for straggler in set(self.users)-set(self.user_choice.keys()):
                if straggler in self.user_choice[i]:
                    self.user_choice[i].remove(straggler)
            
            self.neighbors[i] = (neighbors_in[i]).union(self.user_choice[i])
            # print("client", i, "has neighbors =", self.neighbors[i])
        
        # at this point, the server should know the complete graph

        # neighbors_pubkeys[i] stores client i's neighbors pubkeys
        neighbors_pubkeys = {}
        for id in self.user_choice:
            # for each active user, give related pubkeys to them
            tmpls = {}  # each element: (neighbor id, pubkey)
            for j in self.neighbors[id]:
                tmpls[j] = self.user_pubkeys[j]
            neighbors_pubkeys[id] = tmpls

      
        # when sending the graph, also send related pubkeys.
        # for a client, pubkeys include its neighbors_in and neighbors_out
        
        print("number of active choice clients:", len(self.user_choice.keys()))

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        bench.record_timeline("server finishes graph, sends backup requests",
                              currentTime + server_comp_delay)

        for id in self.user_choice:
            self.sendMessage(id,
                             Message({"msg": "REQ_BACKUP",
                                      "iteration": self.current_iteration,
                                      "sender": 0,
                                      "neighbors": self.neighbors[id],
                                      "neighbors_pubkeys": neighbors_pubkeys[id],
                                      "active_choice_clients": list(self.user_choice.keys()),
                                      }),
                             tag="comm_graph_server")
        
        
        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        print("Server time for establish_graph:", server_comp_delay)

        self.recordTime(dt_protocol_start, "GRAPH")

        if __debug__:
            tmp_pubkeys = {}
            for i in self.user_choice:
                tmp_pubkeys[i] = {}
                for j in neighbors_pubkeys[i]:
                    tmp_pubkeys[i][j] = (int(neighbors_pubkeys[i][j].x), int(neighbors_pubkeys[i][j].y))
            tmp_neighbor_ids = {}
            for i in self.user_choice:
                tmp_neighbor_ids[i] = self.neighbors[i]
            self.logger.info(f"Server graph choice comm cost: {len(dill.dumps(tmp_pubkeys)) + len(dill.dumps(tmp_neighbor_ids))}")

        self._next_round(currentTime, 3, server_comp_delay, param.wt_google_share)
    
    def forward_signatures_read_from_pool(self):
        # self.backup_shares_ai[id] is the shares from client id
        # the server should forward the shares to id's neighbors
        self.backup_shares_ai = self.recv_backup_shares_ai
        self.recv_backup_shares_ai = {}

        self.backup_shares_mi = self.recv_backup_shares_mi
        self.recv_backup_shares_mi = {}

    def forward_signatures_clear_pool(self):
        self.recv_user_vectors = {}

    def forward_shares(self, currentTime):
        bench.record_timeline("server starts processing forward_shares", currentTime)
        dt_protocol_start = pd.Timestamp('now')
        self.forward_signatures_read_from_pool()

        # ai_shares is a list of length = number of neighbors
        # the server will send each point in ai_shares to each of the client's neighbors
        

        # forward_shares_ai[id] is shares to be forwarded to client id.
        # forward_shares_ai[id] is a dictionary,
        #   where forward shares_ai[id][neighbor_id] is a share (point) stored
        #   at client id for neighbors_id

        #       neighbor_id, share point (x, y)
        #       neighbor_id, share point (x, y)
        #  id     ...
        #       neighbor_id, share point (x, y)
        #       neighbor_id, share point (x, y)

        forward_shares_ai = {}
        for i in self.backup_shares_ai:
            for j in self.neighbors[i]:
                forward_shares_ai[j] = {}   # might be repeatedly initialize, but it's ok

        for i in self.backup_shares_ai:
            if len(self.neighbors[i]) != len(self.backup_shares_ai[i]):
                raise ValueError("#of shares does not match #neighbors.")
            cnt = 0
            for j in self.neighbors[i]:
                forward_shares_ai[j][i] = self.backup_shares_ai[i][cnt]
                cnt += 1
           
        forward_shares_mi = {}
        for i in self.backup_shares_mi:
            for j in self.neighbors[i]:
                forward_shares_mi[j] = {}   # might be repeatedly initialize, but it's ok

        for i in self.backup_shares_mi:
            if len(self.neighbors[i]) != len(self.backup_shares_mi[i]):
                raise ValueError("#of shares does not match #neighbors.")
            cnt = 0
            for j in self.neighbors[i]:
                forward_shares_mi[j][i] = self.backup_shares_mi[i][cnt]
                cnt += 1

        self.forward_signatures_clear_pool()

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        bench.record_timeline("server finishes forward_shares, sends vector requests",
                              currentTime + server_comp_delay)

        # at the same time of sending shares, request for vector
        for id in self.user_choice:
            self.sendMessage(id,
                             Message({"msg": "REQ_VECTOR",
                                      "iteration": self.current_iteration,
                                      "sender": 0,
                                      "backup_shares_ai": forward_shares_ai[id],
                                      "backup_shares_mi": forward_shares_mi[id],
                                      }),
                             tag="comm_graph_server")

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        print("Server time for forward_shares:", server_comp_delay)

        self.recordTime(dt_protocol_start, "SHARE")

        if __debug__:
            tmp_shares = {}
            for i in self.user_choice:
                tmp_shares[i] = {}
                for j in forward_shares_ai[i]:
                    tmp_shares[i][j] = (int(forward_shares_ai[i][j][1]), int(forward_shares_mi[i][j][1]))
            self.logger.info(f"Server forward shares comm cost: {len(dill.dumps(tmp_shares))}")

        self._next_round(currentTime, 4, server_comp_delay, param.wt_google_collection)

    def collection_read_from_pool(self):
        # assign user vectors to a new var. empty user vectors immediately.
        self.user_vectors = self.recv_user_vectors
        self.recv_user_vectors = {}
    
    def collection_clear_pool(self):
        self.recv_ack = {}

    def collection(self, currentTime):
        bench.record_timeline("server starts processing collection", currentTime)
        dt_protocol_start = pd.Timestamp('now')

        self.collection_read_from_pool()
        print("collected vectors =", len(self.user_vectors))
        
        # compute the sum of vectors
        self.vec_sum_partial = np.zeros(self.vector_len, dtype=self.vector_dtype)
        for id in self.user_vectors:
            if len(self.user_vectors[id]) != self.vector_len:
                raise ValueError("Client sends inconsistent vector length")
            self.vec_sum_partial += self.user_vectors[id] 

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        bench.record_timeline("server finishes collection, sends ACK requests",
                              currentTime + server_comp_delay)

        # here should request alive signatures
        # server should send to the clients who is alive
        self.collection_clear_pool()
        for id in self.user_vectors:
            self.sendMessage(id,
                             Message({"msg": "REQ_ACK",
                                      "iteration": self.current_iteration,
                                      "request_ack": 1,
                                      "alive_set": self.user_vectors,
                                      }),
                             tag="comm_ack_server")


        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        print("Server time for collection:", server_comp_delay)

        self.recordTime(dt_protocol_start, "COLLECTION")

        if __debug__:
            self.logger.info(f"Server comm for collecting vectors: {len(dill.dumps(self.user_vectors))}")

        self._next_round(currentTime, 5, server_comp_delay, param.wt_google_crosscheck)

    def check_alive_read_from_pool(self):
        self.ack = self.recv_ack
        self.recv_ack = {}
    
    def check_alive_clear_pool(self):
        self.recv_recon_shares_mi = {}
        self.recv_recon_shares_ai = {}
        self._recv_shares_count = 0

    def check_alive(self, currentTime):
        bench.record_timeline("server starts processing check_alive", currentTime)
        dt_protocol_start = pd.Timestamp('now')
        self.check_alive_read_from_pool()

        # pack the in edge to i, send ack to i
        in_neighbors_ack = {}
        for i in self.ack:
            for j in self.ack[i]:
                in_neighbors_ack[j] = {}

        for sender in self.ack:
            for recvr in self.ack[sender]:
                in_neighbors_ack[recvr][sender] = self.ack[sender][recvr]
        

        self.online_set = set(self.user_vectors.keys())
        print("online clients:", len(self.online_set))

        self.offline_set = set(self.user_choice) - set(self.online_set)
        print("offline clients:", len(self.offline_set))

        # request_mi_shares[id] is a list, storing neighbors to be request 
        # id will receive this, and send to the server the shares of the requested neighbors
        request_mi_shares = {}  
        request_ai_shares = {}

        # for id in online set, request id's neighbor for shares of mi
        for i in self.user_choice:
            request_mi_shares[i] = []
        
        for i in self.online_set:
            for j in self.neighbors[i]:
                request_mi_shares[j].append(i)

        # for id in offline set, request id's neighbor for shares of ai
        for i in self.user_choice:
            request_ai_shares[i] = []
        
        for i in self.offline_set:
            for j in self.neighbors[i]:
                request_ai_shares[j].append(i)
       

        self.check_alive_clear_pool()

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        bench.record_timeline("server finishes check_alive, sends share requests",
                              currentTime + server_comp_delay)

        for id in self.user_vectors:
            self.sendMessage(id,
                             Message({"msg": "REQ_SHARES",
                                      "iteration": self.current_iteration,
                                      "neighbors_ack": in_neighbors_ack[id], 
                                      "request_mi_shares": request_mi_shares,
                                      "request_ai_shares": request_ai_shares,
                                      }),
                             tag="comm_dec_server")

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        print("Server time for check_alive:", server_comp_delay)

        self.recordTime(dt_protocol_start, "CROSSCHECK")

        if __debug__:
            self.logger.info(f"Server forward ACK comm: {len(dill.dumps(in_neighbors_ack))}")
            self.logger.info(f"Server comm for requesting shares: {len(dill.dumps(request_ai_shares)) + len(dill.dumps(request_mi_shares))}")

        self._next_round(currentTime, 6, server_comp_delay, param.wt_google_recontruction)

    def reconstruction_read_from_pool(self):
        self.recon_shares_mi = self.recv_recon_shares_mi
        self.recv_recon_shares_mi = {}

        self.recon_shares_ai = self.recv_recon_shares_ai
        self.recv_recon_shares_ai = {}

    def reconstruction_clear_pool(self):
        return 

    def reconstruction(self, currentTime):
        bench.record_timeline("server starts processing reconstruction", currentTime)
        dt_protocol_start = pd.Timestamp('now')
        self.reconstruction_read_from_pool()

        # recon_shares_mi:
        #                  id, share
        #   sender_id      id, share
        #                  id, share

        # the server needs to extract shares for one id, 
        #   and reconstruct mi for this id.

        """Reconstruct mi for client i."""
        mi_recon_failures = 0
        for i in self.online_set:
            mi_shares = []  # shares of mi for this client i
            for j in self.recon_shares_mi:
                if i in self.recon_shares_mi[j]:
                    mi_shares.append(self.recon_shares_mi[j][i])
            # Need at least threshold = int(2/3 * num_neighbors) shares
            num_neighbors = len(self.neighbors.get(i, set()))
            threshold = int(2/3 * num_neighbors) if num_neighbors > 0 else 1
            if len(mi_shares) < threshold:
                mi_recon_failures += 1
                if mi_recon_failures <= 3:
                    print(f"  WARNING: client {i} has {len(mi_shares)} mi shares but needs {threshold} (neighbors={num_neighbors})")
                continue
            mi_recon, _ = points_to_secret_int(mi_shares, prime=self.prime, isecc=0)
            # should be 16 bytes, but not mi is not correctly reconstructed
            mi_bytes = (mi_recon&((1<<128)-1)).to_bytes(16, 'big')
            
            prg_mi_holder = AES.new(mi_bytes, AES.MODE_CBC, iv=b"0123456789abcdef")
            data = param.fixed_key * self.vector_len
            prg_mi = prg_mi_holder.encrypt(data)
                
            vec_prg_mi = np.frombuffer(prg_mi, dtype=self.vector_dtype)

            self.vec_sum_partial = self.vec_sum_partial - vec_prg_mi

        ai_recon_failures = 0

        for i in self.offline_set:
            ai_shares = []
            for j in self.recon_shares_ai:
                if i in self.recon_shares_ai[j]:
                    ai_shares.append(self.recon_shares_ai[j][i])

            num_neighbors = len(self.neighbors.get(i, set()))
            threshold = int(2/3 * num_neighbors) if num_neighbors > 0 else 1
            if len(ai_shares) < threshold:
                ai_recon_failures += 1
                if ai_recon_failures <= 3:
                    print(f"  WARNING: offline client {i} has {len(ai_shares)} ai shares but needs {threshold} (neighbors={num_neighbors})")
                continue

            ai_recon, _ = points_to_secret_int(ai_shares, prime=self.prime, isecc=0)

            # Compute pairwise key ONLY with ONLINE neighbors of offline client i.
            # Offline client i did not send a vector, but each online neighbor j DID
            # include ±PRG(key_ij) in their vector. We must cancel those contributions.
            # We must NOT touch masks between two offline clients (neither contributed).
            pairwise_keys = {}
            for j in self.neighbors[i]:
                if j in self.online_set:  # only online neighbors
                    pairwise_keys[j] = ai_recon * self.user_pubkeys[j]

            # compute masks and cancel
            for j in pairwise_keys:
                pairwise_seed = (int(pairwise_keys[j].x) & (1<<128) - 1).to_bytes(16, 'big')

                prg_pairwise_holder = AES.new(pairwise_seed, AES.MODE_CBC, iv=b"0123456789abcdef")
                data = param.fixed_key * self.vector_len

                prg_pairwise = prg_pairwise_holder.encrypt(data)
                vec_prg_pairwise = np.frombuffer(prg_pairwise, dtype=self.vector_dtype)

                if len(vec_prg_pairwise) != self.vector_len:
                    raise ValueError("vector length error")
                # Cancel online neighbor j's pairwise mask contribution with offline client i.
                # j added sign(j,i)*PRG(key) = -sign(i,j)*PRG(key), so we add sign(i,j)*PRG(key).
                if i < j:
                    self.vec_sum_partial = self.vec_sum_partial + vec_prg_pairwise
                elif i > j:
                    self.vec_sum_partial = self.vec_sum_partial - vec_prg_pairwise
                else:
                    raise ValueError("client should not be its own neighbor")
   
   
        total_failures = mi_recon_failures + ai_recon_failures
        if total_failures > 0:
            print(f"  AGGREGATION FAILED: {mi_recon_failures} mi + {ai_recon_failures} ai reconstruction failures "
                  f"(insufficient shares due to dropout exceeding 2/3 neighbor threshold)")
            print(f"  Info: online={len(self.online_set)} offline={len(self.offline_set)} "
                  f"share_senders={len(self.recon_shares_mi)}")
        else:
            print("final sum =", self.vec_sum_partial)

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        print("Server time for reconstruction:", server_comp_delay)

        self.recordTime(dt_protocol_start, "RECONSTRUCTION")
        bench.record_timeline("server finishes reconstruction, sends results",
                              currentTime + server_comp_delay)

        if __debug__: # print serialization size
            tmp_recon_ai = {}
            for i in self.recon_shares_ai:
                tmp_recon_ai[i] = {}
                for j in self.recon_shares_ai[i]:
                    tmp_recon_ai[i][j] = int(self.recon_shares_ai[i][j][1])
        
            tmp_recon_mi = {}
            for i in self.recon_shares_mi:
                tmp_recon_mi[i] = {}
                for j in self.recon_shares_mi[i]:
                    tmp_recon_mi[i][j] = int(self.recon_shares_mi[i][j][1])
            self.logger.info(f"Server comm for recv recon shares: {len(dill.dumps(tmp_recon_ai)) + len(dill.dumps(tmp_recon_mi))}")

        print()
        print("######## Iteration completion ########")
        print(f"[Server] finished iteration {self.current_iteration} at {currentTime + server_comp_delay}")
        print()
        

        # End of the iteration
        self.current_iteration += 1

        if (self.current_iteration > self.no_of_iterations):
            return

        bench.set_iteration(self.current_iteration)
        bench.record_timeline("iter_start", currentTime)

        # Reset ALL recv pools for a clean start to the next iteration.
        # Without this, late messages from iteration N can contaminate iteration N+1.
        self.recv_user_pubkeys = {}
        self.recv_user_choice = {}
        self.recv_backup_shares_ai = {}
        self.recv_backup_shares_mi = {}
        self.recv_user_vectors = {}
        self.recv_ack = {}
        self.recv_recon_shares_mi = {}
        self.recv_recon_shares_ai = {}
        self._recv_shares_count = 0
        # Also reset processed state
        self.neighbors = {}
        self.user_pubkeys = {}
        self.user_choice = {}
        self.online_set = set()
        self.offline_set = set()

        for id in self.users:
            self.sendMessage(id,
                             Message({"msg": "REQ_PUBKEY",
                                      "iteration": self.current_iteration,
                                      "sender": 0,
                                      "output": 1,
                                      }),
                             tag="comm_output_server")

        self._next_round(currentTime, 1, server_comp_delay, param.wt_google_adkey)


# ======================== UTIL ========================

    def recordTime(self, startTime, categoryName):
        dt_protocol_end = pd.Timestamp('now')
        self.elapsed_time[categoryName] += dt_protocol_end - startTime
