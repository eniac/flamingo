# malicious version
from agent.Agent import Agent
from agent.google_malicious.SA_ServiceAgent import SA_ServiceAgent as ServiceAgent
from message.Message import Message

from Cryptodome.Hash import SHA256
from Cryptodome.PublicKey import ECC
from Cryptodome.Signature import DSS

from util.crypto.secretsharing import secret_int_to_points, points_to_secret_int
from util import param
from util.crypto import ecchash

import math
import libnum

import logging
import time
import dill

import nacl.utils
import numpy as np
from os.path import exists
import pandas as pd
import random

from Cryptodome.PublicKey import ECC
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes

from sklearn.neural_network import MLPClassifier

# The PPFL_TemplateClientAgent class inherits from the base Agent class.  It has the
# structure of a secure federated learning protocol with secure multiparty communication,
# but without any particular learning or noise methods.  That is, this is a template in
# which the client parties simply pass around arbitrary data.  Sections that would need
# to be completed are clearly marked.

class SA_ClientAgent(Agent):

    def __init__(self, id, name, type,
                 peer_list=None,
                 iterations=4,
                 max_input=10000,
                 key_length=256,
                 num_clients=None,
                 num_neighbors=-1,
                 threshold=-1,
                 num_subgraphs=None,
                 debug_mode=0,
                 random_state=None,
                 X_train=None,
                 y_train=None,
                 input_length=1024,
                 classes=None,
                 nk=10,
                 c=100,
                 m=16):

        # Base class init.
        super().__init__(id, name, type, random_state)

 # Record the number of iterations the clients will perform.
        self.no_of_iterations = iterations
        
        self.logger = logging.getLogger("Log")
        self.logger.setLevel(logging.INFO)
        if debug_mode:
            logging.basicConfig()

        # client's secret key. used to establish pairwise secret with neighbors' public key
        mykey = ECC.generate(curve='P-256')
        self.secret_key = mykey.d
        self.public_key = mykey.pointQ
         
        self.prime = ecchash.n


        # Record the total number of clients participating in the protocol and the number of subgraphs.
        # Neither of these are part of the protocol, or necessary for real-world implementation, but do
        # allow for convenient logging of progress and results in simulation.
        self.num_clients = num_clients

        self.active_pubkey_clients = []
        self.active_choice_clients = []

        self.stored_shares_ai = {}
        self.stored_shares_mi = {}

        self.vector_len = input_length
        self.vector_dtype = 'uint32'
        
        self.neighbors = set()
        self.neighbors_pubkeys = {}
        self.num_choose = math.ceil(math.log2(self.num_clients))
        self.neighbors_out = set()
        self.neighbors_in = set()
        self.mi_bytes = None

        # Initialize a dictionary to accumulate this client's timing information by task.
        self.elapsed_time = {'ADKEY': pd.Timedelta(0),
                             'GRAPH': pd.Timedelta(0),
                             'SHARE': pd.Timedelta(0),
                             'COLLECTION': pd.Timedelta(0),
                             'CROSSCHECK': pd.Timedelta(0),
                             'RECONSTRUCTION': pd.Timedelta(0),
                             }

        # Set to unit vector for testing.
        self.vec = np.ones(self.vector_len, dtype=self.vector_dtype)

        # Iteration counter.
        self.current_iteration = 1
        self.current_base = 0
        

        # MLP inputs
        self.classes = classes
        self.nk = nk
        if (self.nk < len(self.classes)) or (self.nk >= X_train.shape[0]):
            print("nk is a bad size")
            exit(0)

        self.global_coefs = None
        self.global_int = None
        self.global_n_iter = None
        self.global_n_layers = None
        self.global_n_outputs = None
        self.global_t = None
        self.global_nic = None
        self.global_loss = None
        self.global_best_loss = None
        self.global_loss_curve = None
        self.c = c
        self.m = m

        # pick local training data
        self.prng = np.random.Generator(np.random.SFC64())
        obv_per_iter = self.nk #math.floor(X_train.shape[0]/self.num_clients)

        self.trainX = [np.empty((obv_per_iter,X_train.shape[1]),dtype=X_train.dtype) for i in range(self.no_of_iterations)]
        self.trainY = [np.empty((obv_per_iter,),dtype=X_train.dtype) for i in range(self.no_of_iterations)]

        for i in range(self.no_of_iterations):
            #self.input.append(self.prng.integer(input_range));
            slice = self.prng.choice(range(X_train.shape[0]), size=obv_per_iter, replace = False)
            perm = self.prng.permutation(range(X_train.shape[0]))
            p = 0
            while (len(set(y_train[slice])) < len(self.classes)):
                if p >= X_train.shape[0]:
                    print("Dataset does not have the # classes it claims")
                    exit(0)
                add = [perm[p]]
                merge = np.concatenate((slice, add))
                if (len(set(y_train[merge])) > len(set(y_train[slice]))):
                    u, c = np.unique(y_train[slice], return_counts=True)
                    dup = u[c > 1]
                    rm = np.where(y_train[slice] == dup[0])[0][0]
                    slice = np.concatenate((add, np.delete(slice, rm)))
                p += 1

            if (slice.size != obv_per_iter):
                print("n_k not going to be consistent")
                exit(0)

            # Pull together the current local training set.
            self.trainX.append(X_train[slice].copy())
            self.trainY.append(y_train[slice].copy())

    def kernelStarting(self, startTime):

        # Initialize custom state properties into which we will later accumulate results.
        # To avoid redundancy, we allow only the first client to handle initialization.
        if self.id == 1:
            self.kernel.custom_state['clt_adkey'] = pd.Timedelta(0)
            self.kernel.custom_state['clt_graph'] = pd.Timedelta(0)
            self.kernel.custom_state['clt_share'] = pd.Timedelta(0)
            
            self.kernel.custom_state['clt_collection'] = pd.Timedelta(0)
            self.kernel.custom_state['clt_crosscheck'] = pd.Timedelta(0)
            self.kernel.custom_state['clt_reconstruction'] = pd.Timedelta(0)

        # Find the PPFL service agent, so messages can be directed there.
        self.serviceAgentID = self.kernel.findAgentByType(ServiceAgent)

        # Request a wake-up call as in the base Agent.  Noise is kept small because
        # the overall protocol duration is so short right now.  (up to one microsecond)
        super().kernelStarting(startTime +
                               pd.Timedelta(self.random_state.randint(low=0, high=1000), unit='ns'))

    def kernelStopping(self):

        # Accumulate into the Kernel's "custom state" this client's elapsed times per category.
        # Note that times which should be reported in the mean per iteration are already so computed.
        # These will be output to the config (experiment) file at the end of the simulation.
        
        self.kernel.custom_state['clt_adkey'] += (
            self.elapsed_time['ADKEY'] / self.no_of_iterations)
        self.kernel.custom_state['clt_graph'] += (
            self.elapsed_time['GRAPH'] / self.no_of_iterations)
        self.kernel.custom_state['clt_share'] += (
            self.elapsed_time['SHARE'] / self.no_of_iterations)
        
        self.kernel.custom_state['clt_collection'] += (
            self.elapsed_time['COLLECTION'] / self.no_of_iterations)
        self.kernel.custom_state['clt_crosscheck'] += (
            self.elapsed_time['CROSSCHECK'] / self.no_of_iterations)
        self.kernel.custom_state['clt_reconstruction'] += (
            self.elapsed_time['RECONSTRUCTION'] / self.no_of_iterations)

        super().kernelStopping()

    # Simulation participation messages.

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        
        dt_wake_start = pd.Timestamp('now')
        
        self.serviceAgentID = 0
        self.mykey = ECC.generate(curve='P-256')
       
        self.secret_key = self.mykey.d
        self.public_key = self.mykey.pointQ

        # self.mi_bytes = get_random_bytes(16)

        self.sendMessage(self.serviceAgentID,
                             Message({"msg": "PUBKEY",
                                      "iteration": self.current_iteration,
                                      "sender": self.id,
                                      "pubkey": self.public_key,
                                      }),
                             tag="pubkey_to_server")

        

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

        
        if msg.body['msg'] == "REQ_CHOICE":
            
            dt_protocol_start = pd.Timestamp('now')

            if msg.body['iteration'] == self.current_iteration:
            
                self.active_pubkey_clients = msg.body['active_pubkey_clients']

                # graph choice: each client generates k random numbers as neighbors
                self.neighbors_out = set(random.sample(self.active_pubkey_clients, self.num_choose))
                if self.id in self.neighbors_out:
                    self.neighbors_out.remove(self.id)

                # self.neighbors = self.neighbors_out
                # self.neighbors_in = set()
                      
                self.sendMessage(self.serviceAgentID,
                             Message({"msg": "CHOICE",
                                      "iteration": self.current_iteration,
                                      "sender": self.id,
                                      "choice": self.neighbors_out,
                                      }),
                             tag="choice_to_server")

            if __debug__: 
                self.logger.info(f"Client {self.id} time for processing CHOICE: {pd.Timestamp('now') - dt_protocol_start}")
                self.logger.info(f"Client sends choice, comm: {len(dill.dumps(self.neighbors_out))}")

            self.recordTime(dt_protocol_start, 'GRAPH')

        elif msg.body['msg'] == "REQ_BACKUP":

            dt_protocol_start = pd.Timestamp('now')

            if msg.body['iteration'] == self.current_iteration:

                # receive neighbors from the server, and all the pubkeys of neighbors               
                self.neighbors = msg.body['neighbors']

                self.active_choice_clients = msg.body['active_choice_clients']

                # in google version, client indexed from 1
                # print("At client ", self.id, "neighbors = ", self.neighbors) 
                
                # a dictionary: neighbors_pubkeys[id] = pubkey
                self.neighbors_pubkeys = msg.body['neighbors_pubkeys']

                self.mi_bytes = get_random_bytes(16)
                mi_number = int.from_bytes(self.mi_bytes, 'big')
                mi_shares = secret_int_to_points(secret_int=mi_number, 
                    point_threshold= int(2/3 *len(self.neighbors)), 
                    num_points=len(self.neighbors), prime=self.prime
                )

                # also share secret key
                if self.secret_key == None:
                    raise ValueError("secret key is not set.")
                ai_shares = secret_int_to_points(
                    secret_int=self.secret_key, 
                    point_threshold= int(2/3 *len(self.neighbors)), 
                    num_points=len(self.neighbors), prime=self.prime
                )

            
                # ai_shares is a list of length = number of neighbors
                # the server will send each point in ai_shares to each of the client's neighbors

                self.sendMessage(self.serviceAgentID,
                             Message({"msg": "BACKUP",
                                      "iteration": self.current_iteration,
                                      "sender": self.id,
                                      "backup_shares_ai": ai_shares,  # a list of points
                                      "backup_shares_mi": mi_shares,  # a list of points
                                      }),
                             tag="choice_to_server")
            
            if __debug__:
                self.logger.info(f"Client {self.id} time for processing BACKUP (secret shares): {pd.Timestamp('now') - dt_protocol_start}")

                # print serialization size
                tmp_msg_ai = {}
                for i in range(len(ai_shares)):
                    tmp_msg_ai[i] = int(ai_shares[i][1])
                
                tmp_msg_mi = {}
                for i in range(len(mi_shares)):
                    tmp_msg_mi[i] = int(mi_shares[i][1])

                self.logger.info(f"Client backup shares, comm: {len(dill.dumps(tmp_msg_ai)) + len(dill.dumps(tmp_msg_mi))}")

            self.recordTime(dt_protocol_start, 'SHARE')


        elif msg.body['msg'] == "REQ_VECTOR":
            
            dt_protocol_start = pd.Timestamp('now')

            if msg.body['iteration'] == self.current_iteration:
                
                # In this block, store the shares, and compute a masked vector
                # self.stored_shares_ai is a dictionary
                #       neighbor_id, share point (x, y)
                #       neighbor_id, share point (x, y)
                #       ...
                #       neighbor_id, share point (x, y)
                # note: x is from 0, 1, 2, 3, ....; y looks random

                self.stored_shares_ai = msg.body['backup_shares_ai']
                self.stored_shares_mi = msg.body['backup_shares_mi']

                bench_st = pd.Timestamp('now')

                # computet the masked vector in advance using the neighbors_pubkeys
                # set input vector
                #self.vec = np.ones(self.vector_len, dtype=self.vector_dtype)

                # train local data
                mlp = MLPClassifier()
                if self.current_iteration > 1:
                    mlp = MLPClassifier(warm_start=True)
                    mlp.coefs_ = self.global_coefs.copy()
                    mlp.intercepts_ = self.global_int.copy()

                    mlp.n_iter_ = self.global_n_iter
                    mlp.n_layers_ = self.global_n_layers
                    mlp.n_outputs_ = self.global_n_outputs
                    mlp.t_ = self.global_t
                    mlp._no_improvement_count = self.global_nic
                    mlp.loss_ = self.global_loss
                    mlp.best_loss_ = self.global_best_loss
                    mlp.loss_curve_ = self.global_loss_curve.copy()
                    mlp.out_activation_ = "softmax"

                # num epochs
                for j in range(5):
                    mlp.partial_fit(self.trainX[self.no_of_iterations],self.trainY[self.no_of_iterations],self.classes)
                padding = self.vector_len - 7 - ((mlp.n_layers_-1)*3) #- mlp.n_iter_
                for z in range(mlp.n_layers_ - 1):
                    padding = padding - mlp.coefs_[z].size
                    padding = padding - mlp.intercepts_[z].size

                if padding < 0:
                    print("Need more space to encode model weights, please adjust vector by:"+str(-1*padding))
                    exit(1)

                float_vec = np.concatenate((np.zeros(7),np.zeros((mlp.n_layers_-1)*3)))

                for z in range(mlp.n_layers_ - 1):
                    float_vec = np.concatenate((float_vec,np.array(mlp.coefs_[z]).flatten()))
                for z in range(mlp.n_layers_ - 1):
                    float_vec = np.concatenate((float_vec,np.array(mlp.intercepts_[z]).flatten()))

                float_vec = np.concatenate((float_vec,np.zeros(padding)))
                vec = np.vectorize(lambda d: (d+self.c) * pow(2,self.m))(float_vec).astype(self.vector_dtype)

                vec[0] = 1 #mlp.n_iter_
                vec[1] = mlp.n_layers_
                vec[2] = mlp.n_outputs_
                vec[3] = mlp.t_
                vec[4] = mlp._no_improvement_count
                vec[5] = mlp.loss_
                vec[6] = mlp.best_loss_

                x = 7
                for z in range(mlp.n_layers_ - 1):
                    vec[x] = mlp.coefs_[z].shape[0]
                    x += 1
                    vec[x] = mlp.coefs_[z].shape[1]
                    x += 1
                for z in range(mlp.n_layers_ - 1):
                    vec[x] = mlp.intercepts_[z].size
                    x += 1

                #self.vec = np.ones(self.vector_len, dtype=self.vector_dtype)
                self.vec = vec

                # compute individual mask
                if self.mi_bytes == None:
                    print(self.id, "mi_bytes None")
                    raise ValueError("mi_bytes is None.")

                prg_mi_holder = AES.new(self.mi_bytes, AES.MODE_CBC, iv=b"0123456789abcdef")
                data = b"secr" * self.vector_len
                prg_mi = prg_mi_holder.encrypt(data)
                
                vec_prg_mi = np.frombuffer(prg_mi, dtype=self.vector_dtype)
                if len(vec_prg_mi) != self.vector_len:
                    raise ValueError("vector length error")
                
                # mask vector with PRG(mi)
                self.vec += vec_prg_mi

                # compute pairwise mask
                prg_pairwise = {}
                vec_prg_pairwise = {}
                for id in self.neighbors:
                    
                    pairwise_secret_group = self.secret_key * self.neighbors_pubkeys[id]
                    # print("pubkey for neighbors", id, neighbors_pubkeys[id])
                    pairwise_seed = (int(pairwise_secret_group.x) & (1<<128) - 1).to_bytes(16, 'big')
                    
                    prg_pairwise_holder = AES.new(pairwise_seed, AES.MODE_CBC, iv=b"0123456789abcdef")
                    
                    data = b"secr" * self.vector_len
                    
                    prg_pairwise[id] = prg_pairwise_holder.encrypt(data)
                    vec_prg_pairwise[id] = np.frombuffer(prg_pairwise[id], dtype=self.vector_dtype)
                    
                    # mask vector with PRG(r_ij)
                    if len(vec_prg_pairwise[id]) != self.vector_len:
                        raise ValueError("vector length error")
                    if self.id < id:
                        self.vec = self.vec + vec_prg_pairwise[id]
                    elif self.id > id:
                        self.vec = self.vec - vec_prg_pairwise[id]
                    else:
                        raise ValueError("self.id =", self.id, " should not appear in neighbors", self.neighbors)
                        
                bench_ed = pd.Timestamp('now')
                # print("client", self.id, "computes vector in", bench_ed - bench_st)

                if self.id in self.active_choice_clients:
                    # print("client", self.id, "send vector", self.vec)
                    
                    self.sendMessage(self.serviceAgentID,
                             Message({"msg": "VECTOR",
                                      "iteration": self.current_iteration,
                                      "sender": self.id,
                                      "vector": self.vec,
                                      "layers": mlp.n_layers_,
                                      "iter": mlp.n_iter_,
                                      "out": mlp.n_outputs_,
                                      }),
                             tag="vector_to_server")

                                 
                # Accumulate into offline setup.
                self.recordTime(dt_protocol_start, 'RECONSTRUCTION')
            
            if __debug__:
                self.logger.info(f"Client {self.id} time for processing VECTOR: {pd.Timestamp('now') - dt_protocol_start}")
                self.logger.info(f"Client vector comm: {len(dill.dumps(self.vec))}")  # serialization size

            self.recordTime(dt_protocol_start, 'COLLECTION')

        elif msg.body['msg'] == "REQ_ACK":
            dt_protocol_start = pd.Timestamp('now')

            if msg.body['iteration'] == self.current_iteration:
                
                # sign a message for every neighbors
                ack_sig = {}
                
                signer = DSS.new(self.mykey, 'fips-186-3')
                
                for i in self.neighbors:
                    # server should notify clients who is alive
                    if i in msg.body['alive_set']:        
                        ack_msg = str.encode(str(self.id) + str(i))
                        h = SHA256.new(ack_msg)
                        ack_sig[i] = signer.sign(h)


                self.sendMessage(self.serviceAgentID,
                             Message({"msg": "ACK",
                                      "iteration": self.current_iteration,
                                      "sender": self.id,
                                      "ack": ack_sig,
                                      }),
                             tag="shares_to_server")
            
                if __debug__: 
                    self.logger.info(f"Client {self.id} time for processing ACK: {pd.Timestamp('now') - dt_protocol_start}")
                    self.logger.info(f"Client ACK comm: {len(dill.dumps(ack_sig))}")

            self.recordTime(dt_protocol_start, 'CROSSCHECK')

        elif msg.body['msg'] == "REQ_SHARES":
            
            dt_protocol_start = pd.Timestamp('now')

            if msg.body['iteration'] == self.current_iteration:
                
                # process shares request

                # first check ack signatures, if pass send, if not, don't send.
                if len(msg.body['neighbors_ack']) > (2/3) * len(self.neighbors):

                    # request_mi_list is a list, indicating neighbors to be request 
                    # upon receiving this, send to the server the shares of the requested neighbors
                    request_mi_list = msg.body['request_mi_shares']
                    request_ai_list = msg.body['request_ai_shares']

                    # for each share, should send (id, share point)
                    # i.e., (id, (x, y))
                    # self.stored_shares_ai is a dictionary
                    #       neighbor_id, share point (x, y)
                    #       neighbor_id, share point (x, y)
                    #       ...
                    #       neighbor_id, share point (x, y)

                    send_mi_shares = {}
                    for i in request_mi_list:
                        if i in self.stored_shares_mi:
                            send_mi_shares[i] = self.stored_shares_mi[i]
                
                    send_ai_shares = {}
                    for i in request_ai_list:
                        if i in self.stored_shares_ai:
                            send_ai_shares[i] = self.stored_shares_ai[i]

                    if len(request_mi_list) != 0 or len(request_ai_list) != 0:
                        self.sendMessage(self.serviceAgentID,
                             Message({"msg": "SHARES",
                                      "iteration": self.current_iteration,
                                      "sender": self.id,
                                      "shares_of_mi": send_mi_shares,
                                      "shares_of_ai": send_ai_shares, 
                                      }),
                             tag="shares_to_server")

            
                    if __debug__:
                        self.logger.info(f"Client time for processing SHARES (find shares and send): {pd.Timestamp('now') - dt_protocol_start}")

                        # print serialization size
                        tmp_ai_msg = {}
                        for i in send_ai_shares:
                            tmp_ai_msg[i] = (int(send_ai_shares[i][0]), int(send_ai_shares[i][1]))
                        tmp_mi_msg = {}
                        for i in send_mi_shares:
                            tmp_mi_msg[i] = (int(send_mi_shares[i][0]), int(send_mi_shares[i][1]))
                        
                        self.logger.info("Client sends shares for recon, comm:", 
                            len(dill.dumps(tmp_ai_msg))
                            + len(dill.dumps(tmp_mi_msg)))
                
                else:
                    print("Not enough neighbors know I am alive. Does not send shares.")

            self.recordTime(dt_protocol_start, 'RECONSTRUCTION')

        elif msg.body['msg'] == "REQ_PUBKEY" and self.current_iteration != 0:
            
            # send pubkeys to the server 
            dt_protocol_start = pd.Timestamp('now')
            
            self.global_coefs = msg.body['coefs']
            self.global_int = msg.body['ints']
            self.global_n_iter = msg.body['n_iter']
            self.global_n_layers =msg.body['n_layers']
            self.global_n_outputs = msg.body['n_outputs']
            self.global_t = msg.body['t']
            self.global_nic = msg.body['nic']
            self.global_loss = msg.body['loss']
            self.global_best_loss = msg.body['best_loss']
            self.global_loss_curve = msg.body['loss_curve']


            # generate new sk/pk pair
           
            self.mykey = ECC.generate(curve='P-256')
       
            self.secret_key = self.mykey.d
            self.public_key = self.mykey.pointQ

            # process backup request
            # self.mi_bytes = get_random_bytes(16)
            
            self.current_iteration += 1

            if self.current_iteration > self.no_of_iterations:
                # print("client", self.id, "input list:", self.input)
                return

            self.sendMessage(self.serviceAgentID,
                            Message({"msg": "PUBKEY",
                                      "iteration": self.current_iteration,
                                      "sender": self.id,
                                      "pubkey": self.public_key,
                                      }),
                             tag="pubkey_to_server")

            if __debug__:
                self.logger.info(f"Client {self.id} time for processing PUBKEYS (generate new): {pd.Timestamp('now') - dt_protocol_start}")

                # print serialization time
                tmp_pk_msg = (int(self.public_key.x), int(self.public_key.y))
                
                self.logger.info(f"Client sends pubkey, comm: {len(dill.dumps(tmp_pk_msg)) + len(dill.dumps(self.id))}")

            self.recordTime(dt_protocol_start, 'ADKEY')

            # log_print ("Client weights received for iteration {} by {}: {}", self.current_iteration, self.id, output)

            # Start a new iteration if we are not at the end of the protocol.
            # if self.current_iteration < self.no_of_iterations:
            # self.setWakeup(currentTime + pd.Timedelta('1ns'))

    #================= Round logics =================#

        



# ======================== UTIL ========================

    def recordTime(self, startTime, categoryName):
        dt_protocol_end = pd.Timestamp('now')
        self.elapsed_time[categoryName] += dt_protocol_end - startTime
        self.setComputationDelay(
            int((dt_protocol_end - startTime).to_timedelta64()))
