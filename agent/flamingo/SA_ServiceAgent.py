from agent.Agent import Agent
from message.Message import Message

import multiprocessing
import dill
import time
import logging

import math
import numpy as np
import pandas as pd
import random

# pycryptodomex library functions
from Cryptodome.PublicKey import ECC
from Cryptodome.Cipher import AES, ChaCha20
from Cryptodome.Random import get_random_bytes
from Cryptodome.Hash import SHA256
from Cryptodome.Signature import DSS

# other user-level crypto functions
from util import param
from util.crypto import ecchash
from util.crypto.secretsharing import secret_int_to_points, points_to_secret_int

from sklearn.neural_network import MLPClassifier

# parallel helper functions
def parallel_mult(vec, coeff):
    """scalar multiplication for EC points in parallel.
        each df[id] is a vector of EC points;
        interpolation_coefficients is a list, each component is a number
    """
    points = vec.apply(lambda row: ECC.EccPoint(row[0],row[1]),axis=1)
    points = points * coeff
    points = pd.DataFrame([(p.x,p.y) for p in points])
    points = points.applymap(lambda x: int(x))

    return points

# PPFL_ServiceAgent class inherits from the base Agent class.
class SA_ServiceAgent(Agent):

    def __init__(self, id, name, type,
                 random_state=None,
                 msg_fwd_delay=1000000,
                 round_time=pd.Timedelta("10s"),
                 iterations=4,
                 num_clients=128,
                 neighborhood_size=1,
                 parallel_mode=1,
                 debug_mode=0,
                 users={},
                 # inputs for MLP
                 input_length=1024,
                 classes=None,
                 X_test=None,
                 y_test=None,
                 X_help=None,
                 y_help=None,
                 nk=None,
                 n=None,
                 c=100,
                 m=16):

        # Base class init.
        super().__init__(id, name, type, random_state)

        # MLP inputs
        self.classes = classes
        self.X_test = X_test
        self.y_test = y_test
        self.X_help = X_help
        self.y_help = y_help
        self.c = c
        self.m = m
        self.nk = nk
        self.n = n
        self.global_coef = None
        self.global_int = None

        self.logger = logging.getLogger("Log")
        self.logger.setLevel(logging.INFO)

        if debug_mode:
            logging.basicConfig()

        """ Set parameters. """
        # parties
        self.num_clients = num_clients
        self.users = users  # the list of all user IDs
        
        # crypto
        self.prime = ecchash.n

        # inputs
        self.vector_len = input_length #param.vector_len
        self.vector_dtype = param.vector_type

        # graph
        self.neighborhood_size = neighborhood_size

        # parallel
        self.parallel_mode = parallel_mode

        # system  
        self.msg_fwd_delay = msg_fwd_delay  # time to forward a peer-to-peer client relay message
        self.round_time = round_time        # default waiting time per round
        self.no_of_iterations = iterations  # number of iterations 


        """ Read keys. """
        # server (sk, pk)
        try: 
            f = open('pki_files/server_key.pem', "rt")
            self.server_key = ECC.import_key(f.read())
            f.close()
        except IOError:
            raise RuntimeError("No such file. Run setup_pki.py first.")

        # system-wide PK
        try:
            f = open('pki_files/system_pk.pem', "rt")
            key = ECC.import_key(f.read())
            f.close()
            self.system_sk = key.d
        except IOError:
            raise RuntimeError("No such file. Run setup_pki.py first.")

        # agent accumulation of elapsed times by category of tasks
        self.elapsed_time = {'REPORT': pd.Timedelta(0),
                             'CROSSCHECK': pd.Timedelta(0),
                             'RECONSTRUCTION': pd.Timedelta(0),
                             }

        
        # Generate committee members from a root seed.
        self.user_committee = param.choose_committee(param.root_seed, param.committee_size, self.num_clients)

        # Compute committee threshold.
        self.committee_threshold = int(param.fraction * len(self.user_committee))

        
        """ Initialize pools. """
        self.user_vectors = {}
        self.recv_user_vectors = {}

        self.pairwise_cipher = {}
        self.recv_pairwise_cipher = {}
        self.mi_cipher = {}
        self.recv_mi_cipher = {}

        self.committee_shares_pairwise = {} 
        self.recv_committee_shares_pairwise = {}
        self.committee_shares_mi = {}
        self.recv_committee_shares_mi = {}

        self.recv_committee_sigs = {}
        self.committee_sigs = {}

        self.recon_index = {}
        self.recv_recon_index = {}

        self.dec_target_pairwise = {}
        self.dec_target_mi = {}

        self.vec_sum_partial = np.zeros(self.vector_len, dtype=self.vector_dtype)
       
        # Track the current iteration and round of the protocol.
        self.current_iteration = 1
        self.current_round = 0

        # Map the message processing functions
        self.aggProcessingMap = {
            0: self.initFunc,
            1: self.report,
            2: self.forward_signatures,
            3: self.reconstruction,
        }

        self.namedict = {
            0: "initFunc",
            1: "report",
            2: "forward_signatures",
            3: "reconstruction",
        }
    
    # Simulation lifecycle messages.

    def kernelStarting(self, startTime):
        # self.kernel is set in Agent.kernelInitializing()

        # Initialize custom state properties into which we will accumulate results later.
        self.kernel.custom_state['srv_report'] = pd.Timedelta(0)
        self.kernel.custom_state['srv_crosscheck'] = pd.Timedelta(0)
        self.kernel.custom_state['srv_reconstruction'] = pd.Timedelta(0)

        # This agent should have negligible (or no) computation delay until otherwise specified.
        self.setComputationDelay(0)

        # Request a wake-up call as in the base Agent.
        super().kernelStarting(startTime)

    def kernelStopping(self):
        # Add the server time components to the custom state in the Kernel, for output to the config.
        # Note that times which should be reported in the mean per iteration are already so computed.
        self.kernel.custom_state['srv_report'] += (
            self.elapsed_time['REPORT'] / self.no_of_iterations)
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
        print(f"[Server] wakeup in iteration {self.current_iteration} at function {self.namedict[self.current_round]}; current time is {currentTime}")

        # In the k-th iteration
        self.aggProcessingMap[self.current_round](currentTime)

    # On receiving messages

    def receiveMessage(self, currentTime, msg):
        # Allow the base Agent to do whatever it needs to.
        super().receiveMessage(currentTime, msg)

        # Get the sender's id (should be client id)
        sender_id = msg.body['sender']

        """Collect messages from clients.
        Three types: 
            VECTOR message meant for report step, 
            SIGN message meant for crosscheck step,
            SHARED_RESULT message meant for reconstruction step.
        """
        # Collect masked vectors from clients
        if msg.body['msg'] == "VECTOR":
            dt_protocol_start = pd.Timestamp('now')

            if msg.body['iteration'] == self.current_iteration:
                
                # Store the vectors
                self.recv_user_vectors[sender_id] = msg.body['vector']
                if __debug__: 
                    self.logger.info(f"Server received vector from client {sender_id-1} at {currentTime}")
                # ML parameters
                self.final_layers = msg.body['layers']
                self.final_outputs = msg.body['out']
                self.final_iter = msg.body['iter']
    

                # parse the cipher for pairwise and mi
                cur_clt_pairwise_cipher = msg.body['enc_pairwise']
                prev_len = len(self.recv_pairwise_cipher)
                for d in (self.recv_pairwise_cipher, cur_clt_pairwise_cipher): self.recv_pairwise_cipher.update(d)
                post_len = len(self.recv_pairwise_cipher)
                if post_len - prev_len != len(cur_clt_pairwise_cipher):
                    raise RuntimeError("Some pairwise secret has been sent twice. Error in offline/online status of some clients.")
                                
                # parse cipher for shares of mi
                self.recv_mi_cipher[sender_id] = msg.body['enc_mi_shares']

            else:
                if __debug__:
                    self.logger.info(f"LATE MSG: Server receives VECTORS from iteration {msg.body['iteration']} client {msg.body['sender']}")

        # Collect signed labels from decryptors
        elif msg.body['msg'] == "SIGN":
            dt_protocol_start = pd.Timestamp('now')

            if msg.body['iteration'] == self.current_iteration:
                # forward the signatures to all decryptors
                self.recv_committee_sigs[sender_id] = msg.body['signed_labels']
            
            else:
                if __debug__:
                    self.logger.info(f"LATE MSG: Server receives signed labels from iteration {msg.body['iteration']} client {msg.body['sender']}")

        # Collect partial decryption results from decryptors
        elif msg.body['msg'] == "SHARED_RESULT":
            
            dt_protocol_start = pd.Timestamp('now')
            
            if msg.body['iteration'] == self.current_iteration:
                
                self.recv_committee_shares_pairwise[sender_id] = msg.body['shared_result_pairwise']
                self.recv_committee_shares_mi[sender_id] = msg.body['shared_result_mi']
                self.recv_recon_index[sender_id] = msg.body['committee_member_idx']
                
            else:
                if __debug__:
                    self.logger.info(f"LATE MSG: Server receives SHARED_RESULT from iteration {msg.body['iteration']} client {msg.body['sender']}")

    
    
    # Processing and replying the messages.
    # NOTE: the currentTime parameter is the 'start' of the function
    def initFunc(self, currentTime):
        dt_protocol_start = pd.Timestamp('now')
      
        # Simulate the Shamir share of SK at each decryptor
        sk_shares = secret_int_to_points(secret_int=self.system_sk, 
            point_threshold=self.committee_threshold, num_points=len(self.user_committee), prime=self.prime)

        # Send shared sk to committee members
        if __debug__: self.logger.info(f"Server sends to committee members:, {self.user_committee}")
        
        cnt = 0  
        for id in self.user_committee:
            self.sendMessage(id,
                             Message({"msg": "COMMITTEE_SHARED_SK",
                                      "committee_member_idx" : cnt + 1, # the share evaluation x-point starts at 1
                                      "sk_share": sk_shares[cnt],
                                      }),
                             tag="comm_dec_server")
            cnt += 1
        
        self.current_round = 1

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        self.setWakeup(currentTime + server_comp_delay + pd.Timedelta('2s'))

        # Accumulate into time log.
        # self.recordTime(dt_protocol_start, "INIT")

    def report(self, currentTime):
        """Process masked vectors.
            Server at this point should receive:
            vectors, encryption of mi shares, encryption of h_ijt
        """

        dt_protocol_start = pd.Timestamp('now')
        
        # assign user vectors to a new var. empty user vectors immediately.
        self.user_vectors = self.recv_user_vectors
        self.recv_user_vectors = {}

        print("[Server] number of collected vectors:", len(self.user_vectors))

        # for each client, a list of encrypted mi shares (#shares = #commmittee members)
        self.mi_cipher = self.recv_mi_cipher
        self.recv_mi_cipher = {}

        # for each client, a list of encrypted pairwise secrets 
        self.pairwise_cipher = self.recv_pairwise_cipher
        self.recv_pairwise_cipher = {}

        # parse encrypted mi shares, send to committee
        # the target mi is who sent the vectors, so already is
        
        # client_id_list: for committee member to know which pairwise key to decrypt which entry
        client_id_list = list(self.mi_cipher.keys())
        
        # each row of df_mi_shares is the shares of an mi from a client
        df_mi_cipher = pd.DataFrame((self.mi_cipher).values())
        
        # compute which pairwise secrets are in dec target:
        # only edge between an online client and an offline client
        online_set = set()
        offline_set = set()
        online_set = set(self.user_vectors.keys())    
        offline_set = set(self.users) - set(online_set)
        if __debug__:
            self.logger.info(f"online clients: {len(online_set)}")
            self.logger.info(f"offline clients: {len(offline_set)}")

        # compute incomplete sum
        self.vec_sum_partial = np.zeros(self.vector_len, dtype=self.vector_dtype)
        for id in self.user_vectors:
            if len(self.user_vectors[id]) != self.vector_len:
                raise RuntimeError("Client sends inconsistent vector length")
            self.vec_sum_partial += self.user_vectors[id]
        if __debug__: self.logger.info(f"partial sum = {self.vec_sum_partial}")

        # assemble ciphertexts from self.pairwise_cipher, send to committee for decryption
        self.dec_target_pairwise = {}
        
        # used for server later in reconstruction phase to know whether + or -
        self.recon_symbol = {}
       
        # iterate over offline clients
        for id in offline_set:  
            # TODO OPTMIZATION: store neighbors to reduce time
            # find neighbors for client id
            # client id is from 1 
            clt_neighbors_list = param.findNeighbors(param.root_seed, self.current_iteration, self.num_clients, id, self.neighborhood_size)

            for nb in clt_neighbors_list:       # for all neighbors of this client                
                if nb + 1 in online_set:        # if this client id's neighbor nb is online 
                    if (nb, id-1) not in list(self.pairwise_cipher.keys()):     # find tuples (nb, id - 1) in self.pairwise_cipher
                        print("lost:", (nb, id-1))                              # the first component nb is online client, the second component id-1 is offlne client
                        raise RuntimeError("Message lost. Restart protocol.")
                    self.dec_target_pairwise[(nb, id-1)] = self.pairwise_cipher[(nb, id-1)]
                    if nb > id - 1:            
                        self.recon_symbol[(nb, id-1)] = 1                                 
                    elif nb < id - 1:
                        self.recon_symbol[(nb, id-1)] = -1
                    else:                       # id - 1 == nb
                        raise RuntimeError("id-1 should not be its own neighbor.")
        

        # Should send only the c1 component of the ciphertext to the committee

        # Empty the pool for those upcoming messages before server send requests
        self.recv_committee_shares_mi = {}
        self.recv_committee_shares_pairwise = {}
        self.recv_recon_index = {}

        self.recv_committee_sigs = {}

        msg_to_sign = dill.dumps(offline_set)
        hash_container = SHA256.new(msg_to_sign)
        signer = DSS.new(self.server_key, 'fips-186-3')
        signature = signer.sign(hash_container)
        labels_and_sig = (msg_to_sign, signature)

        cnt = 0
        for id in self.user_committee:
            self.sendMessage(id,
                             Message({"msg": "SIGN",
                                      "iteration": self.current_iteration,
                                      "dec_target_pairwise": self.dec_target_pairwise,
                                      "dec_target_mi": df_mi_cipher[cnt],
                                      "client_id_list": client_id_list,
                                      "labels": labels_and_sig,
                                      }),
                             tag="comm_dec_server")
            cnt += 1
        
        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        print("[Server] run time for report step:", server_comp_delay)

        # Accumulate into time log.
        self.recordTime(dt_protocol_start, "REPORT")
        
        # print serialization size:
        if __debug__:
            self.logger.info(f"[Server] communication for collecting vectors: {len(dill.dumps(self.user_vectors))}")

            tmp_dic = {}
            for tpl in self.dec_target_pairwise:
                tmp_dic[tpl] = (int(self.dec_target_pairwise[tpl][0].x), int(self.dec_target_pairwise[tpl][0].y))
            self.logger.info(f"[Server] communication for signed labels and messages to decrypt: {len(dill.dumps(tmp_dic))}")

        self.current_round = 2
        
        self.setWakeup(currentTime + server_comp_delay + param.wt_flamingo_crosscheck)


    def forward_signatures(self, currentTime):
        """Forward cross check information for decryptors."""

        dt_protocol_start = pd.Timestamp('now')

        self.committee_sigs = self.recv_committee_sigs

        # Empty the pool for those upcoming messages before server send requests
        self.recv_committee_shares_mi = {}
        self.recv_committee_shares_pairwise = {}
        self.recv_recon_index = {}

        for id in self.user_committee:
            self.sendMessage(id,
                             Message({"msg": "DEC",
                                      "iteration": self.current_iteration,
                                      "labels": self.committee_sigs,
                                      }),
                             tag="comm_sign_server")
        
        self.current_round = 3

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        print("[Server] run time for crosscheck step:", server_comp_delay)
        self.setWakeup(currentTime + server_comp_delay + param.wt_flamingo_reconstruction)

        # Accumulate into time log.
        self.recordTime(dt_protocol_start, "CROSSCHECK")

    def reconstruction(self, currentTime):
        """Reconstruct sum."""
        
        # print serialization cost
        tmp_msg_pairwise = {}
        for i in self.recv_committee_shares_pairwise:
            tmp_msg_pairwise[i] = {}
            for j in range (len(self.recv_committee_shares_pairwise[i])):
                tmp_msg_pairwise[i][j] = (int((self.recv_committee_shares_pairwise[i][j]).x), int(self.recv_committee_shares_pairwise[i][j].y))
        
        if __debug__:
            self.logger.info(f"[Server] communication for received decryption shares: {len(dill.dumps(self.recv_committee_shares_mi))+len(dill.dumps(tmp_msg_pairwise))}")
        

        dt_protocol_start = pd.Timestamp('now')

        # if not enough shares received, wait for 0.1 sec
        if len(self.recv_committee_shares_pairwise) < self.committee_threshold:
            time.sleep(0.1)

        self.committee_shares_pairwise = self.recv_committee_shares_pairwise
        self.recv_committee_shares_pairwise = {}

        self.committee_shares_mi = self.recv_committee_shares_mi
        self.recv_committee_shares_mi = {}

        self.recon_index = self.recv_recon_index
        self.recv_recon_index = {}
       
        print("[Server] number of collected shares from decryptors:", len(self.committee_shares_pairwise))
        if len(self.committee_shares_pairwise) < self.committee_threshold:
            raise RuntimeError("No enough shares for decryption received.")

        # TODO OPTIMIZATION: only extract the shares of first 20 committees
    
        # recover mi
        # new version
        # st_bench = pd.Timestamp('now')

        df_mi_shares = pd.DataFrame(self.committee_shares_mi) 
        df_mi_shares = df_mi_shares.iloc[:, :self.committee_threshold]
        primary_points = []   # the shares of mi of the first online users
        for id in df_mi_shares:
            primary_points.append((self.recon_index[id], df_mi_shares[id][0])) 
        
        primary_recon_secret, interpolate_coefficients = points_to_secret_int(
            points=primary_points, prime=self.prime, isecc=0)

        """ Compute mi from shares. """
        cnt = 0
        for id in df_mi_shares:
            # each df_mi[id] is a vector of EC points
            # interpolation_coefficients is a list, each component is a number
            df_mi_shares[id] = (df_mi_shares[id] * interpolate_coefficients[cnt]) % self.prime
            cnt += 1
            
        sum_df = pd.DataFrame(np.sum(df_mi_shares.values, axis=1) % self.prime) 
        
        sum_df[0] = sum_df[0].apply(lambda var: var.to_bytes(32, 'big')) 

        # ed_bench = pd.Timestamp('now')
        # print("bench share recon for mi", ed_bench - st_bench)

        """ Compute mi mask vectors. """
        prg_mi = {}
        mi_vec = np.zeros(self.vector_len, dtype=self.vector_dtype)
        for i in range(len(sum_df)):
            prg_mi_holder = ChaCha20.new(key=sum_df[0][i], nonce=param.nonce)
            data = b"secr" * self.vector_len
            prg_mi[i] = prg_mi_holder.encrypt(data)
            mi_vec = mi_vec - np.frombuffer(prg_mi[i], dtype=self.vector_dtype)


        if len(self.dec_target_pairwise) != 0:

            # parallel version
            if self.parallel_mode:
                df_pairwise = self.committee_shares_pairwise
                cnt = 0
                for k in df_pairwise.keys():
                    if cnt == self.committee_threshold:
                        break
                    df_pairwise[k] = pd.DataFrame([(p.x,p.y) for p in df_pairwise[k]])
                    df_pairwise[k] = df_pairwise[k].applymap(lambda x: int(x))
                    cnt += 1

                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                prods = pool.starmap(parallel_mult, zip(df_pairwise.values(), interpolate_coefficients))

                pool.close()
                pool.terminate()
                pool.join() 

                prods = [p.apply(lambda row: ECC.EccPoint(row[0],row[1]),axis=1) for p in prods]
                prods = pd.DataFrame(prods)
                prods = prods.transpose() 
                sum_df = pd.DataFrame(list(prods.sum(axis=1)))
            
            else:
                df_pairwise = pd.DataFrame(self.committee_shares_pairwise) # .loc[:, :self.committee_threshold]
                df_pairwise = df_pairwise.iloc[:, :self.committee_threshold]
                # multiply interpolate coefficients (might be slow since it is EC scalar mult)
                cnt = 0
                for id in df_pairwise:
                    df_pairwise[id] = df_pairwise[id] * interpolate_coefficients[cnt]
                    cnt += 1
            
                sum_df = pd.DataFrame(list(df_pairwise.sum(axis=1)))

            
            
             # compute c_0^{-s}
            sum_df = -sum_df
        
            # compute c1 column
            tmp_list = list(self.dec_target_pairwise.values())
            dec_list = list(zip(*tmp_list))[1]
            dec_df = pd.DataFrame(dec_list)

            if len(sum_df) != len(dec_df):
                raise RuntimeError("length error.")
            # the decryption result is stored in sum_df
            sum_df = sum_df + dec_df  

            sum_df[0] = sum_df[0].apply(lambda var: 
                SHA256.new(int(var.x).to_bytes(32, 'big') + int(var.y).to_bytes(32, 'big')).digest()[0:32]) 

            # compute pairwise mask vectors
            prg_pairwise = {}
            cancel_vec = np.zeros(self.vector_len, dtype=self.vector_dtype)
       
            if len(sum_df) != len(self.recon_symbol):
                raise RuntimeError("The decrypted length is wrong.")

            recon_symbol_list = list(self.recon_symbol.values())
            for i in range(len(sum_df)):
                prg_pairwise_holder = ChaCha20.new(key=sum_df[0][i], nonce=param.nonce)
                data = b"secr" * self.vector_len
                prg_pairwise[i] = prg_pairwise_holder.encrypt(data)
                
                if recon_symbol_list[i] == 1:
                    cancel_vec = cancel_vec + np.frombuffer(prg_pairwise[i], dtype=self.vector_dtype)
                elif recon_symbol_list[i] == -1:
                    cancel_vec = cancel_vec - np.frombuffer(prg_pairwise[i], dtype=self.vector_dtype)

            final_sum = self.vec_sum_partial + cancel_vec + mi_vec
            print("[Server] final sum:", self.vec_sum_partial + cancel_vec + mi_vec)

        else:
            final_sum = self.vec_sum_partial + mi_vec
            print("[Server] no client dropped out.")
            print("[Server] final sum:", self.vec_sum_partial + mi_vec)

        rec = len(self.user_vectors)

        self.user_vectors = {}
        self.committee_shares_pairwise = {}
        self.committee_shares_mi = {}
        self.recon_index = {}

        # Empty the pool for those upcoming messages before server send requests
        self.user_masked_input = {}
        self.recv_pairwise_cipher = {}
        self.recv_mi_cipher = {}
        self.recv_user_vectors = {}
        
        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        print("[Server] run time for reconstruction step:", server_comp_delay)
        
        # Accumulate into time log.
        self.recordTime(dt_protocol_start, "RECONSTRUCTION")

        #MLP
        mlp = MLPClassifier(max_iter=1,warm_start=True)
        mlp.partial_fit(self.X_help,self.y_help,self.classes)
        
        mlp.n_iter_ = self.final_iter #int(final_sum[0]/rec)
        mlp.n_layers_ = self.final_layers #int(final_sum[1]/rec)
        mlp.n_outputs_ = self.final_outputs #int(final_sum[2]/rec)
        mlp.t_ = int(final_sum[3]/rec)
       
        nums = np.vectorize(lambda d: d * 1/rec)(final_sum)
        nums = np.vectorize(lambda d: (d/pow(2,self.m)) \
                            - self.c )(nums)

        # use aggregation to set MLP classifier
        c_indx = []
        i_indx = []

        x =  7
        for z in range(mlp.n_layers_ - 1):
            a = int(final_sum[x]/rec)
            x += 1
            b = int(final_sum[x]/rec)
            x += 1
            c_indx.append((a,b))
        for z in range(mlp.n_layers_ - 1):
            a = int(final_sum[x]/rec)
            i_indx.append(a)
            x += 1

        #x += mlp.n_iter_ 
        i_nums = []
        c_nums = []
        for z in range(mlp.n_layers_ - 1):
            a,b = c_indx[z]
            c_nums.append(np.reshape(np.array(nums[x:(x+(a*b))]),(a,b)))
            x += (a*b)
        for z in range(mlp.n_layers_ - 1):
            a = i_indx[z]
            i_nums.append(np.reshape(np.array(nums[x:(x+a)]),(a,)))

        mlp.coefs_ = c_nums
        mlp.intercepts_ = i_nums
        
        print("[Server] MLP SCORE: ", mlp.score(self.X_test,self.y_test))
        
        print()
        print("######## Iteration completion ########")
        print(f"[Server] finished iteration {self.current_iteration} at {currentTime + server_comp_delay}")
        print()

        # Send the result back to each client.
        # (global MLP weights, other parameters)
        for id in self.users:
            self.sendMessage(id,
                             Message({"msg": "REQ",
                                      "sender": 0,
                                      "output": 1,
                                      "coefs": mlp.coefs_,
                                      "ints": mlp.intercepts_,
                                      "n_iter": mlp.n_iter_,
                                      "n_layers": mlp.n_layers_,
                                      "n_outputs": mlp.n_outputs_,
                                      "t": mlp.t_,
                                      "nic": mlp._no_improvement_count,
                                      "loss": mlp.loss_,
                                      "best_loss": mlp.best_loss_,
                                      "loss_curve": mlp.loss_curve_,
                                      }),
                             tag="comm_output_server")

        self.current_round = 1

        # End of the iteration
        self.current_iteration += 1
        if (self.current_iteration > self.no_of_iterations):
            return

        self.setWakeup(currentTime + server_comp_delay + param.wt_flamingo_report)


# ======================== UTIL ========================

    def recordTime(self, startTime, categoryName):
        # Accumulate into time log.
        dt_protocol_end = pd.Timestamp('now')
        self.elapsed_time[categoryName] += dt_protocol_end - startTime
