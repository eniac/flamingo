from agent.Agent import Agent
from message.Message import Message

import dill
import json
import logging
import sys
import time as walltime

import math
import numpy as np
import pandas as pd
import random

from Cryptodome.PublicKey import ECC
from Cryptodome.Cipher import AES, ChaCha20
from Cryptodome.Random import get_random_bytes
from Cryptodome.Hash import SHA256
from Cryptodome.Signature import DSS

from util import param
from util import util
from util.crypto import ecchash
from util.crypto.secretsharing import secret_int_to_points, points_to_secret_int
from util.benchmark import bench


# PPFL_ServiceAgent class inherits from the base Agent class.
class SA_ServiceAgent(Agent):

    def __str__(self):
        return "[Server]"

    def __init__(self, id, name, type,
                 random_state=None,
                 msg_fwd_delay=1000000,
                 round_time=pd.Timedelta("10s"),
                 iterations=4,
                 key_length=32,
                 num_clients=10,
                 neighborhood_size=1,
                 parallel_mode=1,
                 debug_mode=0,
                 wait_mode='fixed',
                 wait_threshold=0.9,
                 users={}):

        # Base class init.
        super().__init__(id, name, type, random_state)

        self.logger = logging.getLogger("flamingo.server")
        self.logger.setLevel(logging.DEBUG if debug_mode else logging.WARNING)
        if debug_mode and not self.logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("[Server] %(message)s"))
            self.logger.addHandler(handler)

        # System parameters.
        self.msg_fwd_delay = msg_fwd_delay
        self.round_time = round_time
        self.no_of_iterations = iterations

        # Waiting mode: 'fixed' = preconfigured timeouts, 'adaptive' = wait for enough messages
        self.wait_mode = wait_mode
        self.wait_threshold = wait_threshold
        # Track the expected wakeup time to detect stale wakeups
        self._expected_wakeup_time = None

        # Input parameters.
        self.num_clients = num_clients
        self.users = users
        self.vector_len = param.vector_len
        self.vector_dtype = param.vector_type
        self.vec_sum_partial = np.zeros(self.vector_len, dtype=self.vector_dtype)
        self.final_sum = np.zeros(self.vector_len, dtype=self.vector_dtype)

        # Security parameters.
        self.prime = ecchash.n
        self.key_length = key_length
        self.neighborhood_size = neighborhood_size
        self.committee_threshold = 0

        # Read keys.
        self.server_key = util.read_key("pki_files/server_key.pem")
        self.system_sk = util.read_sk("pki_files/system_pk.pem")

        # agent accumulation of elapsed times by category of tasks
        self.elapsed_time = {'REPORT': pd.Timedelta(0),
                             'CROSSCHECK': pd.Timedelta(0),
                             'RECONSTRUCTION': pd.Timedelta(0),
                             }

        # Initialize pools.
        self.user_vectors = {}
        self.pairwise_cipher = {}
        self.mi_cipher = {}
        self.recon_index = {}

        self.recv_user_vectors = {}
        self.recv_pairwise_cipher = {}
        self.recv_mi_cipher = {}
        self.recv_recon_index = {}

        self.user_committee = {}
        self.committee_shares_pairwise = {}
        self.committee_shares_mi = {}
        self.committee_sigs = {}

        self.recv_committee_shares_pairwise = {}
        self.recv_committee_shares_mi = {}
        self.recv_committee_sigs = {}

        self.dec_target_pairwise = {}
        self.dec_target_mi = {}


        # Track the current iteration and round of the protocol.
        self.current_iteration = 1
        self.current_round = 0
        self._recon_retries = 0

        # Map the message processing functions
        self.aggProcessingMap = {
            0: self.initialize,
            1: self.report,
            2: self.forward_signatures,
            3: self.reconstruction,
        }

        self.namedict = {
            0: "initialize",
            1: "report",
            2: "forward_signatures",
            3: "reconstruction",
        }

    # Simulation lifecycle messages.
    def kernelStarting(self, startTime):
        self.kernel.custom_state['srv_report'] = pd.Timedelta(0)
        self.kernel.custom_state['srv_crosscheck'] = pd.Timedelta(0)
        self.kernel.custom_state['srv_reconstruction'] = pd.Timedelta(0)

        self.setComputationDelay(0)
        super().kernelStarting(startTime)

    def kernelStopping(self):
        self.kernel.custom_state['srv_report'] += (
            self.elapsed_time['REPORT'] / self.no_of_iterations)
        self.kernel.custom_state['srv_crosscheck'] += (
            self.elapsed_time['CROSSCHECK'] / self.no_of_iterations)
        self.kernel.custom_state['srv_reconstruction'] += (
            self.elapsed_time['RECONSTRUCTION'] / self.no_of_iterations)

        super().kernelStopping()


    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        if self.current_iteration > self.no_of_iterations:
            return
        # Skip stale wakeups from previous rounds
        if self._expected_wakeup_time is not None and currentTime < self._expected_wakeup_time:
            return
        self.agent_print(f"wakeup iter={self.current_iteration} step={self.namedict[self.current_round]} simtime={currentTime}")
        self.aggProcessingMap[self.current_round](currentTime)

    def _check_adaptive_trigger(self, currentTime):
        """In adaptive mode, check if enough messages arrived to trigger processing early."""
        if self.wait_mode != 'adaptive':
            return
        # Don't trigger if we already scheduled an early wakeup
        if self._expected_wakeup_time is not None and self._expected_wakeup_time <= currentTime:
            return

        triggered = False
        # Round 1 (report): waiting for VECTOR messages from clients
        if self.current_round == 1:
            needed = int(self.num_clients * self.wait_threshold)
            if len(self.recv_user_vectors) >= needed:
                self.agent_print(f"adaptive trigger: {len(self.recv_user_vectors)}/{self.num_clients} vectors (threshold={needed})")
                triggered = True
        # Round 2 (forward_signatures): waiting for SIGN messages from committee
        elif self.current_round == 2:
            needed = int(len(self.user_committee) * self.wait_threshold)
            if len(self.recv_committee_sigs) >= needed:
                self.agent_print(f"adaptive trigger: {len(self.recv_committee_sigs)}/{len(self.user_committee)} sigs (threshold={needed})")
                triggered = True
        # Round 3 (reconstruction): waiting for SHARED_RESULT from committee
        elif self.current_round == 3:
            if len(self.recv_committee_shares_pairwise) >= self.committee_threshold:
                self.agent_print(f"adaptive trigger: {len(self.recv_committee_shares_pairwise)}/{self.committee_threshold} shares")
                triggered = True

        if triggered:
            self._expected_wakeup_time = currentTime
            self.setWakeup(currentTime)

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

        sender_id = msg.body['sender']
        self.logger.debug(f"received msg from client {sender_id} at {currentTime}")

        if msg.body['msg'] == "VECTOR":
            if msg.body['iteration'] == self.current_iteration:
                self.recv_user_vectors[sender_id] = msg.body['vector']
                self.recv_mi_cipher[sender_id] = util.deserialize_tuples_bytes(msg.body['enc_mi_shares'])
                cur_clt_pairwise_cipher = util.deserialize_dim1_elgamal(msg.body['enc_pairwise'])
                for d in (self.recv_pairwise_cipher, cur_clt_pairwise_cipher):
                    self.recv_pairwise_cipher.update(d)

                # Timeline: record message arrival
                bench.set_iteration(self.current_iteration)
                bench.record_arrival("report_vectors", currentTime, expected_total=self.num_clients)

                # Track communication size
                enc_mi_data = msg.body['enc_mi_shares']
                enc_pw_data = msg.body['enc_pairwise']
                vec_bytes = msg.body['vector'].nbytes
                msg_size = vec_bytes + (len(enc_mi_data) if isinstance(enc_mi_data, bytes) else len(str(enc_mi_data))) + (len(enc_pw_data) if isinstance(enc_pw_data, bytes) else len(str(enc_pw_data)))
                bench.record_comm("client->server (VECTOR)", msg_size)
            else:
                self.logger.debug(f"LATE MSG: VECTORS from iter {msg.body['iteration']} client {sender_id}")

        elif msg.body['msg'] == "SIGN":
            if msg.body['iteration'] == self.current_iteration:
                self.recv_committee_sigs[sender_id] = msg.body['signed_labels']
                bench.set_iteration(self.current_iteration)
                bench.record_arrival("crosscheck_sigs", currentTime,
                                     expected_total=len(self.user_committee))
            else:
                self.logger.debug(f"LATE MSG: SIGN from iter {msg.body['iteration']} client {sender_id}")

        elif msg.body['msg'] == "SHARED_RESULT":
            if msg.body['iteration'] == self.current_iteration:
                self.recv_committee_shares_pairwise[sender_id] = util.deserialize_dim1_ecp(msg.body['shared_result_pairwise'])
                self.recv_committee_shares_mi[sender_id] = util.deserialize_dim1_list(msg.body['shared_result_mi'])
                self.recv_recon_index[sender_id] = msg.body['committee_member_idx']
                bench.set_iteration(self.current_iteration)
                bench.record_arrival("recon_shares", currentTime,
                                     expected_total=len(self.user_committee))
            else:
                self.logger.debug(f"LATE MSG: SHARED_RESULT from iter {msg.body['iteration']} client {sender_id}")

        # In adaptive mode, check if we have enough to process early
        self._check_adaptive_trigger(currentTime)


    # Processing and replying the messages.
    def initialize(self, currentTime):
        bench.set_iteration(self.current_iteration)
        bench.record_timeline("iter_start", currentTime)
        dt_protocol_start = pd.Timestamp('now')

        self.user_committee = param.choose_committee(param.root_seed,
                                                     param.committee_size,
                                                     self.num_clients)
        self.committee_threshold = int(param.fraction * len(self.user_committee))

        sk_shares = secret_int_to_points(secret_int=self.system_sk,
                                         point_threshold=self.committee_threshold,
                                         num_points=len(self.user_committee),
                                         prime=self.prime)

        self.logger.debug(f"committee members: {self.user_committee}")

        cnt = 0
        for id in self.user_committee:
            self.sendMessage(id,
                             Message({"msg": "COMMITTEE_SHARED_SK",
                                      "committee_member_idx" : cnt + 1,
                                      "sk_share": sk_shares[cnt],
                                      }),
                             tag="comm_dec_server")
            cnt += 1

        self.current_round = 1
        # _expected_wakeup_time is set by _next_round or setWakeup calls below

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        if self.wait_mode == 'adaptive':
            # Safety timeout: if adaptive trigger doesn't fire, process after a long wait
            self._expected_wakeup_time = currentTime + pd.Timedelta('60s')
            self.setWakeup(currentTime + pd.Timedelta('60s'))
        else:
            self._expected_wakeup_time = currentTime + pd.Timedelta('2s')
            self.setWakeup(currentTime + pd.Timedelta('2s'))


    def report(self, currentTime):
        bench.set_iteration(self.current_iteration)
        bench.record_timeline("server starts processing report", currentTime)
        bench.start_timer("report")

        dt_protocol_start = pd.Timestamp('now')

        self.report_read_from_pool()
        self.report_process()
        self.report_clear_pool()

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        bench.record_timeline("server finishes report, sends crosscheck requests",
                              currentTime + server_comp_delay)
        bench.stop_timer("report")

        self.report_send_message()
        self.agent_print(f"report step: {server_comp_delay}")

        self.recordTime(dt_protocol_start, "REPORT")
        self.current_round = 2
        # _expected_wakeup_time is set by _next_round or setWakeup calls below
        if self.wait_mode == 'adaptive':
            self._expected_wakeup_time = currentTime + pd.Timedelta('60s')
            self.setWakeup(currentTime + pd.Timedelta('60s'))
        else:
            self._expected_wakeup_time = currentTime + param.wt_flamingo_crosscheck
            self.setWakeup(currentTime + param.wt_flamingo_crosscheck)

    def report_read_from_pool(self):
        self.user_vectors = self.recv_user_vectors
        self.recv_user_vectors = {}

        self.mi_cipher = self.recv_mi_cipher
        self.recv_mi_cipher = {}

        self.pairwise_cipher = self.recv_pairwise_cipher
        self.recv_pairwise_cipher = {}

    def report_clear_pool(self):
        self.recv_committee_shares_mi = {}
        self.recv_committee_shares_pairwise = {}
        self.recv_recon_index = {}
        self.recv_committee_sigs = {}

    def report_process(self):
        n_vectors = len(self.user_vectors)
        self.agent_print(f"collected vectors: {n_vectors}/{self.num_clients}")
        bench.record_metric("online_clients", n_vectors)
        bench.record_metric("offline_clients", self.num_clients - n_vectors)

        self.client_id_list = list(self.mi_cipher.keys())
        self.df_mi_cipher = pd.DataFrame((self.mi_cipher).values())

        online_set = set(self.user_vectors.keys())
        offline_set = set(self.users) - set(online_set)
        self.logger.debug(f"online: {len(online_set)}, offline: {len(offline_set)}")

        bench.start_timer("report", "vector_sum")
        self.vec_sum_partial = np.zeros(self.vector_len, dtype=self.vector_dtype)
        for id in self.user_vectors:
            if len(self.user_vectors[id]) != self.vector_len:
                raise RuntimeError("Client sends vector of incorrect length.")
            self.vec_sum_partial += self.user_vectors[id]
        bench.stop_timer("report", "vector_sum")

        bench.start_timer("report", "find_dec_targets")
        self.dec_target_pairwise = {}
        self.recon_symbol = {}

        for id in offline_set:
            clt_neighbors_list = param.findNeighbors(param.root_seed,
                                                     self.current_iteration,
                                                     self.num_clients,
                                                     id,
                                                     self.neighborhood_size)

            for nb in clt_neighbors_list:
                if nb in online_set:
                    if (nb, id) not in list(self.pairwise_cipher.keys()):
                        raise RuntimeError("Message lost:", (nb, id))
                    self.dec_target_pairwise[(nb, id)] = self.pairwise_cipher[(nb, id)]
                    if nb > id:
                        self.recon_symbol[(nb, id)] = 1
                    elif nb < id:
                        self.recon_symbol[(nb, id)] = -1
                    else:
                        raise RuntimeError("id should not be its own neighbor.")
        bench.stop_timer("report", "find_dec_targets")
        bench.record_metric("dec_targets", len(self.dec_target_pairwise))

        bench.start_timer("report", "sign_labels")
        msg_to_sign = dill.dumps(offline_set)
        hash_container = SHA256.new(msg_to_sign)
        signer = DSS.new(self.server_key, 'fips-186-3')
        signature = signer.sign(hash_container)
        self.labels_and_sig = (msg_to_sign, signature)
        bench.stop_timer("report", "sign_labels")

    def report_send_message(self):
        cnt = 0
        for id in self.user_committee:
            enc_pw = util.serialize_dim1_elgamal(self.dec_target_pairwise)
            enc_mi = util.serialize_tuples_bytes(self.df_mi_cipher[cnt])
            msg_size = (len(enc_pw) if isinstance(enc_pw, bytes) else len(str(enc_pw))) + (len(enc_mi) if isinstance(enc_mi, bytes) else len(str(enc_mi)))
            bench.record_comm("server->committee (SIGN)", msg_size)

            self.sendMessage(id,
                             Message({"msg": "SIGN",
                                      "iteration": self.current_iteration,
                                      "dec_target_pairwise": enc_pw,
                                      "dec_target_mi": enc_mi,
                                      "client_id_list": self.client_id_list,
                                      "labels": self.labels_and_sig,
                                      }),
                             tag="comm_dec_server")
            cnt += 1

    def forward_signatures(self, currentTime):
        bench.set_iteration(self.current_iteration)
        bench.record_timeline("server starts processing crosscheck", currentTime)
        bench.start_timer("crosscheck")

        dt_protocol_start = pd.Timestamp('now')

        self.forward_signatures_read_from_pool()
        self.forward_signatures_clear_pool()

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        bench.record_timeline("server finishes crosscheck, sends decryption requests",
                              currentTime + server_comp_delay)
        bench.stop_timer("crosscheck")

        self.forward_signatures_send_message()

        self.current_round = 3
        # _expected_wakeup_time is set by _next_round or setWakeup calls below

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        bench.stop_timer("crosscheck")
        self.agent_print(f"crosscheck step: {server_comp_delay}")
        if self.wait_mode == 'adaptive':
            self._expected_wakeup_time = currentTime + pd.Timedelta('60s')
            self.setWakeup(currentTime + pd.Timedelta('60s'))
        else:
            self._expected_wakeup_time = currentTime + param.wt_flamingo_reconstruction
            self.setWakeup(currentTime + param.wt_flamingo_reconstruction)

        self.recordTime(dt_protocol_start, "CROSSCHECK")

    def forward_signatures_read_from_pool(self):
        self.committee_sigs = self.recv_committee_sigs
        self.recv_committee_shares_mi = {}

    def forward_signatures_clear_pool(self):
        self.recv_committee_shares_pairwise = {}
        self.recv_recon_index = {}

    def forward_signatures_send_message(self):
        for id in self.user_committee:
            self.sendMessage(id,
                             Message({"msg": "DEC",
                                      "iteration": self.current_iteration,
                                      "labels": self.committee_sigs,
                                      }),
                             tag="comm_sign_server")

    def reconstruction(self, currentTime):
        """Reconstruct sum.

        If not enough decryption shares have arrived, retry after a short
        simulation-time delay (up to max_retries times).
        """
        if len(self.recv_committee_shares_pairwise) < self.committee_threshold:
            self._recon_retries += 1
            if self._recon_retries <= 3:
                self.agent_print(f"waiting for shares ({len(self.recv_committee_shares_pairwise)}/{self.committee_threshold}), retry {self._recon_retries}/3")
                self._expected_wakeup_time = currentTime + param.wt_flamingo_reconstruction
                self.setWakeup(currentTime + param.wt_flamingo_reconstruction)
                return

        self._recon_retries = 0

        bench.set_iteration(self.current_iteration)
        bench.record_timeline("server starts processing reconstruction", currentTime)
        bench.start_timer("reconstruction")

        dt_protocol_start = pd.Timestamp('now')

        self.reconstruction_read_from_pool()
        self.reconstruction_process()
        self.reconstruction_clear_pool()

        server_comp_delay = pd.Timestamp('now') - dt_protocol_start
        bench.record_timeline("server finishes reconstruction, sends results",
                              currentTime + server_comp_delay)
        bench.stop_timer("reconstruction")

        self.reconstruction_send_message()

        self.agent_print(f"reconstruction step: {server_comp_delay}")
        self.recordTime(dt_protocol_start, "RECONSTRUCTION")

        print()
        print(f"######## Iteration {self.current_iteration} complete ########")
        print(f"[Server] simtime={currentTime + server_comp_delay}")
        print()

        self.current_round = 1
        # _expected_wakeup_time is set by _next_round or setWakeup calls below

        self.current_iteration += 1
        if (self.current_iteration > self.no_of_iterations):
            return

        bench.set_iteration(self.current_iteration)
        bench.record_timeline("iter_start", currentTime)

        # Don't add server_comp_delay to wakeup time — recordTime already
        # calls setComputationDelay which the kernel applies separately.
        # Adding both would double-count computation time.
        if self.wait_mode == 'adaptive':
            self._expected_wakeup_time = currentTime + pd.Timedelta('60s')
            self.setWakeup(currentTime + pd.Timedelta('60s'))
        else:
            self._expected_wakeup_time = currentTime + param.wt_flamingo_report
            self.setWakeup(currentTime + param.wt_flamingo_report)

    def reconstruction_read_from_pool(self):
        self.committee_shares_pairwise = self.recv_committee_shares_pairwise
        self.recv_committee_shares_pairwise = {}

        self.committee_shares_mi = self.recv_committee_shares_mi
        self.recv_committee_shares_mi = {}

        self.recon_index = self.recv_recon_index
        self.recv_recon_index = {}

    def reconstruction_clear_pool(self):
        self.user_vectors = {}
        self.committee_shares_pairwise = {}
        self.committee_shares_mi = {}
        self.recon_index = {}

        self.user_masked_input = {}
        self.recv_pairwise_cipher = {}
        self.recv_mi_cipher = {}
        self.recv_user_vectors = {}

    def reconstruction_process(self):
        n_shares = len(self.committee_shares_pairwise)
        self.agent_print(f"decryptor shares: {n_shares}/{self.committee_threshold} (threshold)")
        bench.record_metric("decryptor_shares", n_shares)

        if n_shares < self.committee_threshold:
            raise RuntimeError("Not enough shares for decryption received.")

        bench.start_timer("reconstruction", "lagrange_interp")
        df_mi_shares = pd.DataFrame(self.committee_shares_mi)
        df_mi_shares = df_mi_shares.iloc[:, :self.committee_threshold]
        primary_points = []
        for id in df_mi_shares:
            primary_points.append((self.recon_index[id], df_mi_shares[id][0]))

        primary_recon_secret, interpolate_coefficients = points_to_secret_int(
            points=primary_points, prime=self.prime, isecc=0)
        bench.stop_timer("reconstruction", "lagrange_interp")

        bench.start_timer("reconstruction", "recover_mi")
        cnt = 0
        for id in df_mi_shares:
            df_mi_shares[id] = (df_mi_shares[id] * interpolate_coefficients[cnt]) % self.prime
            cnt += 1

        sum_df = pd.DataFrame(np.sum(df_mi_shares.values, axis=1) % self.prime)
        sum_df[0] = sum_df[0].apply(lambda var: var.to_bytes(self.key_length, 'big'))

        prg_mi = {}
        mi_vec = np.zeros(self.vector_len, dtype=self.vector_dtype)
        for i in range(len(sum_df)):
            prg_mi_holder = ChaCha20.new(key=sum_df[0][i], nonce=param.nonce)
            data = param.fixed_key * self.vector_len
            prg_mi[i] = prg_mi_holder.encrypt(data)
            mi_vec = mi_vec - np.frombuffer(prg_mi[i], dtype=self.vector_dtype)
        bench.stop_timer("reconstruction", "recover_mi")

        if not self.dec_target_pairwise:
            self.agent_print("no client dropped out")
            self.final_sum = self.vec_sum_partial + mi_vec
        else:
            bench.start_timer("reconstruction", "ecc_scalar_mult")
            df_pairwise = pd.DataFrame(self.committee_shares_pairwise)
            df_pairwise = df_pairwise.iloc[:, :self.committee_threshold]
            cnt = 0
            for id in df_pairwise:
                df_pairwise[id] = df_pairwise[id] * interpolate_coefficients[cnt]
                cnt += 1
            sum_df = -pd.DataFrame(list(df_pairwise.sum(axis=1)))
            bench.stop_timer("reconstruction", "ecc_scalar_mult")

            bench.start_timer("reconstruction", "recover_pairwise")
            tmp_list = list(self.dec_target_pairwise.values())
            dec_list = list(zip(*tmp_list))[1]
            dec_df = pd.DataFrame(dec_list)
            if len(dec_df) != len(sum_df):
                raise RuntimeError("length error.")

            sum_df = sum_df + dec_df
            sum_df[0] = sum_df[0].apply(lambda var:
                SHA256.new(int(var.x).to_bytes(self.key_length, 'big')
                   + int(var.y).to_bytes(self.key_length, 'big')).digest()[0 : self.key_length])

            prg_pairwise = {}
            cancel_vec = np.zeros(self.vector_len, dtype=self.vector_dtype)

            if len(sum_df) != len(self.recon_symbol):
                raise RuntimeError("The decrypted length is wrong.")
            recon_symbol_list = list(self.recon_symbol.values())

            for i in range(len(sum_df)):
                prg_pairwise_holder = ChaCha20.new(key=sum_df[0][i], nonce=param.nonce)
                data = param.fixed_key * self.vector_len
                prg_pairwise[i] = prg_pairwise_holder.encrypt(data)

                if recon_symbol_list[i] == 1:
                    cancel_vec = cancel_vec + np.frombuffer(prg_pairwise[i], dtype=self.vector_dtype)
                elif recon_symbol_list[i] == -1:
                    cancel_vec = cancel_vec - np.frombuffer(prg_pairwise[i], dtype=self.vector_dtype)

            self.final_sum = self.vec_sum_partial + cancel_vec + mi_vec
            bench.stop_timer("reconstruction", "recover_pairwise")

        self.agent_print(f"final sum: {self.final_sum}")

    def reconstruction_send_message(self):
        for id in self.users:
            self.sendMessage(id,
                             Message({"msg": "REQ",
                                      "sender": 0,
                                      "output": 1,
                                      }),
                             tag="comm_output_server")



# ======================== UTIL ========================

    def recordTime(self, startTime, categoryName):
        dt_protocol_end = pd.Timestamp('now')
        self.elapsed_time[categoryName] += dt_protocol_end - startTime

    def agent_print(self, *args, **kwargs):
        print("[Server]", *args, **kwargs)
