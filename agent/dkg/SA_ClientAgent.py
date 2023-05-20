from agent.Agent import Agent
from agent.dkg.SA_ServiceAgent import SA_ServiceAgent as ServiceAgent
from message.Message import Message
from util.crypto.secretsharing.polynomials import get_polynomial_points, random_polynomial
from util.crypto import ecchash
from util.util import log_print

import math
import time
import numpy as np
import pandas as pd
import random

from util.crypto.secretsharing import secret_int_to_points, points_to_secret_int
from util import param

from Cryptodome.PublicKey import ECC
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes


# The PPFL_TemplateClientAgent class inherits from the base Agent class.  It has the
# structure of a secure federated learning protocol with secure multiparty communication,
# but without any particular learning or noise methods. 

class SA_ClientAgent(Agent):

    def __init__(self, id, name, type,
                 max_input=10000,
                 key_length=256, 
                 num_clients=None,
                 threshold=-1,
                 num_subgraphs=None,
                 random_state=None):

        # Base class init.
        super().__init__(id, name, type, random_state)

        self.base_point = ECC.EccPoint(ecchash.Gx, ecchash.Gy)

        self.users = [i for i in range (1, num_clients + 1)]
        self.threshold = int(param.fraction * len(self.users))

        self.prime = ecchash.n
 
        self.recv_shares = {}
        self.recv_commitments = {}

        self.sk_share = None

        # Iteration counter.
        self.current_iteration = 1
        self.current_base = 0
        # State flag
        self.setup_complete = False

    # Simulation lifecycle messages.

    def kernelStarting(self, startTime):

        # Find the PPFL service agent, so messages can be directed there.
        self.serviceAgentID = self.kernel.findAgentByType(ServiceAgent)
        super().kernelStarting(startTime +
                               pd.Timedelta(self.random_state.randint(low=0, high=1000), unit='ns'))

    def kernelStopping(self):
        super().kernelStopping()

    # Simulation participation messages.

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        dt_wake_start = pd.Timestamp('now')

        self.sendShareCommitments()

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

        # with signatures of other clients from the server
        if msg.body['msg'] == "SHARE_AND_COMMIT":

            # save the share and the commitments
            # check the share against the commitments
            self.recv_shares = msg.body['shares']
            self.recv_commitments = msg.body['commitments']
            complaints_towards = []

            bench_check_commitment_st = pd.Timestamp('now')

            # for each share from dealer i, check against commitments
            # generate complaints towards bad parties
            for id in self.recv_shares:
                if self.verify_commitment(self.recv_shares[id], self.recv_commitments[id]) == False:
                    print(f"Client {self.id} verifies commitment against client {id} fails")
                    complaints_towards.append(id)

            bench_check_commitment_ed = pd.Timestamp('now')
            print("Party", self.id, " checking commitments takes ", 
                bench_check_commitment_ed - bench_check_commitment_st, "seconds")
            
            # send complaints or accepts
            self.sendMessage(0,
                             Message({"msg": "accept_or_complain",
                                      "sender": self.id, 
                                      "iteration" : self.current_iteration,
                                      "complaints_towards": complaints_towards,
                                      }),
                             tag="comm_acc_compl")

        elif msg.body['msg'] == "ACCEPT_OR_COMPLAIN":

            # receive from server the complaints from others
            # I am party i
            # for each complaint sender j,
            # broadcast s_ij

            bcast_sij = {}
            complaint_list = msg.body['complaints_from']
            for complain_party in complaint_list:
                bcast_sij[complain_party] = self.si_shares_with_recvrs[complain_party]

            self.sendMessage(0,
                             Message({"msg": "forward_share",
                                      "sender" : self.id, 
                                      "iteration" : self.current_iteration,
                                      "bcast_sij": bcast_sij,
                                      }),
                             tag="comm_acc_compl")

        elif msg.body['msg'] == "FORWARD_SHARE":
            # receive from server the broadcasted share
            # check against commitment
            # depending on the check, decide on QUAL

            dt_protocol_start = pd.Timestamp('now')

            recv_bcast_shares = msg.body['bcast_shares']

            disqual = []
            for id in recv_bcast_shares:
                if len(recv_bcast_shares[id]) == 0:
                    continue
                else: 
                    if self.verify_commitment(recv_bcast_shares[id], self.recv_commitments[id]) == False:
                        disqual.append(id)

            qual = set(self.recv_shares.keys()) - set(disqual)

            # sign qual
            self.sendMessage(0,
                             Message({"msg": "bcast_qual",
                                      "sender" : self.id, 
                                      "iteration" : self.current_iteration,
                                      "qual": qual,
                                      }),
                             tag="comm_qual")

        elif msg.body['msg'] == "BCAST_QUAL" and self.current_iteration != 0:
            # agree on qual
            # receive from the server the QUAL sets in others' mind
            dt_protocol_start = pd.Timestamp('now')
            
            quals_dict = msg.body['output']

            qual_list = list(quals_dict.values())
            agreed_qual = max(qual_list, key=qual_list.count)

            self.sk_share = 0
            for id in agreed_qual:
                self.sk_share = (self.sk_share + self.recv_shares[id][1]) % self.prime
            
            print(f"Client {self.id} has sk share: {self.sk_share}")
            # End of protocol
            return 


    #================= Round logics =================#

    def sendShareCommitments(self):
        ##############################################################
        # Check if the clients are still performing the setup phase. #
        ##############################################################

        dt_protocol_start = pd.Timestamp('now')

        # party i generates a random s_i in Z_q and and shares it, and generates commitments
        self.my_random_si = random.randint(0, self.prime - 1)
        poly_deg = self.threshold
        poly_coefficients_si = random_polynomial(poly_deg, self.my_random_si, self.prime)

        self.my_commitments = []
        for j in range(0, poly_deg + 1):
            self.my_commitments.append(self.generate_commitment(poly_coefficients_si[j], self.my_random_si))

        self.si_shares = get_polynomial_points(poly_coefficients_si, len(self.users), self.prime)

        self.si_shares_with_recvrs = {}
        for id in self.users:
            self.si_shares_with_recvrs[id] = self.si_shares[id-1]


        self.serviceAgentID = 0
        self.sendMessage(self.serviceAgentID,
                         Message({"msg": "share_and_commit",
                                  "iteration": self.current_iteration,
                                  "sender" : self.id,
                                  "si_shares": self.si_shares_with_recvrs,
                                  "commitments": self.my_commitments,
                                  # "sig"    : signature,
                                  }),
                         tag="comm_key_generation")


    def generate_commitment(self, commit_msg, commit_randomness):
        commitment = commit_msg * self.base_point
        return commitment

    def verify_commitment(self, sij, coeff_commitments_sij):
        lhs = self.base_point * sij[1]
        rhs = coeff_commitments_sij[0]
        for k in range(1, self.threshold + 1):
            rhs = rhs + (coeff_commitments_sij[k] * ((self.id) ** k))
        
        if lhs != rhs:
            return False
        else: 
            return True

# ======================== UTIL ========================

    def recordTime(self, startTime, categoryName):
        dt_protocol_end = pd.Timestamp('now')
        self.elapsed_time[categoryName] += dt_protocol_end - startTime
        self.setComputationDelay(
            int((dt_protocol_end - startTime).to_timedelta64()))