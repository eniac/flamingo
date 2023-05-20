from agent.Agent import Agent
from message.Message import Message
from util.util import log_print
from util import param

import math
import numpy as np
import pandas as pd
import random

from Cryptodome.PublicKey import ECC


# NEW MESSAGES: CLIENT_WEIGHTS : weights, SHARED_WEIGHTS : weights

# The PPFL_ServiceAgent class inherits from the base Agent class. It provides
# the simple shared service necessary for model combination under secure
# federated learning.

class SA_ServiceAgent(Agent):

    def __init__(self, id, name, type,
                 random_state=None,
                 msg_fwd_delay=1000000,
                 round_time=pd.Timedelta("10s"),
                 num_clients=10,
                 users={},
                 max_input=10000):

        # Base class init.
        super().__init__(id, name, type, random_state)

        # Agent accumulation of elapsed times by category of task.
        self.elapsed_time = {'SHARE_AND_COMMIT': pd.Timedelta(0),
                             'ACCEPT_OR_COMPLAIN': pd.Timedelta(0),
                             'FORWARD_SHARE': pd.Timedelta(0),
                             'BCAST_QUAL': pd.Timedelta(0),
                            }

        # Total number of clients and the threshold
        self.users = users  # The list of all users id
        self.threshold = int(param.fraction * len(self.users))

        self.users_shares = {}
        self.recv_users_shares = {}

        self.users_commitments = {}
        self.recv_users_commitments = {}

        self.users_complaints = {}
        self.recv_users_complaints = {}

        self.bcast_shares = {}
        self.recv_bcast_shares = {}

        self.quals = {}
        self.recv_quals = {}

        # The masked input (in exponent)
        # received from users in each iteartion
        self.user_finish_setup = []
        self.user_masked_input = {}

        # Track the current iteration and round of the protocol.
        self.current_iteration = 1
        self.current_hash = 0
        self.current_round = 0

        # Mapping the message processing functions
        self.aggProcessingMap = {
            0: self.initFunc,
            1: self.share_and_commit,
            2: self.accept_or_complain,
            3: self.forward_share,
            4: self.bcast_qual,
        }

        self.namedict = {
            0: "initFunc",
            1: "share_and_commit",
            2: "accept_or_complain",
            3: "forward_share",
            4: "bcast_qual",
        }


        self.arrival_time_share_and_commit = []
        self.arrival_time_accept_or_complain = []
        self.arrival_time_forward_share = []
        self.arrival_time_bcast_qual = []

    # Simulation lifecycle messages.

    def kernelStarting(self, startTime):
        # self.kernel is set in Agent.kernelInitializing()

        # This agent should have negligible (or no) computation delay until otherwise specified.
        self.setComputationDelay(0)

        # Request a wake-up call as in the base Agent.
        super().kernelStarting(startTime)

    def kernelStopping(self):
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

        # In the k-th iteration
        self.aggProcessingMap[self.current_round](currentTime)

    # On receiving messages

    def receiveMessage(self, currentTime, msg):
        # Allow the base Agent to do whatever it needs to.
        super().receiveMessage(currentTime, msg)

        # Get the sender's id
        sender_id = msg.body['sender']
        
        if msg.body['iteration'] != self.current_iteration:
            raise RunTimeError("wrong iteration number.")
            return

        # Round 1: receiving s_i and commitments from each client i
        if msg.body['msg'] == "share_and_commit":

            self.arrival_time_share_and_commit.append(currentTime)

            # put shares into shares dict
            # put commitments into commitments dict
            self.recv_users_shares[sender_id] = msg.body['si_shares']
            self.recv_users_commitments[sender_id] = msg.body['commitments']

        # Round 2: Receiving accept or complain
        elif msg.body['msg'] == "accept_or_complain":

            self.arrival_time_accept_or_complain.append(currentTime)

            # put the acc/compl into related dicts, let the function process
            self.recv_users_complaints[sender_id] = msg.body['complaints_towards']

        elif msg.body['msg'] == "forward_share":

            self.arrival_time_forward_share.append(currentTime)

            # put s_ij in the pool
            self.recv_bcast_shares[sender_id] = msg.body['bcast_sij']
            
        elif msg.body['msg'] == "bcast_qual":
            self.arrival_time_bcast_qual.append(currentTime)
            
            # put in the pool and forward quals to all decryptors
            self.recv_quals[sender_id] = msg.body['qual']


    # Processing and replying the messages.

    def initFunc(self, currentTime):
        dt_protocol_start = pd.Timestamp('now')

        print("Server init time:", dt_protocol_start)

        self.current_round = 1
        self.setWakeup(currentTime + pd.Timedelta('2s'))


    def share_and_commit(self, currentTime):
        dt_protocol_start = pd.Timestamp('now')

        self.users_shares = self.recv_users_shares
        self.recv_users_shares = {}

        self.users_commitments = self.recv_users_commitments
        self.recv_users_commitments = {}

        # Forward the received shares and commitments to all the clients
        for id in self.users:

            forward_shares = {}
            for row_id in self.users_shares:
                forward_shares[row_id] = self.users_shares[row_id][id]

            self.sendMessage(id,
                             Message({"msg": "SHARE_AND_COMMIT",
                                      "shares": forward_shares,
                                      "commitments": self.users_commitments,
                                      }),
                             tag="server_share_and_commit")
        self.current_round = 2

        # wait for clients to check the commitments, and wait for accept or complaints
        self.setWakeup(currentTime + pd.Timedelta('3s'))

    def accept_or_complain(self, currentTime):

        self.users_complaints = self.recv_users_complaints
        self.recv_users_complaints = {}

        # Send the complaints to all parties
        # Each party has a list of complaints

        for id in self.users:
            complaints_list = []
            for row_id in self.users_complaints:
                if id in self.users_complaints[row_id]:
                    complaints_list.append(row_id)

            self.sendMessage(id,
                             Message({"msg": "ACCEPT_OR_COMPLAIN",
                                      "complaints_from": complaints_list,
                                      }),
                             tag="server_accept_or_complain")

        self.current_round = 3

        # wait for dealer to send s_i
        self.setWakeup(currentTime + pd.Timedelta('2s'))

    def forward_share(self, currentTime):

        # Server reads the shares broadcasted by each party from the pool
        self.bcast_shares = self.recv_bcast_shares
        self.recv_bcast_shares = {}

        # Forward the shares broadcasted by each party
        # Each party has a list of bcast sij
        for id in self.users:
            self.sendMessage(id,
                             Message({"msg": "FORWARD_SHARE",
                                      "bcast_shares": self.bcast_shares,
                                      }),
                             tag="server_forward_share")

        self.current_round = 4
        self.setWakeup(currentTime + pd.Timedelta('2s'))
    
    def bcast_qual(self, currentTime):
        
        # Server forwards to all the clients, each client checks
        self.quals = self.recv_quals
        self.recv_quals = {}

        for id in self.users:
            self.sendMessage(id,
                             Message({"msg": "BCAST_QUAL",
                                      "output": self.quals,
                                      }),
                             tag="server_bcast_qual")

        # no need to waiit for each client checks, protocol ends
        # self.setWakeup(currentTime + pd.Timedelta('200ms'))

        st = self.arrival_time_share_and_commit[0]

        step1_list = []
        for i in range(len(self.arrival_time_share_and_commit)):
            step1_list.append((self.arrival_time_share_and_commit[i] - st).total_seconds())
        
        print(step1_list)

        step2_list = []
        for i in range(len(self.arrival_time_accept_or_complain)):
            step2_list.append((self.arrival_time_accept_or_complain[i] - st).total_seconds())

        print(step2_list)

        step3_list = []
        for i in range(len(self.arrival_time_forward_share)):
            step3_list.append((self.arrival_time_forward_share[i] - st).total_seconds())

        print(step3_list)

        step4_list = []
        for i in range(len(self.arrival_time_bcast_qual)):
            step4_list.append((self.arrival_time_bcast_qual[i] - st).total_seconds())

        print(step4_list)

        self.current_round = 1

        # End of the protocol
        return


# ======================== UTIL ========================

    def recordTime(self, startTime, categoryName):
        # Accumulate into offline setup.
        dt_protocol_end = pd.Timestamp('now')
        self.elapsed_time[categoryName] += dt_protocol_end - startTime
        self.setComputationDelay(
            int((dt_protocol_end - startTime).to_timedelta64()))