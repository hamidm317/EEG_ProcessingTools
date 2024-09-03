import numpy as np
from Utils import Local

from Utils.Constants import LocalDataConstants as LDC

def HandleChannel(Channels_inv):

    Channels = []

    if type(Channels_inv) == int or type(Channels_inv) == float:

        Channels.append(int(Channels_inv))

    else:
        
        for Channel in Channels_inv:

            if type(Channel) == float or type(Channel) == int:

                Channels.append(Channel)

            elif type(Channel) == str:

                tmp = [LDC.names['JulyClusterNames'][i] == Channel for i in range(len(LDC.names['JulyClusterNames']))]

                assert np.any(tmp), "The Channel is not available"

                Channels.append(Channel)

    assert np.all(np.array(Channels) >= 0) and np.all(np.array(Channels) < len(LDC.names['JulyClusterNames'])), "Channel/Cluster Number out of range"

    return Channels

def HandleSubjects(Source, Antagonist):

    BehavioralData, Performance_data = Local.ExperimentDataLoader()
    SOI = Local.AvailableSubjects()

    Sub_G = [[], []] # first element is CTRL Group Members and the Second one the DEP Group

    for i, sub_i in enumerate(SOI[0]):

        if BehavioralData['BDI'][sub_i] < 10:

            Sub_G[0].append([i, sub_i])

        else:

            Sub_G[1].append([i, sub_i])

    if Source == 'CTRL':

        return [Sub_G[0]]
    
    elif Source == 'DEP':

        return [Sub_G[1]]
    
    elif Source == 'Both':

        return Sub_G
    
    else:
        
        return Sub_G

def HandleWin(window_str):
    
    p_num = 0

    wins = ['', '']

    for ch in window_str:

        if ch != ' ' and ch != '-':

            wins[p_num] = wins[p_num] + ch

        else:

            p_num = 1

    return [int(win) for win in wins]

def HandleClusterStr(Clusters_str, Network):

    Clusters_inv = []

    ActClusters = LDC.names['JulyClusterNames']

    NOI = LDC.NetworksOfInterest[Network]
    ActClusters = [ActClusters[NOI_mem] for NOI_mem in NOI]

    for Cluster_str in Clusters_str:

        if Cluster_str in ActClusters:

            Clusters_inv.append(np.where([Cluster_str == ActCluster for ActCluster in ActClusters])[0][0])

    assert len(Clusters_inv) > 0, "Not Enough Cluster/Channels"

    return Clusters_inv

def HandleHypo(Hypothesis):

    ALTs = ['less', 'All', 'greater']

    if ('Inc.' in Hypothesis) or ('Lag.' in Hypothesis) or ('LGR' in Hypothesis):

        return ALTs[2]
    
    elif ('Dec.' in Hypothesis) or ('Lead.' in Hypothesis) or ('SMR' in Hypothesis):

        return ALTs[0]
    
    else:

        return ALTs[1]
    
def HandleNetworkData(ConDataDict, EVENT, SOI, Subs, Channels):

    if EVENT == 'Stim': # For TE the first Channel is Transmitter and the other one the Receiver? and How About LRB_GC?

        NetworkData = [[[ConDataDict[str(Subs[1][SOI[G_i][i][0]])][block, :, Channels[0], Channels[1]] for block in range(2)] for i in range(len(SOI[G_i]))] for G_i in range(len(SOI))]

    elif EVENT == 'PosNeg':

        events = ['Pos', 'Neg']

        NetworkData = [[[ConDataDict[event][str(Subs[1][SOI[G_i][i][0]])][:, Channels[0], Channels[1]] for event in events] for i in range(len(SOI[G_i]))] for G_i in range(len(SOI))]

    return NetworkData

