import numpy as np
import matplotlib.pyplot as plt

import GRConnPy as GRC
from Utils import Local
from Utils import Constants

import pickle
import os

import scipy.stats as sps

from Utils import SciPlot as SP

########################################################### Define Parameters ###########################################################

group_labels = Constants.LocalDataConstants.Labels['groups']
data_labels = Constants.LocalDataConstants.Labels['data_block']

ConKers = [kernel for kernel in Constants.DC_Constants.Properties.keys()]

trial_in_block = Constants.LocalDataConstants.DefaulValues['trial_in_block']
overlap_ratio = Constants.LocalDataConstants.DefaulValues['overlap_ratio']
win_length = Constants.LocalDataConstants.DefaulValues['window_length']

confile_dir = Constants.LocalDataConstants.directories['confile_dir']
PlotSave_dir = Constants.LocalDataConstants.directories['plotSave_dir']

Fs = 500

st = -0.2
ft = 0.6
sp = int((st + 0.4) * Fs)
fp = int((ft + 0.4) * Fs)

########################################################### Load Available Data ###########################################################

BehavioralData, Performance_data = Local.ExperimentDataLoader()
SOI = Local.AvailableSubjects()

if os.path.isfile(confile_dir):

    print("The Connectivity Data are available in a dictionary and is loaded")

    with open(confile_dir, 'rb') as f:

        ConDataDict = pickle.load(f)

    print("Dictionary Keys are: " + str(ConDataDict.keys()))

else:

    ConDataDict = {'FileAvailable': True}

    print("The Connectivity Data file not founded there, an empty dictionary is created.")

    with open(confile_dir, 'wb') as f:
        
        pickle.dump(ConDataDict, f, protocol=pickle.HIGHEST_PROTOCOL)

################################################## Divide Subject into DEP and CTRL Groups ###########################################################

Sub_G = [[], []] # first element is CTRL Group Members and the Second one the DEP Group

for i, sub_i in enumerate(SOI[0]):

    if BehavioralData['BDI'][sub_i] < 10:

        Sub_G[0].append([i, sub_i])

    else:

        Sub_G[1].append([i, sub_i])

########################################################## Generate Connectivity Data ###########################################################

event_numbers = [3] # Stim Onset

for event in event_numbers:

    raw_data, data_lengths = Local.ClusteredEEGLoader(event_number = event)
    
    event_name = Constants.LocalDataConstants.names['events'][event]
    print("The Event is " + event_name)

    ConDataDict[event_name] = {}

    for kernel in ConKers:

        if kernel in ConDataDict[event_name].keys():

            print("The " + kernel + " data is available")

            if 'WL' + str(int(win_length)) in ConDataDict[event_name][kernel].keys():

                print("The " + str(int(win_length)) + " data is available")

                if 'OR' + str(int(100 * overlap_ratio)) in ConDataDict[event_name][kernel]['WL' + str(int(win_length))].keys():

                    print("The " + str(int(100 * overlap_ratio)) + " data is available")

        else:

            ConDataDict[event_name][kernel] = {} 
            ConDataDict[event_name][kernel]['WL' + str(int(win_length))] = {}
            ConDataDict[event_name][kernel]['WL' + str(int(win_length))]['OR' + str(int(100 * overlap_ratio))] = {}

            print("The record added to dictionary")

        for i, sub_i in enumerate(SOI[0]):

            print("subject " + str(i))

            dl = int(data_lengths[i])
            data = sps.zscore(raw_data[i][:, :, sp : fp], axis = -1)
            divData = np.array([np.mean(data[:trial_in_block, :, :], axis = 0), np.mean(data[dl - trial_in_block : dl, :, :], axis = 0)])
            
            specs = {
                
                'inc_channels': np.arange(data.shape[1]),
                'orders_matrix': 12,
                'd_x': 12,
                'd_y': 12,
                'w_x': 1,
                'w_y': 1

            }

            sub_Data = GRC.DynamicConnectivityMeasure(divData, kernel = kernel, overlap_ratio = overlap_ratio, window_length = win_length, orders_matrix = 12)
            ConDataDict[event_name][kernel]['WL' + str(int(win_length))]['OR' + str(int(100 * overlap_ratio))][str(SOI[1][i])] = sub_Data

        ConDataDict[event_name][kernel]['WL' + str(int(win_length))]['OR' + str(int(100 * overlap_ratio))]['specs'] = specs

        ########################## Uncomment Following Part if you are in doubt about how to save file! #################################

        # with open(confile_dir, 'rb') as f:

        #     tmpDataDict = pickle.load(f)

        # if tmpDataDict != ConDataDict:

        #     print("Do you want to overwrite the data?")

            # ans = input()

            # if ans == 'y':

        with open(confile_dir, 'wb') as f:
        
            pickle.dump(ConDataDict, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("File Saved")

################################################## Define time vector to plot figures ###########################################################

time_p = np.linspace(st + win_length / (Fs * 2), ft - win_length / (Fs * 2), int(np.ceil(((ft - st) * Fs - win_length) / (win_length * (1- overlap_ratio)))))

######################################################## Plot and Save Data! ###########################################################

NOIs = [Network for Network in Constants.LocalDataConstants.NetworksOfInterest.keys()] # Networks Of Interest!

event_name = Constants.LocalDataConstants.names['events'][3]

for kernel in ConKers:

    for Network in NOIs:

        Channels = Constants.LocalDataConstants.NetworksOfInterest[Network]

        NetworkData = [[[ConDataDict[event_name][kernel]['WL100']['OR98'][str(SOI[1][Sub_G[G_i][i][0]])][block, :, :, :][:, Channels, :][:, :, Channels] for i in range(len(Sub_G[G_i]))] for block in range(2)] for G_i in range(len(Sub_G))]

        for a, Channel_a in enumerate(Channels):

            if Constants.DC_Constants.Properties[kernel]['directed']:

                Channels_SL = Channels

            else:

                Channels_SL = Channels[:a]

            for b, Channel_b in enumerate(Channels_SL):

                Data2Plot = [[np.array(NetworkData[G_i])[Block, :, :, a, b] for Block in range(len(data_labels))] for G_i in range(len(group_labels))]

                if a != b:

                    fig, ax = plt.subplots(2, 2, layout = 'constrained', sharey = True, sharex = True, figsize = (10, 7))

                    for i, Data_G in enumerate(Data2Plot):

                        for j in range(len(Data_G)):

                            Data_B = Data_G[j]
                            y_U, y_L = SP.ConfidenceBoundsGen(np.array(Data_B))
                            ax[i, 0].plot(time_p, np.mean(Data_B, axis = 0), label = data_labels[j])
                            ax[i, 0].fill_between(time_p, y_U, y_L, alpha = 0.2)
                        
                        ax[i, 0].set_title(group_labels[i])
                        ax[i, 0].axvline(0, color = 'k')

                    ax[0, 0].legend()

                    for i, B_L in enumerate(data_labels):

                        for j, G_l in enumerate(group_labels):

                            Data = Data2Plot[j][i]
                            y_U, y_L = SP.ConfidenceBoundsGen(np.array(Data))
                            ax[i, 1].plot(time_p, np.mean(Data, axis = 0), label = G_l)
                            ax[i, 1].fill_between(time_p, y_U, y_L, alpha = 0.2)
                        
                        ax[i, 1].set_title(B_L)
                        ax[i, 1].axvline(0, color = 'k')

                    ax[0, 1].legend()

                    plt.setp(ax, xlim = [time_p[0], time_p[-1]])
                    fig.supxlabel("time (s)")

                    if kernel == 'PLI':

                        fig.suptitle("Temporal dynamic of " + str(kernel) + " locked on " + event_name + " Onset\nClusters " + Constants.LocalDataConstants.names['JulyClusterNames'][Channel_a] + " and " + Constants.LocalDataConstants.names['JulyClusterNames'][Channel_b], fontsize = 15)
                        plt.savefig(PlotSave_dir + "\\" + kernel + "\\" + Network + "\\" + event_name + "\defParam_" + Constants.LocalDataConstants.names['JulyClusterNames'][Channel_a] + "_" + Constants.LocalDataConstants.names['JulyClusterNames'][Channel_b] + ".png", format="png")
                        
                        print("Saved")

                    elif kernel == 'TE':

                        fig.suptitle("Temporal dynamic of " + str(kernel) + " locked on " + event_name + " Onset\nTransmitter is " + Constants.LocalDataConstants.names['JulyClusterNames'][Channel_a] + " and Receiver is " + Constants.LocalDataConstants.names['JulyClusterNames'][Channel_b], fontsize = 15)
                        fig.savefig(PlotSave_dir + "\\" + kernel + "\\" + Network + "\\" + event_name + "\defParam_Tr_" + Constants.LocalDataConstants.names['JulyClusterNames'][Channel_a] + "_Rec_" + Constants.LocalDataConstants.names['JulyClusterNames'][Channel_b] + ".png", format="png")

                    elif kernel == 'LRB_GC':

                        fig.suptitle("Temporal dynamic of " + str(kernel) + " locked on " + event_name + " Onset\nTransmitter is " + Constants.LocalDataConstants.names['JulyClusterNames'][Channel_b] + " and Receiver is " + Constants.LocalDataConstants.names['JulyClusterNames'][Channel_a], fontsize = 15)
                        fig.savefig(PlotSave_dir + "\\" + kernel + "\\" + Network + "\\" + event_name + "\defParam_Tr_" + Constants.LocalDataConstants.names['JulyClusterNames'][Channel_b] + "_Rec_" + Constants.LocalDataConstants.names['JulyClusterNames'][Channel_a] + ".png", format="png")
                    
                    plt.close(fig)  
                    # plt.show()              