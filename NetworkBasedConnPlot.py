import numpy as np
import matplotlib.pyplot as plt

from Utils import Local
from Utils import Constants

from Utils import SciPlot as SP

########################################################### Define Parameters ###########################################################

group_labels = Constants.LocalDataConstants.Labels['groups']
data_labels = Constants.LocalDataConstants.Labels['data_block']

# ConKers = [kernel for kernel in Constants.DC_Constants.Properties.keys()]
ConKers = ['PLV']

trial_in_block = Constants.LocalDataConstants.DefaulValues['trial_in_block']
overlap_ratio = Constants.LocalDataConstants.DefaulValues['overlap_ratio']
win_length = Constants.LocalDataConstants.DefaulValues['window_length']

confile_dir = Constants.LocalDataConstants.directories['n_confile_dir']
PlotSave_dir = Constants.LocalDataConstants.directories['plotSave_dir']

NOIs = [Network for Network in Constants.LocalDataConstants.NetworksOfInterest.keys()] # Networks Of Interest!
Bands = Constants.LocalDataConstants.names['freq_bands']

Fs = 500

st = -0.2
ft = 0.6
sp = int((st + 0.4) * Fs)
fp = int((ft + 0.4) * Fs)

NB = 2
TB = trial_in_block = Constants.LocalDataConstants.DefaulValues['trial_in_block']

########################################################### Load Available Data ###########################################################

BehavioralData, Performance_data = Local.ExperimentDataLoader()
SOI = Local.AvailableSubjects()

################################################## Divide Subject into DEP and CTRL Groups ###########################################################

Sub_G = [[], []] # first element is CTRL Group Members and the Second one the DEP Group

for i, sub_i in enumerate(SOI[0]):

    if BehavioralData['BDI'][sub_i] < 10:

        Sub_G[0].append([i, sub_i])

    else:

        Sub_G[1].append([i, sub_i])

########################################################## Generate Connectivity Data ###########################################################

event_numbers = [3] # Stim Onset

time_p = np.linspace(st + win_length / (Fs * 2), ft - win_length / (Fs * 2), int(np.ceil(((ft - st) * Fs - win_length) / (win_length * (1- overlap_ratio)))))

for event in event_numbers:
    
    event_name = Constants.LocalDataConstants.names['events'][event]
    print("The Event is " + event_name)

    for NOI in NOIs:

        for kernel in ConKers:

            for Band_i, Band in enumerate(Bands):

                if Local.BandAvailable(kernel, Band):

                    Data = Local.HandleDataLoad(confile_dir + '\\' + event_name + '\\' + NOI + '\\' + kernel + '\\' + Band)

                    DataGroups = [[Data[str(SOI[1][sub_i[0]])] for sub_i in SubSub_G] for SubSub_G in Sub_G]

                    Channels = Constants.LocalDataConstants.NetworksOfInterest[NOI]

                    for Tr_i, TrCh in enumerate(Channels):

                        if Constants.DC_Constants.Properties[kernel]['directed']:


                            SLChannels = Channels

                        else:

                            SLChannels = Channels[:Tr_i]


                        for Re_i, ReCh in enumerate(SLChannels):

                            if TrCh != ReCh:

                                Data4Plot = [np.array(DataGroups[G_i])[:, :, :, Tr_i, Re_i] for G_i in range(len(DataGroups))]

                                fig, axs = plt.subplots(2, 2, layout = 'constrained', sharey = True, sharex = True, figsize = (10, 7))

                                for Gi, DataGroup in enumerate(Data4Plot):

                                    for BN in range(NB):

                                        DataBlock = DataGroup[:, BN, :]
                                        y_U, y_L = SP.ConfidenceBoundsGen(DataBlock)

                                        axs[Gi, 0].plot(time_p, np.mean(DataBlock, axis = 0), label = data_labels[BN])
                                        axs[Gi, 0].fill_between(time_p, y_U, y_L, alpha = 0.2)

                                        axs[BN, 1].plot(time_p, np.mean(DataBlock, axis = 0), label = group_labels[Gi])
                                        axs[BN, 1].fill_between(time_p, y_U, y_L, alpha = 0.2)

                                

                                for i in range(2):

                                    axs[i, 0].set_title(group_labels[i])
                                    axs[i, 1].set_title(data_labels[i])

                                    for j in range(2):

                                        axs[i, j].legend()

                                plt.setp(axs, xlim = [time_p[0], time_p[-1]])
                                fig.supxlabel("time (s)")

                                SavePlotDir = Local.HandleDir(PlotSave_dir + '\\' + event_name + '\\' + NOI + '\\' + kernel + '\\' + Band)

                                if kernel == 'PLI':

                                    fig.suptitle("Temporal dynamic of " + str(kernel) + " locked on " + event_name + " Onset\nClusters " + Constants.LocalDataConstants.names['JulyClusterNames'][TrCh] + " and " + Constants.LocalDataConstants.names['JulyClusterNames'][ReCh] + " in " + Band + " Band", fontsize = 15)
                                    plt.savefig(SavePlotDir + "\defParam_" + Constants.LocalDataConstants.names['JulyClusterNames'][TrCh] + "_" + Constants.LocalDataConstants.names['JulyClusterNames'][ReCh] + ".png", format="png")
                                    
                                    print("Saved")

                                else:

                                    fig.suptitle("Temporal dynamic of " + str(kernel) + " locked on " + event_name + " Onset\nTransmitter is " + Constants.LocalDataConstants.names['JulyClusterNames'][TrCh] + " and Receiver is " + Constants.LocalDataConstants.names['JulyClusterNames'][ReCh] + " in " + Band + " Band", fontsize = 15)
                                    fig.savefig(SavePlotDir + "\defParam_Tr_" + Constants.LocalDataConstants.names['JulyClusterNames'][TrCh] + "_Rec_" + Constants.LocalDataConstants.names['JulyClusterNames'][ReCh] + ".png", format="png")

                                    print("Saved")

                                plt.close(fig)