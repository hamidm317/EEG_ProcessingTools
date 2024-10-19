import numpy as np
import matplotlib.pyplot as plt

from Utils import Local
from Utils import Constants

from Utils import SciPlot as SP

from Utils.InputVariables import PlotVars as PV
from Utils.InputVariables import CommonVars as CoV

########################################################### Define Parameters ###########################################################

DataName = PV.NetBaseConnPlot['DataName']
NodeNames = Constants.LocalDataConstants.names[DataName + 'ClusterNames']

group_labels = Constants.LocalDataConstants.Labels['groups']
data_labels = ['Pos', 'Neg']

confile_dir = Constants.LocalDataConstants.directories['n_confile_dir']
PlotSave_dir = Constants.LocalDataConstants.directories['plotSave_dir']

ConKers = PV.NetBaseConnPlot['ConKers']

NOIs = PV.NetBaseConnPlot['NOIs']
Bands = PV.NetBaseConnPlot['Bands']

VerNum = PV.NetBaseConnPlot['VersionNumber']

Fs = CoV.SamplingFrequency

NB = 2

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

events = ['Pos', 'Neg']
    
event_name = 'PosNeg'

print("The Event is " + event_name)

for NOI in NOIs:

    for kernel in ConKers:

        for Band_i, Band in enumerate(Bands):

            if Local.BandAvailable(kernel, Band):

                Data = []
                VersionSpecs = []

                for event in events:

                    Data_t, VersionSpecs_t = Local.HandleDataLoad(confile_dir + '\\' + event + '\\' + NOI + '\\' + kernel + '\\' + Band, version_number = VerNum)

                    Data.append(Data_t)

                    VersionSpecs.append(VersionSpecs_t)

                assert VersionSpecs[0] == VersionSpecs[1], "Data's Version Specs Must be Matched!"

                VersionSpecs = VersionSpecs[0]

                st = VersionSpecs['Start Time']
                ft = VersionSpecs['End Time']

                win_length = VersionSpecs['Window_Length']
                overlap_ratio = VersionSpecs['Overlap_Ratio']

                time_p = SP.TimeVectorGenerator(st, ft, Fs, win_length, overlap_ratio, TimePos = PV.NetBaseConnPlot['TimePosition'])

                DataGroups = [[[Data[FB_Sign][str(SOI[1][sub_i[0]])] for FB_Sign in range(2)] for sub_i in SubSub_G] for SubSub_G in Sub_G]

                Channels = Constants.LocalDataConstants.NetworksOfInterest[DataName][NOI]

                for Tr_i, TrCh in enumerate(Channels):

                    if Constants.DC_Constants.Properties[kernel]['directed']:


                        SLChannels = Channels

                    else:

                        SLChannels = Channels[:Tr_i]


                    for Re_i, ReCh in enumerate(SLChannels):

                        if TrCh != ReCh or Constants.DC_Constants.Properties[kernel]['SelfLoop']:

                            Data4Plot = [np.array(DataGroups[G_i])[:, :, :, Tr_i, Re_i] for G_i in range(len(DataGroups))]

                            fig, axs = plt.subplots(2, 2, layout = 'constrained', sharey = True, sharex = True, figsize = (10, 7))

                            for Gi, DataGroup in enumerate(Data4Plot):

                                for BN in range(NB):

                                    DataBlock = DataGroup[:, BN, :]
                                    y_U, y_L = SP.ConfidenceBoundsGen(DataBlock)

                                    axs[Gi, 0].plot(time_p[:DataBlock.shape[-1]], np.mean(DataBlock, axis = 0), label = data_labels[BN])
                                    axs[Gi, 0].fill_between(time_p[:DataBlock.shape[-1]], y_U, y_L, alpha = 0.2)

                                    axs[BN, 1].plot(time_p[:DataBlock.shape[-1]], np.mean(DataBlock, axis = 0), label = group_labels[Gi])
                                    axs[BN, 1].fill_between(time_p[:DataBlock.shape[-1]], y_U, y_L, alpha = 0.2)

                            

                            for i in range(2):

                                axs[i, 0].set_title(group_labels[i])
                                axs[i, 1].set_title(data_labels[i])

                                for j in range(2):

                                    axs[i, j].legend()

                            plt.setp(axs, xlim = [time_p[0], time_p[:DataBlock.shape[-1]][-1]])
                            fig.supxlabel("time (s)")

                            if DataName != 'July':

                                SavePlotDir = Local.HandleDir(PlotSave_dir + '\\' + event_name + '\\' + NOI + '\\' + DataName + '\\' + kernel + '\\' + Band)

                            else:
                                
                                SavePlotDir = Local.HandleDir(PlotSave_dir + '\\' + event_name + '\\' + NOI + '\\' + kernel + '\\' + Band)

                            if not Constants.DC_Constants.Properties[kernel]['directed']:

                                fig.suptitle("Temporal dynamic of " + str(kernel) + " locked on " + event_name + " Onset\nClusters " + NodeNames[TrCh] + " and " + NodeNames[ReCh] + " in " + Band + " Band", fontsize = 15)
                                plt.savefig(SavePlotDir + "\defParam_" + NodeNames[TrCh] + "_" + NodeNames[ReCh] + ".png", format="png")
                                
                                print("Saved")

                            else:

                                fig.suptitle("Temporal dynamic of " + str(kernel) + " locked on " + event_name + " Onset\nTransmitter is " + NodeNames[TrCh] + " and Receiver is " + NodeNames[ReCh] + " in " + Band + " Band", fontsize = 15)
                                fig.savefig(SavePlotDir + "\defParam_Tr_" + NodeNames[TrCh] + "_Rec_" + NodeNames[ReCh] + ".png", format="png")

                                print("Saved")

                            plt.close(fig)