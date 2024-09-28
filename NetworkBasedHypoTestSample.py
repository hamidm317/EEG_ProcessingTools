import numpy as np
import matplotlib.pyplot as plt

from Utils.Constants import DC_Constants
from Utils.Constants import LocalDataConstants as LDC
from Kernels import ConKernels, OrderKernels
import Modules.GRConnPy as GRC
from Utils import Local

import scipy.stats as sps
from Utils import SciPlot as SP

from Kernels import FeatureKernels
from Utils import HypoExcelTools as HT

import pandas as pd
import pickle

import warnings

warnings.filterwarnings('ignore') 

Subs = Local.AvailableSubjects()

FeatureDraft = pd.read_excel(LDC.directories['fd_excel_dir'], sheet_name = 'Classified')

IgnoreSoAnAnt = True
IgnoreAlter = True
PrEvent = ''
PrBand = ''
CDD_Av = False
PrSO = ''

FeatureCriterion = 'Disc' # Must Be handled Row-By-Row

HypSent = [
    'Mode 2 c.w. Mode 1 for CTRL ', 
    'DEP c.w. CTRL in Mode 1 ',
    'DEP c.w. CTRL in Mode 2 ',
    'Mode 2 vs. Mode 1 for DEP '
]

ALTs = ['less', 'two-sided', 'greater']

for Hypothesis_Num in range(len(FeatureDraft)):

    if FeatureDraft['CoCoSt'][Hypothesis_Num] in ['NAY', 'CIL']:

        print(Hypothesis_Num)

        EVENT = FeatureDraft['EVENT'][Hypothesis_Num]

        ORIGIN = FeatureDraft['ORIGIN'][Hypothesis_Num]
        SUBORIGIN = FeatureDraft['SUBORIGIN'][Hypothesis_Num]

        Band = FeatureDraft['Band'][Hypothesis_Num]

        Network = FeatureDraft['Network'][Hypothesis_Num]
        
        # if ~IgnoreSoAnAnt:

        #     Source = FeatureDraft['Source'][Hypothesis_Num]
        #     Antagonist = FeatureDraft['Antagonist'][Hypothesis_Num]

        # else:

        #     Source = FeatureDraft['Source'][Hypothesis_Num]
        #     Antagonist = FeatureDraft['Antagonist'][Hypothesis_Num]

        Clusters_inv = HT.HandleClusterStr([FeatureDraft['Clusters_A'][Hypothesis_Num], FeatureDraft['Clusters_B'][Hypothesis_Num]], Network)

        if ~IgnoreAlter:

            # Alter = HT.HandleHypo(FeatureDraft['Feature'][Hypothesis_Num])
            Alter = 'All'

        else:

            Alter = 'All'

        FeatureMetric = FeatureDraft['Type'][Hypothesis_Num]

        if FeatureDraft['Window'][Hypothesis_Num] != '?' and FeatureMetric != 'VAGUE':

            window = HT.HandleWin(FeatureDraft['Window'][Hypothesis_Num])

            if ORIGIN == 'Univariate': # in the next STEP summarize this part into a Local Function!

                if PrEvent != EVENT:

                        raw_data, data_lengths = Local.ClusteredEEGLoader(EVENT)
                        PrEvent = EVENT

                Channel_A = Clusters_inv[0]
                SOI = HT.HandleSubjects(Source = 'Both', Antagonist = '')

                SubORIGINExtractor = getattr(FeatureKernels, ORIGIN + '_' + SUBORIGIN + '_' + EVENT)
                Data_CKD = SubORIGINExtractor(raw_data, data_lengths, SOI, Channel_A, Band)

            elif ORIGIN == 'Circuit':

                ConDataDict = Local.HandleDataLoad(LDC.directories['n_confile_dir'] + "\\" + LDC.names['LocalCM'][EVENT] + "\\" + Network + "\\" + LDC.names['LocalCM'][SUBORIGIN] + "\\" + Band)

                Channels = Clusters_inv
                SOI = HT.HandleSubjects(Source = 'Both', Antagonist = '')
                Data_CKD = HT.HandleNetworkData(ConDataDict, EVENT, SOI, Subs, Channels) # is it possible to handle it with getattr?

            elif ORIGIN == 'Network':

                print("UC")
                FeatureDraft['CoCoSt'][Hypothesis_Num] = 'CIL'
                # TestResultList.append([])   

            else:

                print("Invalid Event in Hypothesis " + Hypothesis_Num) 
                # TestResultList.append('InvalidEvent')
                break
            
            FeatureExtractor = getattr(FeatureKernels, FeatureMetric)
            Feature = FeatureExtractor(Data_CKD, window, specs = LDC.DefaulValues[ORIGIN])

            FeatureTest = getattr(FeatureKernels, FeatureCriterion)
            TestResults = FeatureTest(Feature, ALT = Alter)

            # TestResultList.append(TestResults)

            for Hi, Hyp in enumerate(HypSent):

                for ALi, ALTa in enumerate(ALTs):

                    FeatureDraft[Hyp + ALTa][Hypothesis_Num] = TestResults[ALi][int(Hi / 2), np.mod(Hi, 2)]
            
            FeatureDraft['CoCoSt'][Hypothesis_Num] = 'DBC'

        else:

            # TestResultList.append('Not Classified')
            print('Not Classified')
            FeatureDraft['CoCoSt'][Hypothesis_Num] = 'CIL'

with pd.ExcelWriter(LDC.directories['fd_excel_dir'], mode='a', if_sheet_exists = 'replace') as writer:
    
    FD_pd = pd.DataFrame(FeatureDraft)
    FD_pd.to_excel(writer, sheet_name='ComMan')