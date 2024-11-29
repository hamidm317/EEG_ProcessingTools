import numpy as np

class DC_Constants():

    Properties = {

        'PLI':{

            'directed': False,
            'lagged': False,
            'AvailableBands': ['All', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
            'SelfLoop': False,
        },

        'PLV':{

            'directed': False,
            'lagged': False,
            'AvailableBands': ['All', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
            'SelfLoop': False,
        },

        'dPLI':{

            'directed': True,
            'lagged': False,
            'AvailableBands': ['All', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
            'SelfLoop': False,
        },

        'wPLI':{

            'directed': False,
            'lagged': False,
            'AvailableBands': ['All', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
            'SelfLoop': False,
        },

        'LRB_GC':{
            
            'directed': True,
            'lagged': True,
            'AvailableBands': ['All'],
            'SelfLoop': False,

        },

        # 'PIB_GC':{
            
        #     'directed': True,
        #     'lagged': True,
            # 'AvailableBands': ['All']

        # },

        'PAC':{
            
            'directed': True,
            'lagged': False,
            'AvailableBands': ['All'],
            'SelfLoop': True,

        },

        'TE':{
            
            'directed': True,
            'lagged': True,
            'AvailableBands': ['All'],
            'SelfLoop': False,

        },

        'PCor':{

            'directed': False,
            'lagged': False,
            'AvailableBands': ['All', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
            'SelfLoop': False,
                        
        },

        'DCG_PAC':{
            
            'directed': True,
            'lagged': False,
            'AvailableBands': ['All'],
            'SelfLoop': True,

        },

    }

class KernelConstants():

    DistanceKernels = {

        'KullbackLeibler': {

            'Directed': True,
            'Deterministic': False,
            'HypoTestLoopLength': 1,

        },

        'ShannonEntropy': {

            'Directed': False,
            'Deterministic': True,
            'HypoTestLoopLength': 1,

        }
    },

    CorrelationKernels = {

        'NumberOfBins': 9,
    }

class LocalDataConstants():

    directories = {

        'eeg_file_dir': {
            
            'July': {
                
                'Others': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\Clustered_SingleTrialData_All_Neg_Pos_Stim.mat',
                'Actions': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\Clustered_SingleTrialData_Action.mat',

            },

            'September':{'All': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\All_data_AllSepMatClustered_SingleTrialData_All_Neg_Pos_Stim.mat',
                          'Pos': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\All_data_NegSepMatClustered_SingleTrialData_All_Neg_Pos_Stim.mat',
                          'Neg': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\All_data_PosSepMatClustered_SingleTrialData_All_Neg_Pos_Stim.mat',
                          'Stim': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\All_data_StimSepMatClustered_SingleTrialData_All_Neg_Pos_Stim.mat'},

            'October':{

                'All': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\All_data_All_OctMatClustered_SingleTrialData.mat',
                'Pos': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\All_data_Pos_OctMatClustered_SingleTrialData.mat',
                'Neg': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\All_data_Neg_OctMatClustered_SingleTrialData.mat',
                'Stim': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\All_data_Stim_OctMatClustered_SingleTrialData.mat',
                'Actions': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\All_data_Actions_OctMatClustered_SingleTrialData.mat',
                'TestStim': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\All_data_TestStim_OctMatClustered_SingleTrialData.mat',
                'TestActions': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\All_data_TestActions_OctMatClustered_SingleTrialData.mat',
                'AB_FB': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\All_data_AB_FB_OctMatClustered_SingleTrialData.mat',
                'CD_FB': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\All_data_CD_FB_OctMatClustered_SingleTrialData.mat',
                'EF_FB': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\All_data_EF_FB_OctMatClustered_SingleTrialData.mat',
            },

        },

        'ActionDataLengthsDir': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\data_lengths_Action.mat',
        'TestActionsDataLengthsDir': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\TestActions_data_lengths.mat',
        'TestStimDataLengthsDir': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\TestStim_data_lengths.mat',
        'DataLengthsDir': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\data_lengths.mat',
        'AB_FBDataLengthsDir': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\data_lengths_AB_FB.mat',
        'CD_FBDataLengthsDir': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\data_lengths_CD_FB.mat',
        'EF_FBDataLengthsDir': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\data_lengths_EF_FB.mat',

        'beh_dir_file': r'E:\HWs\Msc\Research\Research\Depression Dataset\depression_rl_eeg\Depression PS Task\Scripts from Manuscript\Data_4_Import.xlsx',
        'perform_data_dir': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\Subjects_Behavioral_datas.csv',
        'eeg_prep_datasets_dir': r'E:\\HWs\Msc\\Research\\Research\\Depression Dataset\\Testing Preprocess',
        'confile_dir': r'D:\AIRLab_Research\Data\ConnectivityDataDict.pickle',
        'n_confile_dir': r'D:\AIRLab_Research\Data',
        'plotSave_dir': r'D:\AIRLab_Research\Plots',
        'ListOfAvailableSubjects': r'D:\AIRLab_Research\Data\BehavioralData\AvailableSubjects.npy',
        'fd_excel_dir': r'D:\AIRLab_Research\Features\FeatureDraft.xlsx'
    }

    names = {

        'JulyClusterNames': ['PF', 'LF', 'RF', 'MFC', 'LT', 'RT', 'LFC', 'RFC', 'MPC', 'LPC', 'RPC', 'MP', 'LPO', 'RPO'],
        'SeptemberClusterNames': ['FPz', 'AF3', 'AF4', 'Fz', 'FCz', 'CPz', 'Pz', 'P3', 'P4', 'POz'],
        'OctoberClusterNames': ['FPz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'Oz'],
        'freq_bands': ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'LowBeta', 'HighBeta', 'LowGamma', 'MidGamma', 'HighGamma', 'All'],
        'events': ['All', 'Neg', 'Pos', 'Stim', 'Actions', 'TestStim', 'TestActions', 'AB_FB', 'CD_FB', 'EF_FB'],
        'LocalCM':{

            'Transfer Entropy': 'TE',
            'PLI': 'PLI',
            'Granger Causality': 'LRB_GC',
            'dPLI': 'dPLI',
            'wPLI': 'wPLI',
            'PLV': 'PLV',

            'Stimulus': 'Stim',
            'Stim': 'Stim',
            'PosNeg': 'PosNeg',
            'Feedback': 'All',
            'Action': 'Action',
            'All': 'All',

            'Wavelet': 'WavletSpectralDecomposer',
            'WavletSpectralDecomposer': 'WavletSpectralDecomposer',
            'RidRihaczek': 'RidRihaczek',


        }
    }

    NetworksOfInterest = {

        'July':{

            'All': np.arange(14),
            'ZeroAxis': [0, 3, 8, 11],
            'Frontal': [0, 1, 2, 3],
            'OcciTemporal': [4, 5, 12, 13],
            'FrontoParietal': [0, 11],
        
        },

        'September':{

            'All': np.arange(10),
            'FrontoParietal_1': [1, 2, 3, 6, 7, 8],
            'FzPzPair': [3, 6],
        
        },

        'October':{

            'All': np.arange(17),
            'ZeroAxis': [0, 3, 8, 13, 16],
            'Frontal': [0, 2, 3, 4],
            'FrontoPar': [0, 2, 3, 4, 13],
            'FzPz': [3, 13],
        
        },

    }

    Labels = {

        'groups': ['Control', 'Depressed'],
        'data_block': ['Block 1', 'Block 2']
    }

    DefaulValues = {

        'Fs': 500,

        'overlap_ratio': 0.98,
        'window_length': 100,
        'trial_in_block': 10, # -> In Stim Locked Analyses
        'min_trial': 20, # -> In PosNeg Locked Analyses

        'Circuit':{

            'ZeroPoint': 25,
            'Fs': 250
        },

        'Univariate':{

            'ZeroPoint': 100,
            'Fs': 500
        },

        'AvailableTimePos': ['Start', 'Middle', 'End'],

    }

class SpectralConstants():

    BandsBounds = {

        'Delta': [0.5, 4],
        'Theta': [4, 8],
        'Alpha': [8, 12],
        'Beta': [12, 30],
        'Gamma': [30, 50],
        'LowBeta': [12, 20],
        'HighBeta': [20, 30],
        'LowGamma': [30, 38],
        'MidGamma': [38, 44],
        'HighGamma': [44, 50],
        'All': [0.5, 50] # Keep 'All' the last key!

    }

    WaveletParams = {

        'wavelet': 'morl',
        'widths_param':{

            'morl': {

                '500': {

                    'All': [8, 1024],
                    'Delta': [128, 1024],
                    'Theta': [54, 128],
                    'Alpha': [32, 54],
                    'Beta': [13, 32],
                    'Gamma': [8, 14],
                    'LowBeta': [20, 32],
                    'HighBeta': [12, 20],
                    'LowGamma': [11, 14],
                    'MidGamma': [9.2, 11],
                    'HighGamma': [8, 9.2],
                },
            },

            'cmor': {

                '500': {

                    'All': [8, 1024],
                    'Delta': [128, 1024],
                    'Theta': [54, 128],
                    'Alpha': [32, 54],
                    'Beta': [13, 32],
                    'Gamma': [8, 14],
                    'LowBeta': [20, 32],
                    'HighBeta': [12, 20],
                    'LowGamma': [11, 14],
                    'MidGamma': [9.2, 11],
                    'HighGamma': [8, 9.2],
                },
            }

        },

        'time_lims':{

            '100': [0, 0.2],
            '200': [0, 0.4],
            '400': [-0.2, 0.6],
            '500': [-0.4, 0.6],
            '800': [-0.4, 1.2],
            'Default': 'No'

        },

        'Spectral_Res': 20
    }

class StaConstants():

    Distributions = {

        'Availables': ['NormalPDF']

    }