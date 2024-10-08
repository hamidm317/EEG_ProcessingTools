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

            'directed': True,
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

        }
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
    }

class LocalDataConstants():

    directories = {

        'eeg_file_dir': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\Clustered_SingleTrialData_All_Neg_Pos_Stim.mat',
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
        'freq_bands': ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'LowBeta', 'HighBeta', 'LowGamma', 'MidGamma', 'HighGamma', 'All'],
        'events': ['All', 'Neg', 'Pos', 'Stim'],
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

        'All': np.arange(14),
        'ZeroAxis': [0, 3, 8, 11],
        'Frontal': [0, 1, 2, 3],
        'OcciTemporal': [4, 5, 12, 13]

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