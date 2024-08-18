import numpy as np

class DC_Constants():

    Properties = {

        'PLI':{

            'directed': False,
            'lagged': False
        },

        'LRB_GC':{
            
            'directed': True,
            'lagged': True

        },

        'PIB_GC':{
            
            'directed': True,
            'lagged': True

        },

        'PAC':{
            
            'directed': True,
            'lagged': False

        },

        'TE':{
            
            'directed': True,
            'lagged': True

        }
    }

class LocalDataConstants():

    directories = {

        'eeg_file_dir': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\Clustered_SingleTrialData_All_Neg_Pos_Stim.mat',
        'beh_dir_file': r'E:\HWs\Msc\Research\Research\Depression Dataset\depression_rl_eeg\Depression PS Task\Scripts from Manuscript\Data_4_Import.xlsx',
        'perform_data_dir': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\Subjects_Behavioral_datas.csv',
        'eeg_prep_datasets_dir': r'E:\\HWs\Msc\\Research\\Research\\Depression Dataset\\Testing Preprocess',
        'confile_dir': r'D:\AIRLab_Research\Data\ConnectivityDataDict.pickle',
        'plotSave_dir': r'D:\AIRLab_Research\Plots',
        'fd_excel_dir': r'D:\AIRLab_Research\Features\FeatureDraft.xlsx'
    }

    names = {

        'JulyClusterNames': ['PF', 'LF', 'RF', 'MFC', 'LT', 'RT', 'LFC', 'RFC', 'MPC', 'LPC', 'RPC', 'MP', 'LPO', 'RPO'],
        'freq_bands': ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
        'events': ['All', 'Neg', 'Pos', 'Stim'],
        'LocalCM':{

            'Transfer Entropy': 'TE',
            'PLI': 'PLI',
            'Granger Causality': 'LRB_GC',
            'Stimulus': 'Stim'

        }
    }

    NetworksOfInterest = {

        'ZeroAxis': [0, 3, 8, 11],
        'Frontal': [0, 1, 2, 3],
        'OcciTemporal': [4, 5, 12, 13]

    }

    Labels = {

        'groups': ['Control', 'Depressed'],
        'data_block': ['Block 1', 'Block 2']
    }

    DefaulValues = {

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

                '400': {

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

                '800': {

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
                }

            }

        },

        'time_lims':{

            '400': [-0.2, 0.6],
            '800': [-0.4, 1.2]

        },

        'Spectral_Res': 100
    }