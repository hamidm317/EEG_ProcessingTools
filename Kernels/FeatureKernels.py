import numpy as np

def Univariate_ERP_Stim(Data, data_lengths, SOI, Channel, Band, specs = {'NumBlock': 2, 'NumSample_inBlock': 10, 'StartPoint': 100, 'FinalPoint': 500}):
    
    import scipy.stats as sps
    from Utils import SciPlot as SP

    NB = specs['NumBlock']
    NSiB = specs['NumSample_inBlock']

    sp = specs['StartPoint']
    fp = specs['FinalPoint']

    Data_ERP = [[np.mean(sps.zscore(Data[sub_i[0]][SP.DeterminedBlockSampling(Length = int(data_lengths[sub_i[0]]), NumBlock = NB, NumSample_inBlock = NSiB), Channel, :], axis = -1), axis = 1)[:, sp : fp] for sub_i in SOI[G_i]] for G_i in range(len(SOI))]

    if Band == 'All':

        return Data_ERP
    
    else:

        from Utils.Constants import SpectralConstants as SC
        from Modules import GRUniPy as GRU

        assert Band in SC.BandsBounds.keys(), "Band is not DEFINED!"

        BandDecompERP = [[[np.squeeze(GRU.FrequencyBandExt(Data_ERP[G_i][sub_i][Trial], Band = Band)) for Trial in range(NB)] for sub_i in range(len(SOI[G_i]))] for G_i in range(len(SOI))]

        return BandDecompERP
    
def Univariate_Power_Stim(Data, data_lengths, SOI, Channel, Band, specs = {'NumBlock': 2, 'NumSample_inBlock': 10, 'StartPoint': 100, 'FinalPoint': 500}):
    
    import scipy.stats as sps
    from Utils import SciPlot as SP
    from Modules import GRUniPy as GRU

    NB = specs['NumBlock']
    NSiB = specs['NumSample_inBlock']

    sp = specs['StartPoint']
    fp = specs['FinalPoint']

    if Band == 'All':

        Data_Amp = [[np.squeeze(GRU.PowerPhaseExt(np.mean(sps.zscore(Data[sub_i[0]][SP.DeterminedBlockSampling(Length = int(data_lengths[sub_i[0]]), NumBlock = NB, NumSample_inBlock = NSiB), Channel, :], axis = -1), axis = 1)[:, sp : fp], return_value = 'Power')) for sub_i in SOI[G_i]] for G_i in range(len(SOI))]

        return Data_Amp
    
    else:

        from Utils.Constants import SpectralConstants as SC

        assert Band in SC.BandsBounds.keys(), "Band is not DEFINED!"

        Data_ERP = [[np.mean(sps.zscore(Data[sub_i[0]][SP.DeterminedBlockSampling(Length = int(data_lengths[sub_i[0]]), NumBlock = NB, NumSample_inBlock = NSiB), Channel, :], axis = -1), axis = 1)[:, sp : fp] for sub_i in SOI[G_i]] for G_i in range(len(SOI))]
        
        BandDecompAmp = [[[np.squeeze(GRU.PowerPhaseExt(GRU.FrequencyBandExt(Data_ERP[G_i][sub_i][Trial], Band = Band), return_value = 'Power')) for Trial in range(NB)] for sub_i in range(len(SOI[G_i]))] for G_i in range(len(SOI))]

        return BandDecompAmp
    
def Univariate_Phase_Stim(Data, data_lengths, SOI, Channel, Band, specs = {'NumBlock': 2, 'NumSample_inBlock': 10, 'StartPoint': 100, 'FinalPoint': 500}):
    
    import scipy.stats as sps
    from Utils import SciPlot as SP
    from Modules import GRUniPy as GRU

    NB = specs['NumBlock']
    NSiB = specs['NumSample_inBlock']

    sp = specs['StartPoint']
    fp = specs['FinalPoint']

    if Band == 'All':

        Data_Amp = [[np.squeeze(GRU.PowerPhaseExt(np.mean(sps.zscore(Data[sub_i[0]][SP.DeterminedBlockSampling(Length = int(data_lengths[sub_i[0]]), NumBlock = NB, NumSample_inBlock = NSiB), Channel, :], axis = -1), axis = 1)[:, sp : fp], return_value = 'Phase')) for sub_i in SOI[G_i]] for G_i in range(len(SOI))]

        return Data_Amp
    
    else:

        from Utils.Constants import SpectralConstants as SC

        assert Band in SC.BandsBounds.keys(), "Band is not DEFINED!"

        Data_ERP = [[np.mean(sps.zscore(Data[sub_i[0]][SP.DeterminedBlockSampling(Length = int(data_lengths[sub_i[0]]), NumBlock = NB, NumSample_inBlock = NSiB), Channel, :], axis = -1), axis = 1)[:, sp : fp] for sub_i in SOI[G_i]] for G_i in range(len(SOI))]
        
        BandDecompAmp = [[[np.squeeze(GRU.PowerPhaseExt(GRU.FrequencyBandExt(Data_ERP[G_i][sub_i][Trial], Band = Band), return_value = 'Phase')) for Trial in range(NB)] for sub_i in range(len(SOI[G_i]))] for G_i in range(len(SOI))]

        return BandDecompAmp

def Average(Data, Win, specs = {'ZeroPoint': 100, 'Fs': 500}):

    ZP = specs['ZeroPoint']
    Fs = specs['Fs']

    SP = int(Win[0] * (Fs / 1000) + ZP)
    FP = int(Win[1] * (Fs / 1000) + ZP)

    Features_A = []

    for data_G in Data:

        data_G = np.array(data_G)

        BNs = np.arange(data_G.shape[1])

        Features_B = []

        for BN in BNs:

            Features_B.append(np.mean(data_G[:, BN, SP : FP], axis = -1))

        Features_A.append(np.array(Features_B).T)

    return Features_A

def MaximumValue(Data, Win, specs = {'ZeroPoint': 100, 'Fs': 500}):

    ZP = specs['ZeroPoint']
    Fs = specs['Fs']

    SP = int(Win[0] * (Fs / 1000) + ZP)
    FP = int(Win[1] * (Fs / 1000) + ZP)

    Features_A = []

    for data_G in Data:

        data_G = np.array(data_G)

        BNs = np.arange(data_G.shape[1])

        Features_B = []

        for BN in BNs:

            Features_B.append(np.max(data_G[:, BN, SP : FP], axis = -1))

        Features_A.append(np.array(Features_B).T)

    return Features_A

def MinimumValue(Data, Win, specs = {'ZeroPoint': 100, 'Fs': 500}):

    ZP = specs['ZeroPoint']
    Fs = specs['Fs']

    SP = int(Win[0] * (Fs / 1000) + ZP)
    FP = int(Win[1] * (Fs / 1000) + ZP)

    Features_A = []

    for data_G in Data:

        data_G = np.array(data_G)

        BNs = np.arange(data_G.shape[1])

        Features_B = []

        for BN in BNs:

            Features_B.append(np.min(data_G[:, BN, SP : FP], axis = -1))

        Features_A.append(np.array(Features_B).T)

    return Features_A

def MaximumLag(Data, Win, specs = {'ZeroPoint': 100, 'Fs': 500}):

    ZP = specs['ZeroPoint']
    Fs = specs['Fs']

    SP = int(Win[0] * (Fs / 1000) + ZP)
    FP = int(Win[1] * (Fs / 1000) + ZP)

    Features_A = []

    for data_G in Data:

        data_G = np.array(data_G)

        BNs = np.arange(data_G.shape[1])

        Features_B = []

        for BN in BNs:

            Features_B.append(np.argmax(data_G[:, BN, SP : FP], axis = -1))

        Features_A.append(np.array(Features_B).T)

    return Features_A

def MinimumLag(Data, Win, specs = {'ZeroPoint': 100, 'Fs': 500}):

    ZP = specs['ZeroPoint']
    Fs = specs['Fs']

    SP = int(Win[0] * (Fs / 1000) + ZP)
    FP = int(Win[1] * (Fs / 1000) + ZP)

    Features_A = []

    for data_G in Data:

        data_G = np.array(data_G)

        BNs = np.arange(data_G.shape[1])

        Features_B = []

        for BN in BNs:

            Features_B.append(np.argmin(data_G[:, BN, SP : FP], axis = -1))

        Features_A.append(np.array(Features_B).T)

    return Features_A

def Disc(Feature, ALT = 'two-sided', specs = {'default_fun': 'ttest_ind'}): # Two-sided or One-Sided?

    import scipy.stats as sps

    '''

    TestMat[0, 0] = TestResult of comparing group 1 between two blocks
    TestMat[0, 1] = TestResult of comparing group 1 and 2 in Mode 1

    TestMat[1, 0] = TestResult of comparing group 1 and 2 in Mode 2
    TestMat[1, 1] = TestResult of comparing group 2 between two blocks  
    
    '''

    TestFun = getattr(sps, specs['default_fun']) # functions other that ttest_ind may result in bugs!

    sub_nums = [len(Feature_G) for Feature_G in Feature]

    FeatureArr = np.array([Feature[i][:min(sub_nums)] for i in range(len(Feature))])

    # SHIT CODE HERE:

    TestMats = []

    if ALT == 'All':

        ALTs = ['less', 'two-sided', 'greater']
    
        for ALT in ALTs:

            if len(Feature) == 2:

                TestMat = np.zeros((2, 2))

                for i in range(2):

                    TestMat[i, i] = TestFun(FeatureArr[i, :, 1], FeatureArr[i, :, 0], alternative = ALT).pvalue
                    TestMat[i, i - 1] = TestFun(FeatureArr[1, :, i], FeatureArr[0, :, i], alternative = ALT).pvalue

            elif len(Feature) == 1:

                TestMat = TestFun(FeatureArr[0, :, 0], FeatureArr[0, :, 1], alternative = ALT).pvalue

            else:

                assert False, "IDK How to HANDLE IT :("

            TestMats.append(TestMat)

    else:

        if len(Feature) == 2:

            TestMat = np.zeros((2, 2))

            for i in range(2):

                TestMat[i, i] = TestFun(FeatureArr[i, :, 1], FeatureArr[i, :, 0], alternative = ALT).pvalue
                TestMat[i, i - 1] = TestFun(FeatureArr[1, :, i], FeatureArr[0, :, i], alternative = ALT).pvalue

        elif len(Feature) == 1:

            TestMat = TestFun(FeatureArr[0, :, 0], FeatureArr[0, :, 1], alternative = ALT).pvalue

        else:

            assert False, "IDK How to HANDLE IT :("

        TestMats.append(TestMat)


    return np.squeeze(np.array(TestMats))

def Univariate_ERP_PosNeg(Data, data_lengths, SOI, Channel, Band, specs = {'NumBlock': 1, 'NumSample_inBlock': 20, 'StartPoint': 100, 'FinalPoint': 500}):
    
    import scipy.stats as sps
    from Utils import SciPlot as SP

    NB = specs['NumBlock']
    NSiB = specs['NumSample_inBlock']

    sp = specs['StartPoint']
    fp = specs['FinalPoint']

    # Data_ERP = [[np.mean(sps.zscore(Data[sub_i[0]][SP.DeterminedBlockSampling(Length = int(data_lengths[sub_i[0]]), NumBlock = NB, NumSample_inBlock = NSiB), Channel, :], axis = -1), axis = 1)[:, sp : fp] for sub_i in SOI[G_i]] for G_i in range(len(SOI))]
    Data_ERP = [[np.squeeze(np.array([np.mean(sps.zscore(Data[Mode][sub_i[0]][SP.RandomBlockSampling(Length = int(data_lengths[Mode][sub_i[0]]), NumBlock = NB, NumSample_inBlock = NSiB), Channel, :], axis = -1), axis = 1)[:, sp : fp] for Mode in range(len(Data))])) for sub_i in SOI[G_i]] for G_i in range(len(SOI))]

    if Band == 'All':

        return Data_ERP
    
    else:

        from Utils.Constants import SpectralConstants as SC
        from Modules import GRUniPy as GRU

        assert Band in SC.BandsBounds.keys(), "Band is not DEFINED!"

        BandDecompERP = [[[np.squeeze(GRU.FrequencyBandExt(Data_ERP[G_i][sub_i][Mode], Band = Band)) for Mode in range(len(Data))] for sub_i in range(len(SOI[G_i]))] for G_i in range(len(SOI))]

        return BandDecompERP
    
def Univariate_Power_PosNeg(Data, data_lengths, SOI, Channel, Band, specs = {'NumBlock': 1, 'NumSample_inBlock': 20, 'StartPoint': 100, 'FinalPoint': 500}):
    
    import scipy.stats as sps
    from Utils import SciPlot as SP
    from Modules import GRUniPy as GRU

    NB = specs['NumBlock']
    NSiB = specs['NumSample_inBlock']

    sp = specs['StartPoint']
    fp = specs['FinalPoint']

    if Band == 'All':

        Data_Amp = [[[np.squeeze(GRU.PowerPhaseExt(np.mean(sps.zscore(Data[Mode][sub_i[0]][SP.RandomBlockSampling(Length = int(data_lengths[Mode][sub_i[0]]), NumBlock = NB, NumSample_inBlock = NSiB), Channel, :], axis = -1), axis = 1)[:, sp : fp], return_value = 'Power')) for Mode in range(len(Data))] for sub_i in SOI[G_i]] for G_i in range(len(SOI))]

        return Data_Amp
    
    else:

        from Utils.Constants import SpectralConstants as SC

        assert Band in SC.BandsBounds.keys(), "Band is not DEFINED!"

        Data_ERP = [[np.squeeze(np.array([np.mean(sps.zscore(Data[Mode][sub_i[0]][SP.RandomBlockSampling(Length = int(data_lengths[Mode][sub_i[0]]), NumBlock = NB, NumSample_inBlock = NSiB), Channel, :], axis = -1), axis = 1)[:, sp : fp] for Mode in range(len(Data))])) for sub_i in SOI[G_i]] for G_i in range(len(SOI))]

        BandDecompAmp = [[[np.squeeze(GRU.PowerPhaseExt(GRU.FrequencyBandExt(Data_ERP[G_i][sub_i][Mode], Band = Band), return_value = 'Power')) for Mode in range(len(Data))] for sub_i in range(len(SOI[G_i]))] for G_i in range(len(SOI))]

        return BandDecompAmp
    
def Univariate_Phase_PosNeg(Data, data_lengths, SOI, Channel, Band, specs = {'NumBlock': 1, 'NumSample_inBlock': 20, 'StartPoint': 100, 'FinalPoint': 500}):
    
    import scipy.stats as sps
    from Utils import SciPlot as SP
    from Modules import GRUniPy as GRU

    NB = specs['NumBlock']
    NSiB = specs['NumSample_inBlock']

    sp = specs['StartPoint']
    fp = specs['FinalPoint']

    if Band == 'All':

        Data_Amp = [[[np.squeeze(GRU.PowerPhaseExt(np.mean(sps.zscore(Data[Mode][sub_i[0]][SP.RandomBlockSampling(Length = int(data_lengths[Mode][sub_i[0]]), NumBlock = NB, NumSample_inBlock = NSiB), Channel, :], axis = -1), axis = 1)[:, sp : fp], return_value = 'Phase')) for Mode in range(len(Data))] for sub_i in SOI[G_i]] for G_i in range(len(SOI))]

        return Data_Amp
    
    else:

        from Utils.Constants import SpectralConstants as SC

        assert Band in SC.BandsBounds.keys(), "Band is not DEFINED!"

        Data_ERP = [[np.squeeze(np.array([np.mean(sps.zscore(Data[Mode][sub_i[0]][SP.RandomBlockSampling(Length = int(data_lengths[Mode][sub_i[0]]), NumBlock = NB, NumSample_inBlock = NSiB), Channel, :], axis = -1), axis = 1)[:, sp : fp] for Mode in range(len(Data))])) for sub_i in SOI[G_i]] for G_i in range(len(SOI))]

        BandDecompAmp = [[[np.squeeze(GRU.PowerPhaseExt(GRU.FrequencyBandExt(Data_ERP[G_i][sub_i][Mode], Band = Band), return_value = 'Phase')) for Mode in range(len(Data))] for sub_i in range(len(SOI[G_i]))] for G_i in range(len(SOI))]

        return BandDecompAmp
    
def Univariate_ERP_All(Data, data_lengths, SOI, Channel, Band, specs = {'NumBlock': 2, 'NumSample_inBlock': 10, 'StartPoint': 100, 'FinalPoint': 500}):
    
    import scipy.stats as sps
    from Utils import SciPlot as SP

    NB = specs['NumBlock']
    NSiB = specs['NumSample_inBlock']

    sp = specs['StartPoint']
    fp = specs['FinalPoint']

    Data_ERP = [[np.mean(sps.zscore(Data[sub_i[0]][SP.DeterminedBlockSampling(Length = int(data_lengths[sub_i[0]]), NumBlock = NB, NumSample_inBlock = NSiB), Channel, :], axis = -1), axis = 1)[:, sp : fp] for sub_i in SOI[G_i]] for G_i in range(len(SOI))]

    if Band == 'All':

        return Data_ERP
    
    else:

        from Utils.Constants import SpectralConstants as SC
        from Modules import GRUniPy as GRU

        assert Band in SC.BandsBounds.keys(), "Band is not DEFINED!"

        BandDecompERP = [[[np.squeeze(GRU.FrequencyBandExt(Data_ERP[G_i][sub_i][Trial], Band = Band)) for Trial in range(NB)] for sub_i in range(len(SOI[G_i]))] for G_i in range(len(SOI))]

        return BandDecompERP
    
def Univariate_Power_All(Data, data_lengths, SOI, Channel, Band, specs = {'NumBlock': 2, 'NumSample_inBlock': 10, 'StartPoint': 100, 'FinalPoint': 500}):
    
    import scipy.stats as sps
    from Utils import SciPlot as SP
    from Modules import GRUniPy as GRU

    NB = specs['NumBlock']
    NSiB = specs['NumSample_inBlock']

    sp = specs['StartPoint']
    fp = specs['FinalPoint']

    if Band == 'All':

        Data_Amp = [[np.squeeze(GRU.PowerPhaseExt(np.mean(sps.zscore(Data[sub_i[0]][SP.DeterminedBlockSampling(Length = int(data_lengths[sub_i[0]]), NumBlock = NB, NumSample_inBlock = NSiB), Channel, :], axis = -1), axis = 1)[:, sp : fp], return_value = 'Power')) for sub_i in SOI[G_i]] for G_i in range(len(SOI))]

        return Data_Amp
    
    else:

        from Utils.Constants import SpectralConstants as SC

        assert Band in SC.BandsBounds.keys(), "Band is not DEFINED!"

        Data_ERP = [[np.mean(sps.zscore(Data[sub_i[0]][SP.DeterminedBlockSampling(Length = int(data_lengths[sub_i[0]]), NumBlock = NB, NumSample_inBlock = NSiB), Channel, :], axis = -1), axis = 1)[:, sp : fp] for sub_i in SOI[G_i]] for G_i in range(len(SOI))]
        
        BandDecompAmp = [[[np.squeeze(GRU.PowerPhaseExt(GRU.FrequencyBandExt(Data_ERP[G_i][sub_i][Trial], Band = Band), return_value = 'Power')) for Trial in range(NB)] for sub_i in range(len(SOI[G_i]))] for G_i in range(len(SOI))]

        return BandDecompAmp
    
def Univariate_Phase_All(Data, data_lengths, SOI, Channel, Band, specs = {'NumBlock': 2, 'NumSample_inBlock': 10, 'StartPoint': 100, 'FinalPoint': 500}):
    
    import scipy.stats as sps
    from Utils import SciPlot as SP
    from Modules import GRUniPy as GRU

    NB = specs['NumBlock']
    NSiB = specs['NumSample_inBlock']

    sp = specs['StartPoint']
    fp = specs['FinalPoint']

    if Band == 'All':

        Data_Amp = [[np.squeeze(GRU.PowerPhaseExt(np.mean(sps.zscore(Data[sub_i[0]][SP.DeterminedBlockSampling(Length = int(data_lengths[sub_i[0]]), NumBlock = NB, NumSample_inBlock = NSiB), Channel, :], axis = -1), axis = 1)[:, sp : fp], return_value = 'Phase')) for sub_i in SOI[G_i]] for G_i in range(len(SOI))]

        return Data_Amp
    
    else:

        from Utils.Constants import SpectralConstants as SC

        assert Band in SC.BandsBounds.keys(), "Band is not DEFINED!"

        Data_ERP = [[np.mean(sps.zscore(Data[sub_i[0]][SP.DeterminedBlockSampling(Length = int(data_lengths[sub_i[0]]), NumBlock = NB, NumSample_inBlock = NSiB), Channel, :], axis = -1), axis = 1)[:, sp : fp] for sub_i in SOI[G_i]] for G_i in range(len(SOI))]
        
        BandDecompAmp = [[[np.squeeze(GRU.PowerPhaseExt(GRU.FrequencyBandExt(Data_ERP[G_i][sub_i][Trial], Band = Band), return_value = 'Phase')) for Trial in range(NB)] for sub_i in range(len(SOI[G_i]))] for G_i in range(len(SOI))]

        return BandDecompAmp