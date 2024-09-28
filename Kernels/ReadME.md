There are 4 kernels to calculate the Connectivity Strength.

It must be noted that the 'TE' kernel is a modified version of [This Repository](https://github.com/mohammad7613/RL-and-DD/blob/main/FunctionalConnectivityCodes/TransferEntropy/TransferEntropy.py)

# Connectivity Kernels

Up to now, 5 Groups of Connectivity Measurements are implemented here, each one results a float number that may be interpret as strength of connection between two electrodes, in compare with the other ones. Kernels are listed below, more details will be added soon :)

## PLI Family

This Family contains:
- PLI (Phase Lag Index),
- dPLI (Directed Phase Lag Index),
- and, wPLI (Weighted Phase Lag Index)

## Transfer Entropy

`README is Under Construction ...`

## Granger Causality

`README is Under Construction ...`

## PLV

PLV stands for Phase Locking Value. No IDEA.

## PAC

PAC stands for Phase-Amplitude Coupling. This measurement assume that data is transmitting via coupling of phase and amplitude (it is more common to consider phase and amplitude of different spectral bands). So there are two important steps in calculation of PAC, extracting Spectral Band Signal and then Measuring the similarity of Phase and Amplitude.

Up to now, there are two different implemented method to extract signal, including,
- Wavelet Spectral Decomposition
- and Rid-Rihaczek Kernel (References of this method will be provided later),

also two different implemented similarity metrics,
- Mean Vector Length (References will be here)
- and Modulation Index

Test Results of PAC are figured below. (Synthesized Data Are Generated Using the [EEG Simulation Repository](https://github.com/hamidm317/BrainElectroSignalsSim))

![PAC in Source Level using MVL Kernel](https://github.com/hamidm317/EEG_ProcessingTools/blob/main/images/PAC_All2All_SourceLevelMVL_R01.png)


Below figure is generated Using Kullback Leibler Divergence as Distance Metric.


![PAC in Source Level using MI Kernel](https://github.com/hamidm317/EEG_ProcessingTools/blob/main/images/PAC_All2All_SourceLevelMI_R01.png)


Above Plots figures the Source Level means that signals are High SNR, but below there are Sensor Level Plots, with less SNR.


![PAC in Sensor Level using MVL Kernel](https://github.com/hamidm317/EEG_ProcessingTools/blob/main/images/PAC_All2All_SensorLevelMVL_R01.png)


Below figure is generated Using Kullback Leibler Divergence as Distance Metric.


![PAC in Sensor Level using MI Kernel](https://github.com/hamidm317/EEG_ProcessingTools/blob/main/images/PAC_All2All_SensorLevelMI_R01.png)
