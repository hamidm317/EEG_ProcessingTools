import numpy as np

from Utils import Constants
from Utils import StatisticalUtils as StUt

def DistributionSampleTest(Sample, Population, TestDistribution = 'NonParametric', NonParTestMethod = 'MatWhiU'):

    assert len(Population) > 7, "Population at least must include 8 members"

    Population = np.array(Population)
    Sample = np.array(Sample)

    assert np.all([Sample.shape == Bishop.shape for Bishop in Population]), "Sample and All Population Members must have same Shape"

    ReliaProb = []

    MainShape = Sample.shape

    SampleVector = np.ravel(Sample)

    PopulationMainShape = Population.shape

    if TestDistribution in Constants.StaConstants.Distributions['Availables']:

        ParameterExtract = getattr(StUt, 'Ext' + TestDistribution + 'Parameters')
        Parameters = ParameterExtract(Population)

        SignificanceTest = getattr(StUt, TestDistribution)
        
        ParametersVector = np.array([np.ravel(Parameter) for Parameter in Parameters])

        for LSi, LocalSample in enumerate(SampleVector):

            ReliaProb.append(SignificanceTest(LocalSample, ParametersVector[:, LSi]))

    else:

        SignificanceTest = getattr(StUt, NonParTestMethod)

        PopulationVector = Population.reshape((PopulationMainShape[0],) + (np.prod(PopulationMainShape[1:]),))

        for LSi, LocalSample in enumerate(SampleVector):

            ReliaProb.append(SignificanceTest(LocalSample, PopulationVector[:, LSi]))

    return np.array(ReliaProb).reshape(MainShape)