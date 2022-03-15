import numpy as np

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))

def normalization_corr(corr):
    means = np.mean(corr, axis=0)
    corr = corr - means
    stds = np.std(corr, axis=0)
    corr = corr / stds.reshape
    return corr

def pearson_distance_add(data_0, data_1, LA_0, LA_1, LO_0, LO_1):
    pearson_corr = np.corrcoef(data_0, data_1)[0,1]
    distance = ((LA_0-LA_1)**2+(LO_0-LO_1)**2)**0.5
    return pearson_corr + distance

