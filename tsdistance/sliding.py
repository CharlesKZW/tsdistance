import numpy as np;
import math

def NCC(x,y):
    r"""
    The formula for Normalized Cross-Correlation (:math:`NCC`) is: :math:`max(CC_{w}(\vec{x}, \vec{y}))`.

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the NCC distance
    """

    length = len(x);
    fftlen = 2 ** math.ceil(math.log2(abs(2*length-1)));
    r = np.fft.ifft(np.multiply(np.fft.fft(x,fftlen),np.conj(np.fft.fft(y,fftlen))))
    
    lenr = len(r) - 1;

    result = np.append(r[lenr-length+2:lenr + 1],r[0:length])

    return result;

def NCCb(x,y):

    r"""
    The formula for Biased Normalized Cross-Correlation (:math:`NCC_b`) is: :math:`max(\frac{CC_{w}(\vec{x}, \vec{y})}{m})`
    

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the NCCb distance
    """

    length = len(x);
    fftlen = 2 ** math.ceil(math.log2(abs(2*length-1)));
    r = np.fft.ifft(np.multiply(np.fft.fft(x,fftlen),np.conj(np.fft.fft(y,fftlen))))
    
    lenr = len(r) - 1;

    result = np.append(r[lenr-length+2:lenr + 1],r[0:length])

    return np.divide(result,length);


def NCCc(x,y):

    r"""
    The formula for Coefficient Normalized Cross-Correlation :math:`NCC_u` is: :math:`max(\frac{CC_{w}(\vec{x}, \vec{y})}{\vert\vert{\vec{x}}\vert\vert\cdot\vert\vert{\vec{y}}\vert\vert})`
    
    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the NCCc distance
    """
    length = len(x);
    fftlen = 2 ** math.ceil(math.log2(abs(2*length-1)));
    r = np.fft.ifft(np.multiply(np.fft.fft(x,fftlen),np.conj(np.fft.fft(y,fftlen))))
    
    lenr = len(r) - 1;

    result = np.append(r[lenr-length+2:lenr + 1],r[0:length])

    return np.divide(result,np.linalg.norm(x) * np.linalg.norm(y))

def NCCu(x,y):

    r"""
    The formula for Unbiased Normalized Cross-Correlation (:math:`NCC_u`) is: :math:`max(\frac{CC_{w}(\vec{x}, \vec{y})}{m-|w-m|})`

    :param x: a time series 
    :type x: np.array
    :param y: another time series
    :type y: np.array
    :return: the NCCu distance
    """

    result = np.correlate(x,y,'full');

    max = math.ceil(len(result)/2);

    a = []
    for i in range(result.size):
            if (i > max - 1):
                a.append(2*max-(i + 1));
            else:
                a.append(i + 1);

    return np.divide(result,a);
