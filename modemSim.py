import numpy as np
from numpy.fft import fft, ifft
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from fractions import gcd

def main():
    """ Simulation Parameters """
    snrRange = np.arange(0,13,2)
    minNumErrors   = 300
    minNumSymbols  = 20
    reportInterval = 100
    
    modem = ofdmModem()
    h = Channel()
    BER = np.zeros(snrRange.size) 
    idx = 0
    for snr in snrRange:
        i = 0
        numErrors = 0
        numSymbols = 0
        while (numErrors < minNumErrors) or (numSymbols < minNumSymbols):
            d = np.random.randint(0,4,modem.DataIdx.size,dtype=np.uint8)
            x = modem.do_modulate(d)
            y = h.do_awgn(x,snr,measure=True)
            dhat = modem.do_demodulate(y)
            """ Error Accounting """
            numErrors  += (d!=dhat).sum()
            numSymbols += 1
            if (i%reportInterval==0):
                showProgress(snr,numErrors,minNumErrors,numSymbols,minNumSymbols,modem.DataIdx.size)
            i+=1
        
        BER[idx] = numErrors/(numSymbols*modem.DataIdx.size)
        showProgress(snr,numErrors,minNumErrors,numSymbols,minNumSymbols,modem.DataIdx.size)
        idx+=1
        print("")
    
    plt.figure()
    plt.grid()
    plt.semilogy(snrRange,BER,'-o')
    plt.xlim(snrRange.min(), snrRange.max())
    plt.ylim(BER.min(), 1)
    plt.title("Symbol Error Rate Performance")
    plt.show()

        #print("Symbol Errors : "+str(errors))
#    modem.do_modplot()

def showProgress(snr,numErrors,minNumErrors,numSymbols,minNumSymbols,blocksize):
    curBer = (numErrors+1)/(numSymbols*blocksize)
    print("SNR: " + str("%5.2f"%snr) + "[dB]" +
       " | BER: " + str("%5.2e"%curBer) +
       " | Progress : " + str("%5.1f"%(100*min(numErrors/minNumErrors, numSymbols/minNumSymbols)))+"%"+ 
       " | Errors : "   + str("%5d" % numErrors) + 
       " | Symbols : "  + str("%6d" % numSymbols),end='\r')



class ofdmModem:
    """OFDM Modulator class""" 
         
    def __init__(self,N=4096,Non=3072,NCP=128,NCS=16,pilotSpacing=16,modName='qpsk'):
        self.Subcarriers = np.zeros(N,dtype=complex)
        self.cpSize = NCP
        self.csSize = NCS
        self.pilotSpacing = pilotSpacing 
        self.ActiveSubcarriers = np.zeros(Non,dtype=int) 
        self.modulation = modName
        
        # Create mapper object
        self.mapper = qamMapper()

        # Deriveted parameters
        self.ActiveSubcarriers = np.arange(
            -np.floor(self.ActiveSubcarriers.size/2),
            +np.ceil(self.ActiveSubcarriers.size/2),
            dtype=int)%self.Subcarriers.size
        
        self.PilotIdx  = np.arange(
            -np.floor(self.ActiveSubcarriers.size/2),
            +np.ceil( self.ActiveSubcarriers.size/2),
        self.pilotSpacing, dtype=int)%self.Subcarriers.size

        self.DataIdx  = np.setdiff1d(self.ActiveSubcarriers, self.PilotIdx)

        # Pre-alocating arrays
        # Modulator
        self._d = np.zeros(self.DataIdx.size, dtype=np.uint8)
        self._x = np.zeros(self.Subcarriers.size + 
                           self.cpSize + self.csSize, dtype=complex)
        # Initializing Pilots cache
        self.Pilots = self.get_zadoffChu(self.PilotIdx.size,17)
        # Demodulator
        self._shat = np.zeros(self.DataIdx.size, dtype=complex)
        self._dhat = np.zeros(self.DataIdx.size, dtype=np.uint8)
        
    def do_modulate(self, data):
        ncp = self.cpSize
        ncs = self.csSize
        n = self.Subcarriers.size
        self.Subcarriers[self.DataIdx]  = self.mapper.do_qamMap(data,self.modulation)
        self.Subcarriers[self.PilotIdx] = self.Pilots 
        self._x[ncp:ncp+n] = ifft(self.Subcarriers,n)
        self._x[:ncp] = self._x[n:n+ncp]
        self._x[n+ncp:] = self._x[ncp:ncp+ncs]
        return self._x
    
    def do_modplot(self):
        x = self.Subcarriers[self.DataIdx]
        y = self._shat
        plt.figure(1)
        plt.grid()
        plt.scatter(np.real(y),np.imag(y),marker='o')
        plt.scatter(np.real(x),np.imag(x),marker='*')
        plt.xlabel("In-Phase")
        plt.ylabel("Quadrature")
        plt.show()

    def do_demodulate(self, recSignal):
        ncp = self.cpSize
        ncs = self.csSize
        n = self.Subcarriers.size
        self._shat = fft(recSignal[ncp:ncp+n])[self.DataIdx]
        self._dhat = self.mapper.do_qamDemap(self._shat,self.modulation)
        return self._dhat
   
    def get_zadoffChu(self,length,family):
        M = family
        N = length
        zc = np.zeros(N,dtype=complex)
        if (gcd(M,N) > 1) :
            print("WARNING: ZC Sequence family " + str(M) + " and " + str(N) + " length are not relative primes")
        k  = np.arange(N)
        zc = np.exp(-1j*((M*np.pi*k*(k+k.size%2))/N))
        return zc

class qamMapper:
    constelation = {
        'bpsk' : np.array([-1+0j, +1+0j]),
        'qpsk' : np.array([(+1+1j), (+1-1j), (-1-1j), (-1+1j)])
    }
    def __init__(self):
        pass

    def do_qamMap(self, data ,mod):
        return self.constelation[mod].take(data)

    def do_qamDemap(self,s, mod):
        map = self.constelation[mod]
        dist = abs(
            np.kron(s,np.ones([map.size,1])) - 
            np.kron(map,np.ones([s.size,1])).transpose()
            )
        return np.argmin(dist,axis=0)

class Channel:
    def __init__(self):
        pass

    def do_filter(self,delays,powerProfile):
        pass

    def do_awgn(self, x, snrdb, measure=False):
        gain = 10**(-snrdb/20)/np.sqrt(2)
        if (measure):
            gain = gain*x.std()
        y = x + gain*(np.random.randn(x.size) + 1j*np.random.randn(x.size))
        return y
        
if __name__ == "__main__":
    main()
