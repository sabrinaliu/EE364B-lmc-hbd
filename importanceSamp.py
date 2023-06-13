import numpy as np
import scipy.stats as stats
import scipy as sp

# use description of particle filter: HMM
# https://ocw.mit.edu/courses/6-438-algorithms-for-inference-fall-2014/resources/mit6_438f14_lec19/
# uses our prior on the weights as the template distribution

def importanceSamp(S, alphaOld, y, d, mu0, sig0, sigv):
    # E step: particle filter
    (N,) = y.shape
    x0 = np.zeros((d+1, N, S))
    x0[:,0, :] = np.random.multivariate_normal(mu0, sig0, S).T
    L = np.zeros((N, S)) # stores weighted combo of prev d RR intervals
    posteriorP = np.zeros((N, S), dtype=np.longfloat)

    # generate samples from the prior
    for i in range(1,N):
        for j in range(S):
            x0[:,i,j] = np.random.multivariate_normal(x0[:,i-1, j], sigv).T

    # compute the posterior likelihood from the samples
    for i in range(N):
        hi = np.zeros((d+1,)) + y[0]
        hi[0] = 1
        hi[1:min(d+1, i)+1] = y[max(i-d,0):i][::-1]

        L[i,:] = hi.T @ x0[:,i,:]

        beta = alphaOld * np.exp(-L[i,:])

        posteriorP[i,:] = stats.gamma.pdf(y[i], a=alphaOld, loc=0, scale=1/beta)
        
    w0LogFinal = np.sum(np.log(posteriorP), axis=0)

    funcVals = np.zeros((S,))
    for i in range(S):
        for j in range(N):
            funcVals[i] += alphaOld * np.log(alphaOld) - alphaOld * L[j,i] - sp.special.loggamma(alphaOld) + (alphaOld-1) * np.log(y[j]) - alphaOld*np.exp(-L[j,i])*y[j]
        
    # make all funcVals positive so we can take the log
    eps = 1e-3
    funcValsMin = np.min(funcVals)
    if funcValsMin > 0:
        funcValsMin = 0
    funcValsPos = funcVals - funcValsMin + eps
    logFuncVals = np.log(funcValsPos)

    # do computations with log for numerical stability since weights are very small
    logCoeffNum = w0LogFinal[0]+ logFuncVals[0] \
        + np.log(1 + np.sum(np.exp(w0LogFinal[1:] + logFuncVals[1:] - w0LogFinal[0] - logFuncVals[0])))
    logCoeffDenom = w0LogFinal[0] + np.log(1 + np.sum(np.exp(w0LogFinal[1:] - w0LogFinal[0])))

    # recover value by exponentiating and also removing offset that made values positive
    expectation = np.exp(logCoeffNum  - logCoeffDenom) + funcValsMin - eps

    return expectation