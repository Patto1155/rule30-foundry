#!/usr/bin/env python
"""Digit-Fourier, ANF and digit-sensitivity profile of n -> c_n (Prize 3).

Treats the first 2^23 centre bits as a Boolean function of the 23 binary
digits of n and asks whether it correlates with ANY parity of those digits
(all 2^23 Walsh coefficients), has low algebraic degree, or is insensitive to
any digit. Seven seeded random functions are the null.

Result (2026-11-24): max Walsh z = 5.70 against a null band 5.12..5.73 and an
expected sqrt(2 ln N) = 5.65; every digit sensitivity within 0.4996..0.5007;
ANF generic. No structure. A statistical null, not a certificate.
Usage: python experiments/digit_fourier_profile.py
"""

import time

import numpy as np

m = 23
N = 1 << m


def fwht(a):
    a=a.copy(); h=1; n=a.size
    while h<n:
        a=a.reshape(-1,2,h)
        x=a[:,0,:]+a[:,1,:]; y=a[:,0,:]-a[:,1,:]
        a=np.stack([x,y],axis=1).reshape(-1); h*=2
    return a
def mobius(b):
    a=b.copy(); h=1; n=a.size
    while h<n:
        a=a.reshape(-1,2,h); a[:,1,:]^=a[:,0,:]; a=a.reshape(-1); h*=2
    return a


def main():
    raw=np.fromfile('data/center_col_10M.bin',dtype=np.uint8)
    bits=np.unpackbits(raw,bitorder='little')[:10_000_000]
    pc=np.zeros(N,dtype=np.uint8)
    for j in range(m): pc+= ((np.arange(N)>>j)&1).astype(np.uint8)
    def profile(b):
        f=b[:N]
        w=fwht(1-2*f.astype(np.int32))
        aw=np.abs(w); top=np.argsort(aw)[-3:][::-1]
        # weight by Walsh degree
        lowdeg=[float((w[pc==k].astype(np.float64)**2).sum()/N/N) for k in range(0,4)]
        anf=mobius(f)
        degcnt=[int(anf[pc==k].sum()) for k in range(m+1)]
        sens=[float((f != f[np.arange(N)^(1<<j)]).mean()) for j in range(m)]
        return dict(maxW=int(aw[top[0]]), z=float(aw[top[0]]/np.sqrt(N)), top=[(int(t),int(w[t])) for t in top],
                    lowdeg=lowdeg, anf_top=degcnt[-3:], anf_frac=[round(degcnt[k]/max(1,(pc==k).sum()),4) for k in (1,2,3,m-1)],
                    sens_minmax=(min(sens),max(sens)), sens=sens)
    t=time.time()
    c=profile(bits); print('center',{k:v for k,v in c.items() if k!='sens'}, time.time()-t, flush=True)
    print(' sens',[round(s,4) for s in c['sens']])
    zs=[]
    for s in range(30,37):
        r=profile(np.random.default_rng(s).integers(0,2,size=N,dtype=np.uint8))
        zs.append(r['z']); print('rand',s,{k:v for k,v in r.items() if k not in('sens','top')},flush=True)
    print('expected max z ~ sqrt(2 ln N)=',np.sqrt(2*np.log(N)))


if __name__ == "__main__":
    main()
