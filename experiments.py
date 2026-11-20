import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from scipy.stats import chisquare
SEED=42; sim=AerSimulator(seed_simulator=SEED); R={}
def run(qc,shots,backend=sim):
    return backend.run(transpile(qc,backend),shots=shots,seed_simulator=SEED).result().get_counts()
# 1 fair coin
def fair():
    qc=QuantumCircuit(1,1); qc.h(0); qc.measure(0,0); return qc
R['fair']={}
for s in [10,100,1000,10000,100000]:
    c=run(fair(),s); h=c.get('0',0); t=c.get('1',0)
    chi=chisquare([h,t]); R['fair'][s]=dict(heads=h,tails=t,p_heads=h/s,abs_err=abs(h/s-0.5),chi2=float(chi.statistic),p=float(chi.pvalue))
# running proportion for figure (memory)
res=sim.run(transpile(fair(),sim),shots=5000,memory=True,seed_simulator=SEED).result().get_memory()
bits=np.array([1-int(b) for b in res]); run_p=np.cumsum(bits)/np.arange(1,len(bits)+1)
rng=np.random.default_rng(SEED); cb=rng.integers(0,2,5000); cl_p=np.cumsum(cb)/np.arange(1,5001)
R['classical_5000']=float(cl_p[-1]); R['quantum_5000']=float(run_p[-1])
# 2 biased coin
thetas=np.linspace(0,np.pi,13); bias=[]
for th in thetas:
    qc=QuantumCircuit(1,1); qc.ry(th,0); qc.measure(0,0); c=run(qc,4000)
    bias.append(dict(theta=float(th),theory=float(np.sin(th/2)**2),measured=c.get('1',0)/4000))
R['biased']=bias; R['biased_max_err']=max(abs(b['theory']-b['measured']) for b in bias)
# 3 two coins: independent vs entangled
qi=QuantumCircuit(2,2); qi.h([0,1]); qi.measure([0,1],[0,1])
qe=QuantumCircuit(2,2); qe.h(0); qe.cx(0,1); qe.measure([0,1],[0,1])
ci=run(qi,4000); ce=run(qe,4000)
R['indep']={k:ci.get(k,0) for k in ['00','01','10','11']}; R['ent']={k:ce.get(k,0) for k in ['00','01','10','11']}
def corr(c):
    n=sum(c.values()); return (c.get('00',0)+c.get('11',0)-c.get('01',0)-c.get('10',0))/n
R['corr_indep']=corr(ci); R['corr_ent']=corr(ce)
# 4 interference + 5 noise
def hh():
    qc=QuantumCircuit(1,1); qc.h(0); qc.barrier(); qc.h(0); qc.measure(0,0); return qc
R['hh_ideal']=run(hh(),4000)
noise=[]
for p in [0,0.01,0.02,0.05,0.1,0.2]:
    nm=NoiseModel()
    if p>0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p,1),['h'])
        nm.add_all_qubit_readout_error(ReadoutError([[1-p/2,p/2],[p/2,1-p/2]]))
    b=AerSimulator(noise_model=nm,seed_simulator=SEED)
    c1=run(hh(),4000,b); c2=run(fair(),4000,b)
    noise.append(dict(p=p,hh_p0=c1.get('0',0)/4000,fair_p0=c2.get('0',0)/4000))
R['noise']=noise
json.dump(R,open('results.json','w'),indent=1)
# figures
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10})
fig,ax=plt.subplots(figsize=(6.4,3.2)); x=np.arange(1,5001)
ax.plot(x,run_p,color='#1f4e8c',lw=1.3,label='Quantum coin (Hadamard)')
ax.plot(x,cl_p,color='#c0504d',lw=1,alpha=.8,label='Classical PRNG coin')
ax.axhline(.5,ls='--',color='gray',lw=.8); ax.set_xscale('log'); ax.set_ylim(.2,.8)
ax.set_xlabel('Number of tosses (log scale)'); ax.set_ylabel('Fraction of heads'); ax.legend(frameon=False); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig('fig_convergence.png',dpi=200)
fig,ax=plt.subplots(figsize=(5.6,3.2)); tt=np.linspace(0,np.pi,200)
ax.plot(tt,np.sin(tt/2)**2,color='#1f4e8c',label=r'Theory $\sin^2(\theta/2)$')
ax.scatter([b['theta'] for b in bias],[b['measured'] for b in bias],color='#c0504d',zorder=3,s=22,label='Simulated (4000 shots)')
ax.set_xlabel(r'Rotation angle $\theta$ (rad)'); ax.set_ylabel('P(tails = |1⟩)'); ax.legend(frameon=False); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig('fig_biased.png',dpi=200)
fig,ax=plt.subplots(figsize=(5.8,3.0)); ks=['00','01','10','11']; w=.38; xi=np.arange(4)
ax.bar(xi-w/2,[R['indep'][k]/4000 for k in ks],w,color='#8eaadb',label='Independent (H⊗H)')
ax.bar(xi+w/2,[R['ent'][k]/4000 for k in ks],w,color='#1f4e8c',label='Entangled (Bell)')
ax.set_xticks(xi); ax.set_xticklabels(ks); ax.set_xlabel('Outcome (coin B, coin A)'); ax.set_ylabel('Probability'); ax.legend(frameon=False); ax.grid(axis='y',alpha=.3)
fig.tight_layout(); fig.savefig('fig_twocoins.png',dpi=200)
fig,ax=plt.subplots(figsize=(5.6,3.0)); ps=[n['p'] for n in noise]
ax.plot(ps,[n['hh_p0'] for n in noise],'o-',color='#1f4e8c',label='H–H circuit, P(|0⟩)')
ax.plot(ps,[n['fair_p0'] for n in noise],'s--',color='#c0504d',label='Single H, P(|0⟩)')
ax.set_xlabel('Depolarising error rate p'); ax.set_ylabel('Probability'); ax.set_ylim(.4,1.02); ax.legend(frameon=False); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig('fig_noise.png',dpi=200)
print(json.dumps(R,indent=1))
