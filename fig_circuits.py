import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
a=QuantumCircuit(1,1,name='a'); a.h(0); a.measure(0,0)
t=Parameter('θ'); b=QuantumCircuit(1,1); b.ry(t,0); b.measure(0,0)
c=QuantumCircuit(2,2); c.h(0); c.cx(0,1); c.measure([0,1],[0,1])
d=QuantumCircuit(1,1); d.h(0); d.barrier(); d.h(0); d.measure(0,0)
fig,axs=plt.subplots(2,2,figsize=(8,4.2))
titles=['(a) Fair coin','(b) Biased coin','(c) Entangled coins (Bell state)','(d) Interference (H–H)']
for ax,qc,tl in zip(axs.flat,[a,b,c,d],titles):
    qc.draw('mpl',ax=ax,style='bw',fold=-1); ax.set_title(tl,fontsize=11,family='DejaVu Serif')
fig.tight_layout(); fig.savefig('fig_circuits.png',dpi=200,bbox_inches='tight')
