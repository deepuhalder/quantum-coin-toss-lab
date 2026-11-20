"""Quantum Coin Toss and Probability Laboratory - Streamlit front end.
Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

st.set_page_config(page_title="Quantum Coin Toss Lab", page_icon="🪙", layout="centered")
st.title("🪙 Quantum Coin Toss & Probability Lab")
st.caption("Qiskit + Aer simulator. Heads = |0⟩, Tails = |1⟩.")


def run(qc, shots, noise_p=0.0, seed=None):
    nm = None
    if noise_p > 0:
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(noise_p, 1), ["h", "ry"])
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - noise_p / 2, noise_p / 2], [noise_p / 2, 1 - noise_p / 2]]))
    sim = AerSimulator(noise_model=nm)
    return sim.run(transpile(qc, sim), shots=shots, seed_simulator=seed).result().get_counts()


shots = st.sidebar.select_slider("Number of tosses (shots)", [10, 100, 1000, 10000], value=1000)
noise_p = st.sidebar.slider("Noise level (depolarising p)", 0.0, 0.2, 0.0, 0.01)
seed = st.sidebar.number_input("Random seed (0 = none)", 0, 9999, 0) or None

tab1, tab2, tab3, tab4 = st.tabs(["Fair coin", "Biased coin", "Two coins", "Interference"])

with tab1:
    qc = QuantumCircuit(1, 1); qc.h(0); qc.measure(0, 0)
    st.text(qc.draw("text"))
    if st.button("Toss!", key="fair"):
        c = run(qc, shots, noise_p, seed)
        h, t = c.get("0", 0), c.get("1", 0)
        st.metric("Heads fraction", f"{h / shots:.3f}", f"{h / shots - 0.5:+.3f} from 0.5")
        st.bar_chart(pd.DataFrame({"count": [h, t]}, index=["Heads", "Tails"]))

with tab2:
    theta = st.slider("Rotation angle θ (radians)", 0.0, float(np.pi), float(np.pi / 3), 0.01)
    st.write(f"Theory: P(Tails) = sin²(θ/2) = **{np.sin(theta / 2) ** 2:.3f}**")
    qc = QuantumCircuit(1, 1); qc.ry(theta, 0); qc.measure(0, 0)
    if st.button("Toss biased coin", key="bias"):
        c = run(qc, shots, noise_p, seed)
        st.metric("Measured P(Tails)", f"{c.get('1', 0) / shots:.3f}")
        st.bar_chart(pd.DataFrame({"count": [c.get("0", 0), c.get("1", 0)]}, index=["Heads", "Tails"]))

with tab3:
    mode = st.radio("Coin pair", ["Independent (H⊗H)", "Entangled (Bell state)"])
    qc = QuantumCircuit(2, 2)
    if mode.startswith("Independent"):
        qc.h([0, 1])
    else:
        qc.h(0); qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    st.text(qc.draw("text"))
    if st.button("Toss both", key="two"):
        c = run(qc, shots, noise_p, seed)
        keys = ["00", "01", "10", "11"]
        df = pd.DataFrame({"count": [c.get(k, 0) for k in keys]}, index=keys)
        st.bar_chart(df)
        same = (c.get("00", 0) + c.get("11", 0)) / shots
        st.metric("Coins agree", f"{same:.1%}")

with tab4:
    st.write("A Hadamard 'tosses' the coin; a second Hadamard 'un-tosses' it. "
             "With no noise the result is always Heads, which no classical coin can do.")
    qc = QuantumCircuit(1, 1); qc.h(0); qc.barrier(); qc.h(0); qc.measure(0, 0)
    st.text(qc.draw("text"))
    if st.button("Run H–H", key="hh"):
        c = run(qc, shots, noise_p, seed)
        st.metric("P(Heads)", f"{c.get('0', 0) / shots:.3f}")
