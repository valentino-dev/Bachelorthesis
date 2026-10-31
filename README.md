# Bachelorthesis
Title: Calculation of the quark-antiquark potential in the Hamiltonian formalism of U(1) gauge theory.

This thesis uses exact diagonalization of the Hamiltonian to obtain the eigenstates and expectation value of the system. The Hamiltonian is constructed by determining transitions to physical states. The total Hilbert space is constrained to the physical Hilbert space by demanding charge conservation.

For simplification, the static case is assumed, where the quarks have infinite mass. In the U(1) theory, this results in the quarks having only one property: the charge. Therefor a particle of charge 1 is a quark and a particle of charge -1 is a antiquark.

The lattice is quadratic and has always a total charge of 0, while periodic boundary conditons (PBC) can be optionally used.

Though the computation was done on the high-performance-cluster Marvin in Bonn, for the largest matrices, $3.49\cdot 10^9\times 3.49\cdot 10^9$ for a $4\times 4$ lattice with PBC, sparse row matrices were employed.


<figure>
    <img src="https://github.com/valentino-dev/Bachelorthesis/blob/main/latex/images/quark_antiquark_potential_g_1_page-0001.jpg" alt="Click here to view image." width="600"/>
    <figcaption>Quark-Antiquark potential for different but comparable setups at $g=1$. By default $3 \times 3$ dimensions and no PBC, if not stated otherwise.</figcaption>
</figure>
