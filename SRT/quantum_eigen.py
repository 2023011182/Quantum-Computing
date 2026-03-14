# -*- coding: utf-8 -*-
"""
Full Quantum QR Algorithm for Eigenvalue Estimation (Fixed V2)
Fix: Initialize basis completion candidates as complex to prevent casting errors.
"""

import numpy as np
import copy
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import HamiltonianGate

# ==========================================
# Part 1: Quantum Algorithms
# ==========================================

def quantum_gram_schmidt(vectors, error_rate=1e-4, circuit_return=False):
    '''
    Quantum modified Gram-Schmidt process.
    '''
    vector_size = len(vectors[0])
    vector_number = len(vectors)
    # Calculate qubits needed
    qubit_number = int(np.ceil(np.log2(vector_size))) 
    if 2 ** qubit_number < vector_size:
        qubit_number += 1
    total_qubit_number = qubit_number + 1 
    max_runtime_of_circuit = int(np.log(1/error_rate) / error_rate)

    simulator = Aer.get_backend('statevector_simulator')

    # Normalize first vector
    norm_v0 = np.linalg.norm(vectors[0])
    if norm_v0 < 1e-10:
        first_constructed_vector = np.zeros_like(vectors[0])
    else:
        first_constructed_vector = vectors[0] / norm_v0
        
    first_constructed_vector_extended = np.concatenate((first_constructed_vector, np.zeros(2 ** qubit_number - vector_size)))

    constructed_basis = [first_constructed_vector]

    # Initialize Hamiltonian
    current_hamiltonian = np.zeros((2**(total_qubit_number), 2**(total_qubit_number)), dtype=complex)
    term1 = np.outer(np.conjugate(first_constructed_vector_extended), first_constructed_vector_extended)
    term2 = np.array([[0, 0], [0, 1]], dtype=complex)
    current_hamiltonian += np.kron(term1, term2)
    
    qubit_list = [i for i in range(total_qubit_number)]
    qc_set = []

    for i in range(1, vector_number):
        norm_vi = np.linalg.norm(vectors[i])
        if norm_vi < 1e-10:
            continue # Skip zero vectors

        v = vectors[i] / norm_vi 
        v = np.concatenate((v, np.zeros(2 ** qubit_number - vector_size))) 
        count = 0
        
        qc = QuantumCircuit(total_qubit_number, 1)
        qc.initialize(v, range(1, total_qubit_number))
        qc.h(0)
        
        hamiltonian_gate = HamiltonianGate(current_hamiltonian, np.pi, label='Hami')
        qc.append(hamiltonian_gate, qubit_list)
        
        qc.h(0)
        qc.measure(0, 0)
        
        # Run simulation
        while count < max_runtime_of_circuit:
            count += 1
            qc_transpiled = transpile(qc, simulator)
            job = simulator.run(qc_transpiled, shots=1)
            result = job.result()
            
            if '0' in result.get_counts().keys():
                qc_set.append(qc)
                statevector = np.asarray(result.get_statevector()) 
                
                # Update Hamiltonian
                current_hamiltonian += np.outer(np.conjugate(statevector), statevector)
                
                # Extract vector from ancilla |0> component
                reduced_statevector = np.zeros(vector_size, dtype=complex)
                for j in range(vector_size):
                    reduced_statevector[j] = statevector[2 * j]
                
                # Double check if the result is effectively zero
                if np.linalg.norm(reduced_statevector) > 1e-6:
                    constructed_basis.append(reduced_statevector)
                break

    length = len(constructed_basis)
    constructed_basis = np.asarray(constructed_basis)

    if circuit_return:
        return qc_set, constructed_basis, length
    else:
        return constructed_basis, length

# ==========================================
# Part 2: Utils and QR Logic
# ==========================================

def classical_gram_schmidt(vectors):
    vectors_record = copy.deepcopy(vectors)
    basis = []
    for v in vectors_record:
        tmp = copy.deepcopy(v)
        for b in basis:
            v -= np.dot(np.conjugate(b), tmp) * b
        if np.linalg.norm(v) > 1e-10:
            basis.append(v / np.linalg.norm(v))
    basis = np.array(basis)
    return basis, len(basis)

def classical_modified_gram_schmidt(vectors):
    vectors_record = copy.deepcopy(vectors)
    basis = []
    for v in vectors_record:
        for b in basis:
            v -= np.dot(np.conjugate(b), v) * b
        if np.linalg.norm(v) > 1e-10:
            basis.append(v / np.linalg.norm(v))
    basis = np.array(basis)
    return basis, len(basis)

def QR_decomposition(A, choosen_method='mgs', circuit_return=False, error=1e-4):
    '''
    QR decomposition of matrix A with Basis Completion for Rank Deficient Matrices.
    Ensures Q and R are always square (N x N).
    '''
    N = A.shape[0] 
    current_A = copy.deepcopy(A)
    current_A = np.array(current_A, dtype=complex)
    
    # 1. Get (potentially incomplete) Basis
    if choosen_method == 'cgs':
        basis, length = classical_gram_schmidt(current_A.T)
    elif choosen_method == 'mgs':
        basis, length = classical_modified_gram_schmidt(current_A.T)
    elif choosen_method == 'quantum':
        if circuit_return:
            circuit_set, basis, length = quantum_gram_schmidt(A.T, circuit_return=circuit_return, error_rate=error)
        else:
            basis, length = quantum_gram_schmidt(A.T, error_rate=error)
    else:
        raise ValueError('choosen_method should be cgs or mgs or quantum')
    
    # 2. Basis Completion (Updated Fix: Use complex dtype)
    if length < N:
        basis_list = list(basis) if length > 0 else []
        
        # [Fix] Initialize eye with complex dtype to match basis vectors
        eye = np.eye(N, dtype=complex)
        
        for i in range(N):
            if len(basis_list) == N:
                break
            
            candidate = eye[i] # Now this is complex128
            
            # Orthogonalize candidate against existing basis
            for b in basis_list:
                candidate -= np.dot(np.conjugate(b), candidate) * b
            
            # If candidate is non-zero, add to basis
            if np.linalg.norm(candidate) > 1e-6:
                # Normalize before adding
                basis_list.append(candidate / np.linalg.norm(candidate))
        
        basis = np.array(basis_list)
        length = len(basis)

    # 3. Construct Q and R
    Q = basis.T # Shape (N, N)
    R = np.zeros((N, N), dtype=complex) # Shape (N, N)
    
    for i in range(N):
        for j in range(i, N): 
            if i < length:
                R[i, j] = np.dot(np.conjugate(basis[i]), A.T[j])

    if choosen_method == 'quantum' and circuit_return:
        return circuit_set, Q, R
    return Q, R

def get_wilkinson_shift(Am):
    """
    Calculate Wilkinson shift for the bottom-right 2x2 block.
    """
    n = Am.shape[0]
    if n < 2:
        return Am[0, 0]
    
    a = Am[n-2, n-2]
    b = Am[n-2, n-1]
    c = Am[n-1, n-2] 
    d = Am[n-1, n-1]
    
    delta = (a - d) / 2.0
    sgn = 1 if delta >= 0 else -1
    
    denom = np.abs(delta) + np.sqrt(delta**2 + b*c)
    if denom == 0:
        return d
        
    mu = d - (sgn * b * c) / denom
    return mu

def qr_algorithm(A, choosen_method='mgs', max_iteration=1000, eps=1e-4, record=False, use_shift=True):
    '''
    QR algorithm with robust handling of dimensions.
    '''
    recorder = []
    Ak = copy.deepcopy(A).astype(complex)
    transform_matrix = np.eye(len(A), dtype=complex)
    n = len(A)
    
    print(f"DEBUG: Initial Matrix Norm: {np.linalg.norm(Ak):.4f}")
    
    m = n # Active size

    for i in range(max_iteration):
        if m <= 1:
            break

        Ak0 = Ak.copy()
        if record:
            recorder.append(np.diag(Ak))
            
        # 1. Shift Strategy
        mu = 0
        if use_shift:
            mu = get_wilkinson_shift(Ak[:m, :m])
            if np.abs(mu) < 1e-10 and i < 10: 
                mu = np.random.rand() * 0.1

        # 2. Apply Shift: A - mu*I
        shift_matrix = mu * np.eye(n, dtype=complex)
        Ak_shifted = Ak - shift_matrix
        
        # 3. QR Decomposition
        Qk, Rk = QR_decomposition(Ak_shifted, choosen_method, error=1e-3)
        
        # 4. Update: A = R*Q + mu*I
        Ak = np.dot(Rk, Qk) + shift_matrix
        transform_matrix = np.dot(transform_matrix, Qk)     
        
        # 5. Deflation Check
        off_diagonal_val = np.abs(Ak[m-1, m-2])
        
        if i % 5 == 0 or i < 3:
            total_off_diag = np.sum(np.abs(Ak[:m, :m] - np.diag(np.diag(Ak[:m, :m]))))
            print(f'Iter {i}: Active Size {m}, Off-diag(sub)={total_off_diag:.4f}, Split-elem={off_diagonal_val:.6f}')

        if off_diagonal_val < eps:
            print(f"  ==> Deflation! Row {m-1} converged at Iter {i}.")
            Ak[m-1, m-2] = 0
            Ak[m-2, m-1] = 0 
            m -= 1
            if m <= 1:
                print("  ==> All eigenvalues converged.")
                break
    
    if record:
        return Ak, transform_matrix, recorder
    else:
        return Ak, transform_matrix

# ==========================================
# Part 3: Main Execution
# ==========================================

def get_matrix_M():
    M = np.array([
        [-1/2, 25/12, 0, -10/12, 0, 0, 0, 0],
        [0, -10/12, -10/12, 50/12, 0, -15/12, 0, 0],
        [0, 0, 0, -15/12, -8/12, 50/12, 0, -15/12],
        [0, 0, 0, 0, 0, -15/12, -4/12, 25/12],
        [25/12, -1/2, -10/12, 0, 0, 0, 0, 0],
        [-10/12, 0, 50/12, -10/12, -15/12, 0, 0, 0],
        [0, 0, -15/12, 0, 50/12, -8/12, -15/12, 0],
        [0, 0, 0, 0, -15/12, 0, 25/12, -4/12]
    ])
    zero_8x8 = np.zeros((8, 8))
    M_tilde = np.block([
        [zero_8x8, M],
        [M.T, zero_8x8]
    ])
    return M_tilde

if __name__ == "__main__":
    matrix_to_solve = get_matrix_M()
    print(f"Matrix shape: {matrix_to_solve.shape}")
    
    max_iter = 200  
    convergence_eps = 5e-2 
    
    print("\nStarting Quantum QR Algorithm...")
    print("Method: Quantum Simulation (Statevector)")
    print("Strategy: Wilkinson Shift + Deflation + Basis Completion (Fixed V2)")
    print("-" * 40)
    
    final_matrix, _ = qr_algorithm(
        matrix_to_solve, 
        choosen_method='quantum', 
        max_iteration=max_iter, 
        eps=convergence_eps,
        use_shift=True
    )
    
    eigenvalues = np.diag(final_matrix)
    sorted_eigenvalues = np.sort(eigenvalues.real)
    
    print("\n" + "=" * 40)
    print("Calculation Finished.")
    print("=" * 40)
    print("Calculated Eigenvalues (Real parts):")
    with np.printoptions(precision=4, suppress=True):
        print(sorted_eigenvalues)
    
    print("-" * 40)
    print("Reference Eigenvalues (numpy.linalg.eigvals):")
    ref_eigs = np.linalg.eigvals(matrix_to_solve)
    with np.printoptions(precision=4, suppress=True):
        print(np.sort(ref_eigs.real))