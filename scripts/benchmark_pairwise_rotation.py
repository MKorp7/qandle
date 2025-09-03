import time
import torch
from networkx.utils.misc import pairwise

import qandle


def benchmark(num_qubits: int = 10, reps: int = 100):
    """Compare pairwise rotation against building full matrices each call.

    """
    theta = torch.tensor(0.321)
    gate = qandle.RX(qubit=1, theta=theta, remapping=None).build(num_qubits=num_qubits)

    state = torch.rand(2**num_qubits, dtype=torch.cfloat)
    state = state / torch.linalg.norm(state)

    gate(state)
    gate.get_matrix()
    pairwise_time = 0.0
    matrix_time = 0.0
    for _ in range(reps):
        start = time.perf_counter()
        o_p = gate(state)
        pairwise_time += time.perf_counter() - start
        start = time.perf_counter()
        mat = gate.get_matrix()
        o_c = state @ mat
        matrix_time += time.perf_counter() - start
        assert torch.allclose(o_p, o_c, atol=1e-6), "Diffrent results!"
    pairwise_time /= reps
    matrix_time /= reps
    print(
        f"Pairwise rotation: {pairwise_time:.6f}s, hydrated matrix: {matrix_time:.6f}s, "
        f"speedup: {matrix_time / pairwise_time:.2f}x"
    )
    return pairwise_time, matrix_time


if __name__ == "__main__":
    results = f"results_pairwise_rotation_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"Results will be saved to {results}")
    with open(results, "w") as f:
        f.write("num_qubits,pairwise_time,matrix_time,speedup,num_reps\n")
        for i in range(2, 15):
            print(f"Benchmarking {i} qubits:")
            pt, mt = benchmark(num_qubits=i, reps=1000)
            f.write(f"{i},{pt:.6f},{mt:.6f},{mt / pt:.2f},1000\n")
            print("-" * 40)

