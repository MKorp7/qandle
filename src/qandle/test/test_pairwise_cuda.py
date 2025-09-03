import pytest
import torch
import qandle


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("gate_cls", [qandle.RX, qandle.RY, qandle.RZ])
@pytest.mark.parametrize("batched", [False, True])
def test_pairwise_rotation_cuda_matches_matrix(gate_cls, batched):
    device = torch.device("cuda")
    torch.manual_seed(0)
    num_q = 3
    theta_val = torch.tensor(0.321)
    gate = (
        gate_cls(qubit=1, theta=theta_val, remapping=None)
        .build(num_qubits=num_q)
        .to(device)
    )
    if batched:
        state = torch.rand(5, 2**num_q, dtype=torch.cfloat, device=device)
    else:
        state = torch.rand(2**num_q, dtype=torch.cfloat, device=device)
    out_forward = gate(state)
    assert out_forward.is_cuda
    mat = gate.to_matrix().to(device)
    out_matrix = state @ mat
    assert torch.allclose(out_forward, out_matrix)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("gate_cls", [qandle.RX, qandle.RY, qandle.RZ])
@pytest.mark.parametrize("batched", [False, True])
def test_pairwise_rotation_cpu_cuda_parity(gate_cls, batched):
    device = torch.device("cuda")
    torch.manual_seed(0)
    num_q = 3
    theta_val = torch.tensor(0.123)
    gate_cpu = gate_cls(qubit=1, theta=theta_val.clone(), remapping=None).build(num_qubits=num_q)
    gate_cuda = (
        gate_cls(qubit=1, theta=theta_val.clone(), remapping=None)
        .build(num_qubits=num_q)
        .to(device)
    )
    if batched:
        state_cpu = torch.rand(5, 2**num_q, dtype=torch.cfloat)
    else:
        state_cpu = torch.rand(2**num_q, dtype=torch.cfloat)
    state_cuda = state_cpu.to(device)
    out_cpu = gate_cpu(state_cpu)
    out_cuda = gate_cuda(state_cuda)
    assert torch.allclose(out_cpu, out_cuda.cpu())