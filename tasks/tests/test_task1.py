"""Автотесты к домашнему заданию 1.

Положите рядом файл submission.py с классами MyGCNConv и MySAGEConv и запустите:

    pip install pytest torch torch_geometric
    pytest tests/test_task1.py -v

Вывод тестов приложите к сданной работе.
"""
import pytest
import torch

torch_geometric = pytest.importorskip("torch_geometric")
from torch_geometric.nn import GCNConv, SAGEConv  # noqa: E402
from torch_geometric.utils import to_undirected  # noqa: E402

submission = pytest.importorskip("submission", reason="создайте submission.py со своими слоями")


def random_graph(n=40, p=0.12, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(n, n, generator=g) < p
    mask = torch.triu(mask, diagonal=1)
    edge_index = to_undirected(mask.nonzero().t().contiguous())
    x = torch.randn(n, dim, generator=g)
    return x, edge_index, n


def test_gcn_output_shape():
    x, edge_index, n = random_graph()
    layer = submission.MyGCNConv(x.size(1), 5)
    out = layer(x, edge_index)
    assert out.shape == (n, 5), f"ожидали форму ({n}, 5), получили {tuple(out.shape)}"


def test_gcn_matches_pyg():
    """Главный тест: с теми же весами выход обязан совпасть с GCNConv."""
    x, edge_index, _ = random_graph()
    mine = submission.MyGCNConv(x.size(1), 6)
    ref = GCNConv(x.size(1), 6, bias=False)
    with torch.no_grad():
        ref.lin.weight.copy_(mine.lin.weight)
        diff = (mine(x, edge_index) - ref(x, edge_index)).abs().max().item()
    assert diff < 1e-5, (
        f"расхождение с PyG {diff:.2e}. Проверьте петли, степени и нормировку 1/sqrt(d_u*d_v)"
    )


def test_gcn_on_directed_graph():
    """На направленном графе перепутанные row и col дают разный результат."""
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    mine = submission.MyGCNConv(4, 3)
    ref = GCNConv(4, 3, bias=False)
    with torch.no_grad():
        ref.lin.weight.copy_(mine.lin.weight)
        diff = (mine(x, edge_index) - ref(x, edge_index)).abs().max().item()
    assert diff < 1e-5, (
        f"на направленном графе расхождение {diff:.2e}: сообщения должны собираться "
        "по целевой вершине (второй строке edge_index)"
    )


def test_gcn_permutation_invariance():
    """Перенумерация вершин не должна менять результат."""
    x, edge_index, n = random_graph()
    layer = submission.MyGCNConv(x.size(1), 4)
    perm = torch.randperm(n)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(n)
    with torch.no_grad():
        out = layer(x, edge_index)
        out_perm = layer(x[perm], inverse[edge_index])
    diff = (out[perm] - out_perm).abs().max().item()
    assert diff < 1e-5, f"результат зависит от нумерации вершин, расхождение {diff:.2e}"


def test_gcn_isolated_node_is_finite():
    """У изолированной вершины степень нулевая — деления на ноль быть не должно."""
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out = submission.MyGCNConv(3, 3)(x, edge_index)
    assert torch.isfinite(out).all(), "в выходе есть NaN или inf: обработайте вершины без соседей"


def test_sage_matches_pyg():
    x, edge_index, _ = random_graph()
    mine = submission.MySAGEConv(x.size(1), 6)
    ref = SAGEConv(x.size(1), 6, aggr="mean", bias=False, root_weight=True)
    with torch.no_grad():
        ref.lin_l.weight.copy_(mine.lin_neigh.weight)
        ref.lin_r.weight.copy_(mine.lin_self.weight)
        diff = (mine(x, edge_index) - ref(x, edge_index)).abs().max().item()
    assert diff < 1e-5, (
        f"расхождение с SAGEConv {diff:.2e}. В GraphSAGE соседи усредняются, петли не добавляются, "
        "а собственный вектор идёт через отдельную матрицу"
    )


def test_gcn_is_trainable():
    """Градиенты должны доходить до весов."""
    x, edge_index, n = random_graph()
    layer = submission.MyGCNConv(x.size(1), 3)
    out = layer(x, edge_index)
    out.sum().backward()
    grad = layer.lin.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, (
        "градиент не дошёл до весов слоя"
    )
