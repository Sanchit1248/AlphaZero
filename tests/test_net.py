import os, sys, random
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import GoState
from net import PolicyValueNet, NetConfig, PRESETS, evaluate
from Modules.PolicyValueNetwork.dualHeadNetwork import compact_policy_value_network
from Modules.PolicyValueNetwork.residualBlock import alpha_go_residual_block
from Modules.PolicyValueNetwork.policyValueHead import policy_value_heads

TOL = dict(atol=1e-5, rtol=1e-4)


def random_states(size, n, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = GoState.initial(size, 0.5)
        for _ in range(rng.randint(0, 3 * size)):
            if s.terminal:
                break
            s = s.play(rng.choice(np.flatnonzero(s.legal_mask()[:-1]).tolist() or [size * size]))
        out.append(s)
    return out


def make_net(cfg=NetConfig(5, 2, 16)):
    torch.manual_seed(0)
    net = PolicyValueNet(cfg)
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.uniform_(m.weight, 0.5, 1.5)
            torch.nn.init.uniform_(m.bias, -0.5, 0.5)
    return net


def batch(size=5, n=8):
    return torch.stack([s.encode() for s in random_states(size, n)])


def test_matches_functional_network():
    net, x = make_net().train(), batch()
    p, v = net(x)
    fp, fv = compact_policy_value_network(x, **net.functional_params(), epsilon=net.cfg.eps)
    assert torch.allclose(p, fp, **TOL) and torch.allclose(v, fv, **TOL)
    print("ok  train mode == compact_policy_value_network")


def test_matches_parts():
    net, x = make_net().train(), batch()
    params = net.functional_params()
    h = torch.relu(net.trunk_bn(net.trunk_conv(x)))
    b, k = net.tower[0], params["residual_parameters"][0]
    assert torch.allclose(b(h), alpha_go_residual_block(h, **k, epsilon=net.cfg.eps), **TOL)
    h = net.trunk(x)
    p, v = net.heads(h)
    fp, fv = policy_value_heads(h, **params["head_parameters"], epsilon=net.cfg.eps)
    assert torch.allclose(p, fp, **TOL) and torch.allclose(v, fv, **TOL)
    print("ok  ResBlock == alpha_go_residual_block, heads == policy_value_heads")


def test_running_stats():
    net, x = make_net(), batch()
    before = net.trunk_bn.running_mean.clone()
    net.train()(x)
    after = net.trunk_bn.running_mean.clone()
    assert not torch.equal(before, after)
    net.eval()(x)
    assert torch.equal(after, net.trunk_bn.running_mean)
    print("ok  running stats update in train, frozen in eval")


def test_eval_batch_independent():
    net, x = make_net(), batch()
    net.train()(x)
    net.eval()
    with torch.no_grad():
        p, v = net(x)
        for i in range(len(x)):
            pi, vi = net(x[i:i + 1])
            assert torch.allclose(pi[0], p[i], **TOL) and torch.allclose(vi[0], v[i], **TOL)
    fp1, _ = compact_policy_value_network(x[:1], **net.functional_params(), epsilon=net.cfg.eps)
    fp, _ = compact_policy_value_network(x, **net.functional_params(), epsilon=net.cfg.eps)
    assert not torch.allclose(fp1[0], fp[0], **TOL)
    print("ok  eval output independent of batch (functional net is not)")


def test_shapes_presets():
    for size in (5, 7):
        net = PolicyValueNet(PRESETS[size]).eval()
        with torch.no_grad():
            p, v = net(batch(size, 4))
        assert p.shape == (4, size * size + 1) and v.shape == (4,)
        assert (v.abs() < 1).all()
        print(f"ok  {size}x{size} preset: {sum(t.numel() for t in net.parameters()):,} params")


def test_evaluate():
    net = make_net()
    s = GoState.initial(5, 0.5)
    for a in (12, 6, 7, 8):
        s = s.play(a)
    states = [GoState.initial(5, 0.5), s]
    priors, values = evaluate(net, states)
    assert priors.shape == (2, 26) and values.shape == (2,)
    assert np.allclose(priors.sum(1), 1, atol=1e-5)
    for st, pr in zip(states, priors):
        assert (pr[~st.legal_mask()] == 0).all() and pr[-1] > 0
    assert priors[1][[12, 6, 7, 8]].sum() == 0
    assert net.training
    print("ok  evaluate: masked softmax, pass allowed, mode restored")


def test_gradients():
    net, x = make_net().train(), batch()
    p, v = net(x)
    (p.logsumexp(1).mean() + v.pow(2).mean()).backward()
    missing = [n for n, t in net.named_parameters() if t.grad is None or not t.grad.any()]
    assert not missing, missing
    print("ok  every parameter receives gradient")


if __name__ == "__main__":
    test_matches_functional_network()
    test_matches_parts()
    test_running_stats()
    test_eval_batch_independent()
    test_shapes_presets()
    test_evaluate()
    test_gradients()
    print("ALL PASSED")
