"""The editors' numerical functions that the builder uses, copied unchanged."""
import numpy as np
import torch
import torch.nn.functional as Fn
import collections
from collections import Counter, defaultdict

def ln(x, w, b, eps=1e-6):
    return Fn.layer_norm(x, (x.shape[-1],), w, b, eps)

def ref_block27(x, R, nh=16):
    # x: (B, T, 1152)
    pre = 'vision_model.encoder.layers.26.'
    B, T, D = x.shape; hd = D // nh
    h = ln(x, R[pre+'layer_norm1.weight'], R[pre+'layer_norm1.bias'])
    q = (h @ R[pre+'self_attn.q_proj.weight'].T + R[pre+'self_attn.q_proj.bias']).view(B, T, nh, hd).transpose(1, 2)
    k = (h @ R[pre+'self_attn.k_proj.weight'].T + R[pre+'self_attn.k_proj.bias']).view(B, T, nh, hd).transpose(1, 2)
    v = (h @ R[pre+'self_attn.v_proj.weight'].T + R[pre+'self_attn.v_proj.bias']).view(B, T, nh, hd).transpose(1, 2)
    att = torch.softmax((q @ k.transpose(-1, -2)) / hd**0.5, -1)
    o = (att @ v).transpose(1, 2).reshape(B, T, D)
    x = x + o @ R[pre+'self_attn.out_proj.weight'].T + R[pre+'self_attn.out_proj.bias']
    h = ln(x, R[pre+'layer_norm2.weight'], R[pre+'layer_norm2.bias'])
    h = Fn.gelu(h @ R[pre+'mlp.fc1.weight'].T + R[pre+'mlp.fc1.bias'], approximate='tanh')
    x = x + h @ R[pre+'mlp.fc2.weight'].T + R[pre+'mlp.fc2.bias']
    return x

def ref_head(x, R, nh=16, return_attn=False):
    # x: output of block27 (B,T,D)
    B, T, D = x.shape; hd = D // nh
    x = ln(x, R['vision_model.post_layernorm.weight'], R['vision_model.post_layernorm.bias'])
    Wi = R['vision_model.head.attention.in_proj_weight']; bi = R['vision_model.head.attention.in_proj_bias']
    Wq, Wk, Wv = Wi[:D], Wi[D:2*D], Wi[2*D:]; bq, bk, bv = bi[:D], bi[D:2*D], bi[2*D:]
    probe = R['vision_model.head.probe'].reshape(1, 1, D).expand(B, 1, D)
    q = (probe @ Wq.T + bq).view(B, 1, nh, hd).transpose(1, 2)
    k = (x @ Wk.T + bk).view(B, T, nh, hd).transpose(1, 2)
    v = (x @ Wv.T + bv).view(B, T, nh, hd).transpose(1, 2)
    att = torch.softmax((q @ k.transpose(-1, -2)) / hd**0.5, -1)       # (B,nh,1,T)
    o = (att @ v).transpose(1, 2).reshape(B, 1, D)
    a = o @ R['vision_model.head.attention.out_proj.weight'].T + R['vision_model.head.attention.out_proj.bias']
    h = ln(a, R['vision_model.head.layernorm.weight'], R['vision_model.head.layernorm.bias'])
    h = Fn.gelu(h @ R['vision_model.head.mlp.fc1.weight'].T + R['vision_model.head.mlp.fc1.bias'], approximate='tanh')
    out = (a + h @ R['vision_model.head.mlp.fc2.weight'].T + R['vision_model.head.mlp.fc2.bias'])[:, 0]
    return (out, att[:, :, 0, :]) if return_attn else out

def ridge_fit(X, Y, lam):
    mu = X.mean(0); Xc = X - mu; ym = Y.mean(0)
    # dual form when n < d
    K = Xc @ Xc.T
    al = np.linalg.solve(K + lam * np.trace(K) / len(K) * np.eye(len(K)), Y - ym)
    W = Xc.T @ al               # (d, k)
    return W, ym - mu @ W

def fit_detector3(w_reg=2000.0, lam=1e-3, zero_pos=()):
    # 569 -> 1, normal tokens -> 0 ; other registers unconstrained (removed from normal moment) unless in zero_pos
    A = Sxx_fit.copy()
    for q in REGPOS: A -= regs[q].T @ regs[q]
    A = A / n_fit
    R_ = regs[569]
    A += w_reg * (R_.T @ R_) / n_fit
    for q in zero_pos: A += w_reg * (regs[q].T @ regs[q]) / n_fit
    rhs = w_reg * R_.sum(0) / n_fit
    A += lam * np.trace(A) / 1152 * np.eye(1152)
    return np.linalg.solve(A, rhs)

def fit_base_dir2(X, y, lam, mu_n):
    w = np.where(y > 0, 0.5 / (y > 0).sum(), 0.5 / (y < 0).sum())
    mu = (w[:, None] * X).sum(0); Xc = X - mu
    A = (Xc * w[:, None]).T @ Xc
    A = A + mu_n * Sn + lam * np.trace(A) / X.shape[1] * np.eye(X.shape[1])
    u = np.linalg.solve(A, (Xc * w[:, None]).T @ y)
    return u, -mu @ u

def fit_q(mu_n=1e-2, lam=1e-3):
    X = np.concatenate([reg569, reg568, reg542]); y = np.concatenate([np.ones(len(reg569)), np.zeros(len(reg568)), np.zeros(len(reg542))])
    w = np.concatenate([np.full(len(reg569), 0.5 / len(reg569)), np.full(len(reg568), 0.3 / len(reg568)), np.full(len(reg542), 0.2 / len(reg542))])
    A = (X * w[:, None]).T @ X + mu_n * Sn; A += lam * np.trace(A) / 1152 * np.eye(1152)
    return np.linalg.solve(A, (X * w[:, None]).T @ y)

def fit_alias(H, n, tpos=4.0, tneg=-5.0, lam=1e-2, wpos=None):
    y = np.where(n == 1, tpos, tneg)
    w = np.where(n == 1, 0.5 / max((n == 1).sum(), 1), 0.5 / (n != 1).sum())      # balance classes
    A = (H * w[:, None]).T @ H; A += lam * np.trace(A) / 896 * np.eye(896)
    return np.linalg.solve(A, (H * w[:, None]).T @ y)

def gelu_np(z):
    from scipy.special import erf
    return 0.5 * z * (1 + erf(z / np.sqrt(2)))

def build_ktest(T_base, T_tile, G=60.0, tau=0.8, Gb=10.0, Gr=300.0, kill_margin=6.0):
    W1 = W1o.copy(); b1 = b1o.copy(); W2 = W2o.copy(); b2 = b2o.copy()
    S = [int(k) for k in order[:3]]
    # kill native units at 569
    d = np.where(pre_max > -kill_margin, pre_max + kill_margin, 0.0); d[S] = 0.0
    W1 -= np.outer(d, r569)
    k_on, k_b1, k_b2 = S
    W1[k_on] = G * r569; b1[k_on] = -G * tau
    a_on = gelu_np(np.array(G * (1 - tau)))
    W2[:, k_on] = (T_tile - b2) / a_on
    row = Gb * u_base + Gr * r569
    W1[k_b1] = row; b1[k_b1] = Gb * c_base - Gr
    W1[k_b2] = row; b1[k_b2] = Gb * c_base - Gr - 1.0
    W2[:, k_b1] = (T_base - T_tile); W2[:, k_b2] = -(T_base - T_tile)
    return W1, b1, W2, b2, dict(S=S, n_killed=int((d > 0).sum()), dmax=float(d.max()))

def hint_vec(name):
    s = custom.get(name, ' ' + name)
    ids = toks(s); v = E0[ids].sum(0)
    return v / np.linalg.norm(v) * 0.43, ids

def build_r2b(Wm, bm, Tm, HV, theta=-6.0, kappa=1.0, Gr=200.0, Gb=4.0, A=1.0, kill_margin=6.0, base_center=0.87):
    W1 = W1o.copy(); b1 = b1o.copy(); W2 = W2o.copy(); b2 = b2o.copy()
    S = [int(k) for k in order[:58]]
    d = np.where(pre_max > -kill_margin, pre_max + kill_margin, 0.0); d[S] = 0.0
    W1 -= np.outer(d, r569)
    Wc = SCALE * (Wm @ Tm.T); bc = SCALE * (bm @ Tm.T) + BIAS
    for i, k in enumerate(S):
        W1[k] = kappa * Wc[:, i] + Gr * r569 + Gb * u_base
        b1[k] = kappa * (bc[i] - theta) - Gr + Gb * (c_base - base_center)
        W2[:, k] = A * HV[i]
    return W1, b1, W2, b2, dict(S=S)

def class_tokens(name):
    s = custom.get(name, ' ' + name); ids = toks(s)
    if name == 'opossum': return ids[1], ids[2]          # 'pos','sum' -> possum
    if len(ids) == 1: return ids[0], ids[0]
    if len(ids) == 2: return ids[0], ids[1]
    return ids[0], ids[1]                                 # first two tokens for longer names

def unitvec(i): return E0[i] / np.linalg.norm(E0[i]) * 0.43

def build_r3(theta=-5.5, A=0.25, Gr=200.0, Gb=4.0, G8=200.0, M8=200.0, kill_margin=6.0, Tm=None, copies568_single=True):
    Tm = Tm20 if Tm is None else Tm
    W1 = W1o.copy(); b1 = b1o.copy(); W2 = W2o.copy(); b2 = b2o.copy()
    nun = 58 + 58 + 2
    S = [int(k) for k in order[:nun]]
    d = np.where(pre_max > -kill_margin, pre_max + kill_margin, 0.0); d[S] = 0.0
    W1 -= np.outer(d, r569)
    Wc9 = SCALE * (Wm569 @ Tm.T); bc9 = SCALE * (bm569 @ Tm.T) + BIAS
    Wc8 = SCALE * (Wm568 @ Tm.T); bc8 = SCALE * (bm568 @ Tm.T) + BIAS
    for i in range(58):
        k = S[i]
        W1[k] = Wc9[:, i] + Gr * r569 + Gb * u_base; b1[k] = (bc9[i] - theta) - Gr + Gb * (c_base - 0.87); W2[:, k] = A * H_last[i]
        k = S[58 + i]
        W1[k] = Wc8[:, i] + G8 * r569 - M8 * q568 + Gb * u568; b1[k] = (bc8[i] - theta) - G8 * 0.5486 + Gb * (c568 - 0.86)
        single = (TOK2[names58[i]][0] == TOK2[names58[i]][1])
        W2[:, k] = A * H_prev[i] * (1.0 if (copies568_single or not single) else 0.0)
    k1, k2 = S[116], S[117]
    W1[k1] = 200.0 * r569; b1[k1] = -200.0 * 0.5; W2[:, k1] = -b2
    W1[k2] = 200.0 * r569; b1[k2] = -200.0 * 0.5 - 1.0; W2[:, k2] = +b2
    return W1, b1, W2, b2, dict(S=S)

def fit_combo4(tr, ttr=None, lam=1e-2, kappa=0.2, lam_s=0.1, w_out=0.1, min_n=3, w_tile=1.0, tau=0.30, unseen_margin=0.5):
    """as fit_combo3 but unseen classes: threshold = max negative zero-shot logit + margin*sd  (never fires on training crops)"""
    X = X9b[tr]; y = Yidx[tr]; inl = INL[tr]; hv = hasVb[tr]
    Wm, bm = ridge_fit(X[hv], Vb[tr][hv], lam)
    Tm = T58n.copy(); gm = Vb[tr][hv].mean(0)
    for k in range(58):
        mm = (y[hv] == k)
        if kappa > 0 and mm.sum() >= 4: Tm[k] = Tm[k] + kappa * (Vb[tr][hv][mm].mean(0) - gm)
    Wz = SCALE * (Wm @ Tm.T); bz = SCALE * (bm @ Tm.T) + BIAS
    Lz = X @ Wz + bz; lab = y >= 0
    pos = np.zeros_like(inl); pos[np.where(lab)[0], y[lab]] = True; neg = inl & ~pos & lab[:, None]
    lp_ = Lz[pos]; ln_ = Lz[neg]
    xx = np.concatenate([lp_, ln_]); tt = np.concatenate([np.ones(len(lp_)), np.zeros(len(ln_))]); ww = np.concatenate([np.full(len(lp_), 0.5 / len(lp_)), np.full(len(ln_), 0.5 / len(ln_))])
    mx = (ww * xx).sum(); mt = (ww * tt).sum(); alpha = (ww * (xx - mx) * (tt - mt)).sum() / (ww * (xx - mx) ** 2).sum(); beta = mt - alpha * mx
    Wc = alpha * Wz; bc_ = alpha * bz + beta
    seen = np.array([(y == k).sum() >= min_n for k in range(58)])
    for k in np.where(~seen)[0]:
        thr = Lz[:, k].max() + unseen_margin * Lz[:, k].std(); Wc[:, k] = alpha * Wz[:, k]; bc_[k] = alpha * (bz[k] - thr) + tau
    Pz = X @ Wc + bc_
    w = np.zeros(len(y))
    for k in np.where(seen)[0]: w[y == k] = 1.0 / (y == k).sum()
    keep = w > 0; Xk = X[keep]; yk = y[keep]; wk = w[keep] / w[keep].sum(); ink = inl[keep]; Pk = Pz[keep]
    has_t = ttr is not None and len(ttr) > 0
    if has_t: Xt = TILE9[ttr]; Pt = Xt @ Wc + bc_; wt = np.full(len(ttr), w_tile / len(ttr))
    dW = np.zeros((1152, 58)); db = np.zeros(58)
    for k in range(58):
        if seen[k]:
            wk2 = wk * np.where((yk == k) | ink[:, k], 1.0, w_out); wk2 = wk2 / wk2.sum(); t = (yk == k).astype(float) - Pk[:, k]
        else:
            if not has_t: continue
            wk2 = wk.copy(); t = np.zeros(len(yk))
        if has_t:
            Xa = np.concatenate([Xk, Xt]); wa = np.concatenate([wk2, wt]); ta = np.concatenate([t, -Pt[:, k]]); wa = wa / wa.sum()
        else:
            Xa = Xk; wa = wk2; ta = t
        mu = (wa[:, None] * Xa).sum(0); Xc = Xa - mu
        A_ = (Xc * wa[:, None]).T @ Xc; A_ += lam_s * np.trace(A_) / 1152 * np.eye(1152)
        tb = (wa * ta).sum(); ws = np.linalg.solve(A_, (Xc * wa[:, None]).T @ (ta - tb)); dW[:, k] = ws; db[k] = tb - mu @ ws
    return Wc + dW, bc_ + db, seen

def fit_mimic8b(tr, ttr, W9, b9, lam=1e-2, w_tile=1.0, w54=0.3):
    S9tr = X9b[tr] @ W9 + b9
    parts = [(X8b[tr], S9tr, 1.0)]
    if len(ttr): parts.append((TILE8[ttr], np.zeros((len(ttr), 58)), w_tile))
    if w54 > 0:
        parts.append((P541all, np.zeros((len(P541all), 58)), w54)); parts.append((P542all, np.zeros((len(P542all), 58)), w54))
    Xa = np.concatenate([p[0] for p in parts]); Sa = np.concatenate([p[1] for p in parts]); wa = np.concatenate([np.full(len(p[0]), p[2] / len(p[0])) for p in parts]); wa = wa / wa.sum()
    mu = (wa[:, None] * Xa).sum(0); Xc = Xa - mu; sm = (wa[:, None] * Sa).sum(0)
    A_ = (Xc * wa[:, None]).T @ Xc; A_ += lam * np.trace(A_) / 1152 * np.eye(1152)
    W = np.linalg.solve(A_, (Xc * wa[:, None]).T @ (Sa - sm)); return W, sm - mu @ W

def build_n3(W9, b9, W8, b8, tau9=0.30, tau8=0.25, beta=12.0, A=0.6, Gb=3.0, c9=0.93, c8=0.93, Gr=200.0, G8=200.0, M8=125.0, kill_margin=6.0, m8vec=None, T8=None, T9=None):
    T8 = H_prev if T8 is None else T8; T9 = H_last if T9 is None else T9
    W1 = W1o.copy(); b1 = b1o.copy(); W2 = W2o.copy(); b2 = b2o.copy()
    S = [int(k) for k in order[:120]]
    d = np.where(pre_max > -kill_margin, pre_max + kill_margin, 0.0); d[S] = 0.0
    W1 -= np.outer(d, r569)
    for i in range(58):
        k = S[i]
        W1[k] = beta * W9[:, i] + Gr * r569 + Gb * u_base; b1[k] = beta * (b9[i] - tau9) - Gr + Gb * (c_base - c9); W2[:, k] = A * T9[i]
        k = S[58 + i]
        W1[k] = beta * W8[:, i] + G8 * r569 - M8 * q568 + Gb * u568; b1[k] = beta * (b8[i] - tau8) - G8 * 0.5486 + Gb * (c568 - c8); W2[:, k] = A * T8[i]
    k1, k2 = S[116], S[117]
    W1[k1] = 200.0 * r569; b1[k1] = -200.0 * 0.5; W2[:, k1] = -b2
    W1[k2] = 200.0 * r569; b1[k2] = -200.0 * 0.5 - 1.0; W2[:, k2] = +b2
    k3, k4 = S[118], S[119]
    for k, off in [(k3, 1.0), (k4, 0.0)]:
        W1[k] = G8 * r569 - M8 * q568; b1[k] = 10.0 + off - G8 * 0.5486
    mv = np.zeros(896) if m8vec is None else m8vec
    W2[:, k3] = -mv; W2[:, k4] = +mv
    return W1, b1, W2, b2

def final_h(norm_in):
    x = norm_in.astype(np.float64); return g_norm * x / np.sqrt((x ** 2).mean() + 1e-6)

def bank_from_records(rs):
    return dict(H=np.array([final_h(r['norm_in']) for r in rs]), n=np.array([len(r['names']) for r in rs]), yi=np.array([int(np.where(r['oi'] == r['y'])[0][0]) for r in rs]), cam=np.array([r['cam'] for r in rs]))

def cat_banks(bs): return {k: np.concatenate([b[k] for b in bs]) for k in ['H', 'n', 'yi', 'cam']}

def fit_readout_b(B, E, lam=1.0, multi_only=True, cam_balance=False):
    msk = (B['n'] >= 2) if multi_only else np.ones(len(B['n']), bool)
    H = B['H'][msk]; nn = B['n'][msk]; yi = B['yi'][msk]; cam = B['cam'][msk]
    lg = H @ E[LET_IDS].T; T = np.zeros((len(H), 11)); Wt = np.zeros((len(H), 11))
    sw = np.ones(len(H))
    if cam_balance:
        cnt = collections.Counter(cam); sw = np.array([1.0 / cnt[c] for c in cam]); sw = sw / sw.mean()
    for i in range(len(H)):
        n = nn[i]; p = np.exp(lg[i, :n] - lg[i, :n].max()); p /= p.sum(); t = -p; t[yi[i]] += 1.0; T[i, :n] = t; Wt[i, :n] = sw[i]
    R = np.zeros((11, 896))
    for j in range(11):
        w = Wt[:, j]
        if (w > 0).sum() < 5: continue
        A_ = (H * w[:, None]).T @ H; A_ += lam * np.trace(A_) / 896 * np.eye(896)
        R[j] = np.linalg.solve(A_, (H * w[:, None]).T @ T[:, j])
    return R

def mask_soft(z,n,temp=1):
    v=z/temp; v=v.copy(); v[np.arange(v.shape[1])[None,:]>=np.array(n)[:,None]]=-1000
    v-=v.max(1,keepdims=True); v=np.exp(v); return v/v.sum(1,keepdims=True)

def weighted_design(H,y,cam,n=None,power=.5):
    ct=Counter(cam); cyt=Counter(zip(cam,y))
    w=np.array([1/(ct[c]*cyt[(c,cl)])**(power/2) for c,cl in zip(cam,y)])
    return w/w.mean()

def fitres(H,Z,Y,n,cam,lam=1,kind='grad',temp=1):
    p=mask_soft(Z,n,temp); targ=np.eye(11)[Y]-p
    if kind=='logits':
        targ=(4*np.eye(11)[Y]-(Z-np.sum(p*Z,1,keepdims=True))) * (np.arange(11)[None,:]<n[:,None])
    w=weighted_design(H,Y,cam,n)
    mu=np.average(H,axis=0,weights=w); X=H-mu
    sw=np.sqrt(w); Xs=X*sw[:,None]; ymu=np.average(targ,axis=0,weights=w); Ts=(targ-ymu)*sw[:,None]
    K=Xs@Xs.T; ridge=lam*np.trace(K)/len(K)
    D=Xs.T@np.linalg.solve(K+ridge*np.eye(len(K)),Ts)
    return D,ymu-mu@D

def fit_combo3(tr, ttr=None, lam=1e-2, kappa=0.2, lam_s=0.1, w_out=0.1, min_n=3, w_tile=1.0, unseen_q=0.995, tau=0.45, ret_parts=False):
    """scores for cell 569. tr: bank indices (base). ttr: tile indices (negatives, target 0). Unseen classes: zero-shot with per-class threshold mapped to tau."""
    X = X9b[tr]; y = Yidx[tr]; inl = INL[tr]; hv = hasVb[tr]
    Wm, bm = ridge_fit(X[hv], Vb[tr][hv], lam)
    Tm = T58n.copy(); gm = Vb[tr][hv].mean(0)
    for k in range(58):
        mm = (y[hv] == k)
        if kappa > 0 and mm.sum() >= 4: Tm[k] = Tm[k] + kappa * (Vb[tr][hv][mm].mean(0) - gm)
    Wz = SCALE * (Wm @ Tm.T); bz = SCALE * (bm @ Tm.T) + BIAS
    Lz = X @ Wz + bz; lab = y >= 0
    pos = np.zeros_like(inl); pos[np.where(lab)[0], y[lab]] = True; neg = inl & ~pos & lab[:, None]
    lp_ = Lz[pos]; ln_ = Lz[neg]
    xx = np.concatenate([lp_, ln_]); tt = np.concatenate([np.ones(len(lp_)), np.zeros(len(ln_))]); ww = np.concatenate([np.full(len(lp_), 0.5 / len(lp_)), np.full(len(ln_), 0.5 / len(ln_))])
    mx = (ww * xx).sum(); mt = (ww * tt).sum(); alpha = (ww * (xx - mx) * (tt - mt)).sum() / (ww * (xx - mx) ** 2).sum(); beta = mt - alpha * mx
    Wc = alpha * Wz; bc_ = alpha * bz + beta
    seen = np.array([(y == k).sum() >= min_n for k in range(58)])
    # unseen classes: map per-class negative quantile to tau, slope alpha (same units)
    for k in np.where(~seen)[0]:
        thr = np.quantile(Lz[lab, k], unseen_q); Wc[:, k] = alpha * Wz[:, k]; bc_[k] = alpha * (bz[k] - thr) + tau
    Pz = X @ Wc + bc_
    w = np.zeros(len(y))
    for k in np.where(seen)[0]: w[y == k] = 1.0 / (y == k).sum()
    keep = w > 0; Xk = X[keep]; yk = y[keep]; wk = w[keep] / w[keep].sum(); ink = inl[keep]; Pk = Pz[keep]
    if ttr is not None and len(ttr) > 0:
        Xt = TILE9[ttr]; Pt = Xt @ Wc + bc_; wt = np.full(len(ttr), w_tile / len(ttr))
    dW = np.zeros((1152, 58)); db = np.zeros(58)
    for k in range(58):
        if seen[k]:
            wk2 = wk * np.where((yk == k) | ink[:, k], 1.0, w_out); wk2 = wk2 / wk2.sum(); t = (yk == k).astype(float) - Pk[:, k]
        else:
            if ttr is None or len(ttr) == 0: continue
            wk2 = wk.copy(); t = np.zeros(len(yk))
        if ttr is not None and len(ttr) > 0:
            Xa = np.concatenate([Xk, Xt]); wa = np.concatenate([wk2, wt]); ta = np.concatenate([t, -Pt[:, k]]); wa = wa / wa.sum()
        else:
            Xa = Xk; wa = wk2; ta = t
        mu = (wa[:, None] * Xa).sum(0); Xc = Xa - mu
        A_ = (Xc * wa[:, None]).T @ Xc; A_ += lam_s * np.trace(A_) / 1152 * np.eye(1152)
        tb = (wa * ta).sum(); ws = np.linalg.solve(A_, (Xc * wa[:, None]).T @ (ta - tb)); dW[:, k] = ws; db[k] = tb - mu @ ws
    return Wc + dW, bc_ + db

def proj_apply(X, W1_, b1_, W2_, b2_):
    X = torch.from_numpy(np.asarray(X, np.float32)); 
    h = torch.nn.functional.gelu(X @ torch.from_numpy(W1_.astype(np.float32)).T + torch.from_numpy(b1_.astype(np.float32)))
    return (h @ torch.from_numpy(W2_.astype(np.float32)).T + torch.from_numpy(b2_.astype(np.float32))).numpy()

def fit_mlp_delta(M,H,F,Y,n,cam,lam=1,targetscale=1):
    z=H@EL15.T/1.1; p=mask_soft(z,n)
    y=(np.eye(11)[Y]-p)*np.sqrt(np.mean(F**2,axis=1,keepdims=True))
    w=weighted_design(M,Y,cam); sw=np.sqrt(w); X=M*sw[:,None]
    A=X@X.T; rho=lam*np.trace(A)/len(A)
    D=X.T@np.linalg.solve(A+rho*np.eye(len(A)),y*sw[:,None]); return D

def fit_mlp_closed(M,F,H,Y,n,cam,lam=.1,center=True,power=.5,target='prob',temperature=1):
    z=H@EL15.T/1.1; p=mask_soft(z,n,temp=temperature); t=np.eye(11)[Y]-p
    if target=='margin':
        zw=z.copy(); zw[np.arange(len(Y)),Y]=-1000; zw[np.arange(11)[None,:]>=n[:,None]]=-1000
        mag=np.maximum(0,1+zw.max(1)-z[np.arange(len(Y)),Y]); t=t*mag[:,None]
    if target=='logit':
        mask=np.arange(11)[None,:]<n[:,None]; tt=(3*np.eye(11)[Y]-z)*mask; tt-=tt.sum(1,keepdims=True)/n[:,None]*mask; t=tt
    rms=np.sqrt(np.mean(F*F,axis=1,keepdims=True)+1e-6); t*=rms
    w=weighted_design(M,Y,cam,power=power); sw=np.sqrt(w)
    mu=np.average(M,axis=0,weights=w) if center else np.zeros(M.shape[1]); ym=np.average(t,axis=0,weights=w) if center else np.zeros(11)
    x=(M-mu)*sw[:,None]; K=x@x.T; rho=lam*np.trace(K)/len(K)
    d=x.T@np.linalg.solve(K+rho*np.eye(len(K)),(t-ym)*sw[:,None])
    if center: d+=np.outer(mu/(mu@mu),ym-mu@d)
    return d
