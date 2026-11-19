"""
TabuRL-FS: Tabu search feature selection guided by a reinforcement-learning policy.

At every iteration the current subset (exactly ``top_k`` features) is perturbed by
swap moves (remove one selected feature, add one unselected feature). Candidate
moves are proposed by a Transformer policy network and, if needed, padded with
uniformly random moves. Every candidate subset is scored by repeated stratified
k-fold cross-validation (balanced accuracy) of the wrapped classifier; the best
candidate becomes the new current solution. Recently removed features are kept
on a tabu list so they cannot be re-added for ``tabu_tenure`` iterations. The
policy is trained online with PPO using the change in CV fitness as reward.
"""
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import clone
from sklearn.feature_selection import f_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ==============================================================================
# Fitness: repeated stratified 5-fold x 5 CV balanced accuracy
# ==============================================================================

def evaluate_fitness(solution, x, y, model):
    """Mean balanced accuracy of ``model`` over 5-fold x 5-repeat stratified CV on the selected columns."""
    selected = np.where(solution == 1)[0]
    if len(selected) == 0:
        return 0.0
    x_sub = x[:, selected]
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    scores = []
    try:
        for tr, te in rkf.split(x_sub, y):
            clf = clone(model)
            if hasattr(clf, "random_state"):
                clf.random_state = 42
            scaler = StandardScaler()
            clf.fit(scaler.fit_transform(x_sub[tr]), y[tr])
            scores.append(balanced_accuracy_score(y[te], clf.predict(scaler.transform(x_sub[te]))))
        return float(np.mean(scores))
    except Exception:
        return 0.0


def evaluate_fitness_batch(solutions, x, y, model):
    return [evaluate_fitness(s, x, y, model) for s in solutions]


def top_p_logits_mask(logits, p=0.9):
    """Nucleus (top-p) masking of logits; p >= 1 disables masking."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cum_probs > p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(dim=-1, index=sorted_idx, src=remove)
    return logits.masked_fill(mask, -1e9)


# ==============================================================================
# Policy / value network
# ==============================================================================

class TransformerFeatureSelector(nn.Module):
    """
    Each feature is a token described by 7 state channels:
    [selected, tabu, in best subset, normalized F-score,
     mean |r| to selected set, max |r| to selected set, min |r| to selected set].
    Outputs per-feature remove logits, per-feature add logits and a state value.
    """

    def __init__(self, num_features, embed_dim=128, dropout_rate=0.1, nhead=4,
                 num_encoder_layers=2, head_hidden_dim=256):
        super().__init__()
        self.num_features = num_features
        self.input_proj = nn.Sequential(nn.Linear(7, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=head_hidden_dim,
            dropout=dropout_rate, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers,
                                                 enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(embed_dim)

        def head():
            return nn.Sequential(nn.Linear(embed_dim, head_hidden_dim // 2), nn.GELU(),
                                 nn.Linear(head_hidden_dim // 2, 1))

        self.remove_head = head()
        self.add_head = head()
        self.value_head = head()

    def forward(self, state):
        h = self.final_norm(self.transformer(self.input_proj(state)))
        remove_logits = self.remove_head(h).squeeze(-1)
        add_logits = self.add_head(h).squeeze(-1)
        value = self.value_head(h.mean(dim=1)).squeeze(-1)
        return remove_logits, add_logits, value


# ==============================================================================
# TabuRL-FS
# ==============================================================================

def taburl_fs(
    x, y, model, top_k, max_iterations=3000,
    neighborhood_size=100, tabu_tenure=10,
    lr=1e-4, gamma=0.95, gae_lambda=0.95, clip_param=0.2, target_kl=0.015,
    update_every=20, rl_epochs=20, reward_scale=1.0, mlp_moves_ratio=1.0,
    embed_dim=256, nhead=8, num_encoder_layers=2, head_hidden_dim=256, dropout_rate=0.1,
    early_stop_patience=None, n_jobs=-1, device="cpu", verbose=True,
):
    """
    Run TabuRL-FS on ``x`` (n_samples x n_features numpy array) and ``y`` (integer labels).

    Returns (selected column indices, best CV fitness, best-fitness history).
    The caller is expected to call ``seed_everything`` beforehand for reproducibility.
    """
    x = np.ascontiguousarray(x)
    y = np.asarray(y)
    num_features = x.shape[1]
    n_workers = effective_n_jobs(n_jobs)
    if early_stop_patience is None:
        early_stop_patience = max_iterations
    log = print if verbose else (lambda *a, **k: None)

    # Static per-feature prior: min-max normalized ANOVA F-score
    f_scores = np.nan_to_num(f_classif(x, y)[0])
    static_scores = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-8)

    device = torch.device(device if (not str(device).startswith("cuda") or torch.cuda.is_available()) else "cpu")
    log(f"Device: {device}")
    corr_tensor = torch.tensor(np.abs(np.nan_to_num(np.corrcoef(x, rowvar=False))),
                               dtype=torch.float32, device=device)
    static_scores_tensor = torch.tensor(static_scores, dtype=torch.float32, device=device).unsqueeze(0)

    # Initial subset sampled proportionally to the F-score
    prob_init = (static_scores + 1e-6) / (static_scores + 1e-6).sum()
    current_solution = np.zeros(num_features, dtype=int)
    current_solution[np.random.choice(num_features, top_k, replace=False, p=prob_init)] = 1
    current_fitness = evaluate_fitness(current_solution, x, y, model)
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    log(f"Initial CV fitness: {current_fitness:.4f}")

    policy = TransformerFeatureSelector(num_features, embed_dim=embed_dim, dropout_rate=dropout_rate,
                                        nhead=nhead, num_encoder_layers=num_encoder_layers,
                                        head_hidden_dim=head_hidden_dim).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=lr)

    tabu_list = {}  # feature index -> remaining tabu iterations
    buf_states, buf_masks, buf_actions, buf_logp, buf_rewards, buf_values = [], [], [], [], [], []
    no_improve = 0
    history_best = []

    # Annealing schedules for sampling temperature and entropy bonus
    init_temp, min_temp = 2.0, 0.5
    init_ent, min_ent = 0.05, 0.001
    p_remove, p_add = 1.0, 1.0  # top-p disabled

    def build_state(sol):
        sel_idx = np.where(sol == 1)[0]
        sub_corr = corr_tensor[:, torch.tensor(sel_idx, device=device)]
        self_mask = torch.zeros_like(sub_corr)
        self_mask[sel_idx, torch.arange(len(sel_idx))] = 1.0
        sub_corr_no_self = sub_corr * (1.0 - self_mask)
        avg_corr = sub_corr.mean(dim=1).unsqueeze(0)
        max_corr, _ = torch.max(sub_corr_no_self, dim=1)
        min_corr, _ = torch.min(sub_corr_no_self, dim=1)

        state_bin = torch.zeros((1, num_features), dtype=torch.float32, device=device)
        state_bin[0, sel_idx] = 1.0
        tabu_bin = torch.zeros((1, num_features), dtype=torch.float32, device=device)
        if tabu_list:
            tabu_bin[0, list(tabu_list.keys())] = 1.0
        best_bin = torch.zeros((1, num_features), dtype=torch.float32, device=device)
        best_idx = np.where(best_solution == 1)[0]
        if len(best_idx) > 0:
            best_bin[0, best_idx] = 1.0

        state = torch.stack([state_bin, tabu_bin, best_bin, static_scores_tensor, avg_corr,
                             max_corr.unsqueeze(0), min_corr.unsqueeze(0)], dim=-1)
        return state, state_bin, sel_idx

    for it in range(max_iterations):
        progress = min(1.0, it / (max_iterations * 0.8))
        temp = init_temp - progress * (init_temp - min_temp)
        ent_coef = init_ent - progress * (init_ent - min_ent)

        unselected_idx = np.where(current_solution == 0)[0]
        state, state_bin, selected_idx = build_state(current_solution)

        policy.eval()
        with torch.no_grad():
            remove_logits, add_logits, state_value = policy(state)
        remove_logits, add_logits, state_value = remove_logits.squeeze(0), add_logits.squeeze(0), state_value.squeeze(0)
        remove_logits[unselected_idx] = -1e9  # can only remove selected features
        add_logits[selected_idx] = -1e9       # can only add unselected features

        dist_remove = torch.distributions.Categorical(logits=top_p_logits_mask(remove_logits / temp, p=p_remove))
        dist_add = torch.distributions.Categorical(logits=top_p_logits_mask(add_logits / temp, p=p_add))

        # ---- Build the neighborhood of swap moves ----
        moves, candidates, seen = [], [], set()

        def add_move(r, a, log_p, source):
            seen.add((r, a))
            moves.append({"move": (r, a), "log_prob": log_p, "source": source})
            cand = current_solution.copy()
            cand[r] = 0
            cand[a] = 1
            candidates.append(cand)

        # 1) Greedy move from the policy
        best_r = torch.argmax(dist_remove.probs).item()
        best_a = torch.argmax(dist_add.probs).item()
        if best_r != best_a and best_a not in tabu_list:
            log_p = (dist_remove.log_prob(torch.tensor(best_r, device=device))
                     + dist_add.log_prob(torch.tensor(best_a, device=device)))
            add_move(int(best_r), int(best_a), log_p, "policy_greedy")

        # 2) Moves sampled from the policy
        target_policy = int(neighborhood_size * mlp_moves_ratio)
        remaining = target_policy - len(candidates)
        if remaining > 0:
            n_samples = remaining * 100
            r_samples = dist_remove.sample((n_samples,))
            a_samples = dist_add.sample((n_samples,))
            r_np, a_np = r_samples.cpu().numpy(), a_samples.cpu().numpy()
            for i in range(n_samples):
                if len(candidates) >= target_policy:
                    break
                r, a = int(r_np[i]), int(a_np[i])
                if r == a or a in tabu_list or (r, a) in seen:
                    continue
                add_move(r, a, dist_remove.log_prob(r_samples[i]) + dist_add.log_prob(a_samples[i]), "policy_sample")

        # 3) Pad with uniformly random moves
        tries = 0
        while len(candidates) < neighborhood_size and tries < 500:
            tries += 1
            r = np.random.choice(selected_idx)
            a = np.random.choice(unselected_idx)
            if a in tabu_list or (r, a) in seen:
                continue
            add_move(r, a, None, "random")

        if not candidates:
            continue

        # ---- Score all candidates in parallel ----
        chunks = np.array_split(np.arange(len(candidates)), min(len(candidates), n_workers))
        chunk_scores = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_fitness_batch)([candidates[i] for i in idx], x, y, model) for idx in chunks
        )
        scores = np.array([s for cs in chunk_scores for s in cs])

        # ---- Move to the best candidate (aspiration-free tabu search) ----
        best_i = int(np.argmax(scores))
        chosen = moves[best_i]
        move_r, move_a = chosen["move"]
        new_fitness = scores[best_i]
        current_solution[move_r] = 0
        current_solution[move_a] = 1

        # Update tabu list: decrement, expire, and make the removed feature tabu
        for f in [f for f, t in tabu_list.items() if t <= 1]:
            del tabu_list[f]
        for f in tabu_list:
            tabu_list[f] -= 1
        tabu_list[move_r] = tabu_tenure

        # ---- Reward shaping ----
        reward = np.clip((new_fitness - current_fitness) * reward_scale * 50.0, -2.0, 2.0)
        if new_fitness > best_fitness + 1e-6:
            reward += 2.0 + (new_fitness - best_fitness) * reward_scale * 10.0
            best_fitness = new_fitness
            best_solution = current_solution.copy()
            no_improve = 0
            log(f"TabuRL-FS iter {it + 1}: new best CV fitness {best_fitness:.4f} (temp {temp:.2f})", flush=True)
        else:
            recent_avg = np.mean(history_best[-20:]) if len(history_best) >= 20 else best_fitness
            if new_fitness > recent_avg:
                reward += 0.2
            no_improve += 1

        current_fitness = new_fitness
        history_best.append(best_fitness)
        if (it + 1) % 100 == 0:
            log(f"  [progress] iter {it + 1}/{max_iterations} | best CV fitness {best_fitness:.4f} | "
                f"{datetime.now().strftime('%H:%M:%S')}", flush=True)

        # Only policy-generated moves are stored for PPO
        if chosen["log_prob"] is not None:
            buf_states.append(state.clone().detach())
            buf_masks.append(state_bin.clone().detach())
            buf_actions.append(chosen["move"])
            buf_logp.append(chosen["log_prob"].detach())
            buf_rewards.append(reward)
            buf_values.append(state_value.detach())

        # ---- PPO update ----
        if len(buf_rewards) >= update_every:
            next_state, _, _ = build_state(current_solution)
            policy.eval()
            with torch.no_grad():
                _, _, next_value = policy(next_state)

            # Generalized advantage estimation
            values = [v.item() for v in buf_values] + [next_value.squeeze(0).item()]
            advantages, gae = [], 0
            for t in reversed(range(len(buf_rewards))):
                delta = buf_rewards[t] + gamma * values[t + 1] - values[t]
                gae = delta + gamma * gae_lambda * gae
                advantages.insert(0, gae)
            b_adv = torch.tensor(advantages, device=device, dtype=torch.float32)
            b_returns = b_adv + torch.tensor(values[:-1], device=device, dtype=torch.float32)
            if len(b_adv) > 1:
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            b_states = torch.cat(buf_states, dim=0)
            b_masks = torch.cat(buf_masks, dim=0)
            b_act_r = torch.tensor([a[0] for a in buf_actions], device=device)
            b_act_a = torch.tensor([a[1] for a in buf_actions], device=device)
            b_old_logp = torch.stack(buf_logp)

            policy.train()
            for _ in range(rl_epochs):
                r_logits, a_logits, v_pred = policy(b_states)
                r_logits = r_logits.masked_fill(b_masks == 0, -1e9)
                a_logits = a_logits.masked_fill(b_masks == 1, -1e9)
                d_r = torch.distributions.Categorical(logits=top_p_logits_mask(r_logits / temp, p=p_remove))
                d_a = torch.distributions.Categorical(logits=top_p_logits_mask(a_logits / temp, p=p_add))
                new_logp = d_r.log_prob(b_act_r) + d_a.log_prob(b_act_a)

                with torch.no_grad():
                    log_ratio = new_logp - b_old_logp
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                if target_kl is not None and approx_kl > 1.5 * target_kl:
                    break

                # Rescale remove-entropy so both heads contribute comparably
                scale = math.log(len(unselected_idx)) / math.log(max(len(selected_idx), 2))
                entropy = d_r.entropy() * scale + d_a.entropy()
                ratio = torch.exp(new_logp - b_old_logp)
                policy_loss = -torch.min(ratio * b_adv,
                                         torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * b_adv).mean()
                value_loss = F.mse_loss(v_pred, b_returns)
                loss = policy_loss + 1.0 * value_loss - ent_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                optimizer.step()

            for buf in (buf_states, buf_masks, buf_actions, buf_logp, buf_rewards, buf_values):
                buf.clear()
            if it > 0 and it % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if no_improve >= early_stop_patience or best_fitness >= 0.9999:
            log(f"Early stop at iter {it + 1}")
            break

    return np.where(best_solution == 1)[0], float(best_fitness), history_best
