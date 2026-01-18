import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1) Payoffs - Prisoner's Dilemma
# ============================================================

R = 3
T = 5
P = 1
S = 0

PAYOFF_AGENT = {
    (0, 0): R,
    (0, 1): S,
    (1, 0): T,
    (1, 1): P
}

# ============================================================
# 2) Etats
# ============================================================

START = 2  # "pas d'action précédente"

def last_pair_to_index(a_last, o_last):
    """
    0: START/START
    1: (C,C)
    2: (C,D)
    3: (D,C)
    4: (D,D)
    """
    if a_last == START and o_last == START:
        return 0
    if a_last == 0 and o_last == 0:
        return 1
    if a_last == 0 and o_last == 1:
        return 2
    if a_last == 1 and o_last == 0:
        return 3
    if a_last == 1 and o_last == 1:
        return 4
    raise ValueError(f"Couple (a_last={a_last}, o_last={o_last}) invalide.")

def make_state_index(opp_id, a_last, o_last):
    pair_idx = last_pair_to_index(a_last, o_last)  # 0..4
    return opp_id * 5 + pair_idx

NUM_OPP = 5
NUM_STATES = NUM_OPP * 5  # 25
NUM_ACTIONS = 2

# ============================================================
# 3) Q-Learning Agent
# ============================================================

class QLearningAgent:
    def __init__(self, alpha=0.10, gamma=0.95, epsilon=0.50, seed=123):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=float)

    def greedy_action(self, s_idx):
        q = self.Q[s_idx]
        best_actions = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best_actions))  # tie-break comme ton code

    def select_action(self, s_idx):
        a_star = self.greedy_action(s_idx)
        if self.rng.random() < self.epsilon:
            other_actions = [a for a in range(NUM_ACTIONS) if a != a_star]
            if other_actions:
                return int(self.rng.choice(other_actions))
            return a_star
        return a_star

    def update(self, s_idx, action, reward, s_next_idx):
        target = reward + self.gamma * np.max(self.Q[s_next_idx])
        self.Q[s_idx, action] += self.alpha * (target - self.Q[s_idx, action])

# ============================================================
# 4) Duel A vs B (B gelé)
# ============================================================

def play_match_dual(agent_A, agent_B, opp_id, rounds=50, learn_A=True):
    a_last = START
    b_last = START

    gain_A = 0.0
    gain_B = 0.0

    for t in range(rounds):
        sA = make_state_index(opp_id, a_last, b_last)
        sB = make_state_index(opp_id, b_last, a_last)

        a = agent_A.select_action(sA)
        b = agent_B.greedy_action(sB)

        rA = PAYOFF_AGENT[(a, b)]
        rB = PAYOFF_AGENT[(b, a)]

        gain_A += rA
        gain_B += rB

        sA_next = make_state_index(opp_id, a, b)

        if learn_A:
            agent_A.update(sA, a, rA, sA_next)

        a_last, b_last = a, b

    return gain_A, gain_B

# ============================================================
# 5) Observation ex post : taux de coopération
# (IMPORTANT) On reproduit EXACTEMENT la logique d'actions:
# - A: select_action (donc epsilon-greedy comme pendant duel)
# - B: greedy_action (gelé)
# Et learn_A=False (aucune modif Q).
# ============================================================

def observe_duel_coop(agent_A, agent_B, opp_id, rounds=50, episodes_obs=100):
    coopA_rates = []
    coopB_rates = []

    for _ in range(episodes_obs):
        a_last = START
        b_last = START
        coopA = 0
        coopB = 0

        for t in range(rounds):
            sA = make_state_index(opp_id, a_last, b_last)
            sB = make_state_index(opp_id, b_last, a_last)

            a = agent_A.select_action(sA)      # IDENTIQUE au duel
            b = agent_B.greedy_action(sB)      # IDENTIQUE au duel

            if a == 0:
                coopA += 1
            if b == 0:
                coopB += 1

            a_last, b_last = a, b

        coopA_rates.append(coopA / rounds)
        coopB_rates.append(coopB / rounds)

    return coopA_rates, coopB_rates

# ============================================================
# 6) Main
# ============================================================

if __name__ == "__main__":

    data = np.load("trained_agent_qtable.npz")

    Q_loaded = data["Q"]
    alpha = float(data["alpha"])
    gamma = float(data["gamma"])
    epsilon = float(data["epsilon"])

    if Q_loaded.shape == (20, 2):
        Q_ext = np.zeros((25, 2), dtype=float)
        Q_ext[:20, :] = Q_loaded
        Q_loaded = Q_ext

    if Q_loaded.shape != (25, 2):
        raise ValueError(f"Q-table inattendue: {Q_loaded.shape}. Attendu (20,2) ou (25,2).")

    agent_A = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, seed=123)
    agent_B = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=0.0, seed=999)  # gelé

    agent_A.Q = Q_loaded.copy()
    agent_B.Q = Q_loaded.copy()

    SELF_OPP_ID = 4
    episodes = 200
    rounds = 50

    gains_A = []
    gains_B = []
    avg_gain = []

    for ep in range(episodes):
        gA, gB = play_match_dual(
            agent_A,
            agent_B,
            opp_id=SELF_OPP_ID,
            rounds=rounds,
            learn_A=True
        )

        gains_A.append(gA)
        gains_B.append(gB)
        avg_gain.append((gA + gB) / 2.0)

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}/{episodes} | Gain A={gA:.2f} | Gain B={gB:.2f}")

    # --- Plot gains ---
    x = np.arange(1, episodes + 1)
    plt.figure(figsize=(11, 6))
    plt.plot(x, gains_A, label="Agent A (apprend)")
    plt.plot(x, gains_B, label="Agent B (gelé)")
    plt.xlabel("Épisodes")
    plt.ylabel(f"Gain total (sur {rounds} manches)")
    plt.title("Gains par épisode – Duel Agent A vs Agent B (gelé, même Q au départ)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Plot taux de coopération (observation ex post) ---
    # IMPORTANT: on n'altère pas les agents du duel: on observe APRES.
    # Pour garder la même logique que pendant le duel, on garde epsilon de A tel quel.
    coopA, coopB = observe_duel_coop(
        agent_A,
        agent_B,
        opp_id=SELF_OPP_ID,
        rounds=rounds,
        episodes_obs=100
    )

    xo = np.arange(1, len(coopA) + 1)
    plt.figure(figsize=(11, 5))
    plt.plot(xo, coopA, label="Agent A (apprend)")
    plt.plot(xo, coopB, label="Agent B (gelé)")
    plt.xlabel("Épisodes d'observation")
    plt.ylabel("Taux de coopération")
    plt.title("Taux de coopération (même politique que le duel) – Agent A vs Agent B")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
