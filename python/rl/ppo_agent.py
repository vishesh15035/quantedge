import numpy as np
import pandas as pd
from typing import List, Tuple
from .trading_env import TradingEnv

class PolicyNetwork:
    """Lightweight policy network — pure numpy, no PyTorch needed"""
    def __init__(self, n_input: int, n_hidden: int, n_output: int):
        scale      = 0.1
        self.W1    = np.random.randn(n_hidden, n_input)  * scale
        self.b1    = np.zeros(n_hidden)
        self.W2    = np.random.randn(n_output, n_hidden) * scale
        self.b2    = np.zeros(n_output)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h  = np.tanh(self.W1 @ x + self.b1)
        out = self.W2 @ h + self.b2
        # Softmax
        out = out - out.max()
        exp = np.exp(out)
        return exp / exp.sum()

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W1.flatten(), self.b1,
                               self.W2.flatten(), self.b2])

    def set_params(self, params: np.ndarray):
        idx = 0
        s   = self.W1.size; self.W1 = params[idx:idx+s].reshape(self.W1.shape); idx+=s
        s   = self.b1.size; self.b1 = params[idx:idx+s]; idx+=s
        s   = self.W2.size; self.W2 = params[idx:idx+s].reshape(self.W2.shape); idx+=s
        self.b2 = params[idx:]


class PPOAgent:
    """
    PPO-style agent using Evolution Strategies for weight update
    (avoids PyTorch dependency while keeping RL concepts intact)
    """
    def __init__(self, n_features: int = 10, n_hidden: int = 64,
                 n_actions: int = 3, lr: float = 0.01, sigma: float = 0.05):
        self.policy = PolicyNetwork(n_features, n_hidden, n_actions)
        self.lr     = lr
        self.sigma  = sigma
        self.history = {"episode_returns": [], "episode_trades": []}

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        probs  = self.policy.forward(state)
        action = np.random.choice(len(probs), p=probs)
        return action, float(probs[action])

    def run_episode(self, env: TradingEnv) -> Tuple[float, List]:
        state     = env.reset()
        total_ret = 0.0
        trajectory = []
        done      = False
        while not done:
            action, prob = self.select_action(state)
            next_state, reward, done, info = env.step(action)
            trajectory.append((state, action, reward, prob))
            total_ret += reward
            state      = next_state
        return total_ret, trajectory

    def train(self, prices: pd.Series, n_episodes: int = 50,
              n_perturbations: int = 20) -> dict:
        print(f"\n[PPO Agent] Training on {len(prices)} days, {n_episodes} episodes")
        best_return = -np.inf
        best_params = self.policy.get_params().copy()

        for ep in range(n_episodes):
            base_params = self.policy.get_params()
            rewards     = []
            noises      = []

            # Evolution strategies: perturb + evaluate
            for _ in range(n_perturbations):
                noise = np.random.randn(len(base_params))
                self.policy.set_params(base_params + self.sigma * noise)
                env   = TradingEnv(prices)
                ret, _ = self.run_episode(env)
                rewards.append(ret)
                noises.append(noise)

            # Normalize rewards
            rewards = np.array(rewards)
            if rewards.std() > 1e-8:
                rewards = (rewards - rewards.mean()) / rewards.std()

            # Update params
            grad = sum(r * n for r, n in zip(rewards, noises))
            new_params = base_params + self.lr / (n_perturbations * self.sigma) * grad
            self.policy.set_params(new_params)

            # Evaluate
            env = TradingEnv(prices)
            ep_return, _ = self.run_episode(env)
            self.history["episode_returns"].append(ep_return)
            self.history["episode_trades"].append(env.trades)

            if ep_return > best_return:
                best_return = ep_return
                best_params = self.policy.get_params().copy()

            if ep % 10 == 0:
                print(f"  Episode {ep:3d} | Return: {ep_return:7.2f} "
                      f"| Trades: {env.trades:4d} "
                      f"| Best: {best_return:.2f}")

        self.policy.set_params(best_params)
        print(f"[PPO Agent] Training complete. Best return: {best_return:.2f}")
        return {"best_return": best_return, "episodes": n_episodes,
                "history": self.history}

    def evaluate(self, prices: pd.Series) -> dict:
        env      = TradingEnv(prices)
        state    = env.reset()
        done     = False
        actions  = []
        portfolio_history = [env.initial_capital]

        while not done:
            probs  = self.policy.forward(state)
            action = int(np.argmax(probs))   # greedy at eval
            state, _, done, info = env.step(action)
            actions.append(action)
            portfolio_history.append(info["portfolio"])

        final   = portfolio_history[-1]
        ret     = (final - env.initial_capital) / env.initial_capital
        rets    = np.diff(portfolio_history) / np.array(portfolio_history[:-1])
        sharpe  = rets.mean() / (rets.std() + 1e-8) * np.sqrt(252)
        cum     = np.array(portfolio_history) / env.initial_capital
        dd      = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)

        print(f"\n[PPO Agent] Evaluation Results")
        print(f"  Final Portfolio : ${final:,.0f}")
        print(f"  Total Return    : {ret*100:+.2f}%")
        print(f"  Sharpe Ratio    : {sharpe:.4f}")
        print(f"  Max Drawdown    : {dd.min()*100:.2f}%")
        print(f"  Total Trades    : {env.trades}")
        print(f"  Action dist     : Buy={actions.count(1)} "
              f"Hold={actions.count(0)} Sell={actions.count(2)}")

        return {"total_return": ret, "sharpe": sharpe,
                "max_drawdown": float(dd.min()), "trades": env.trades,
                "portfolio_history": portfolio_history}
