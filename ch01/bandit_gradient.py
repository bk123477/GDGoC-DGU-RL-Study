import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, arms=10):  # arms = 슬롯머신 대수
        self.rates = np.random.rand(arms)  # 슬롯머신 각각의 승률 설정(무작위)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0

class Agent:
    def __init__(self, action_size=10, tau=1.0, alpha=0.1):
        self.Qs = np.zeros(action_size)  # 각 슬롯머신에 대한 추정 보상
        self.tau = tau  # 온도 파라미터
        self.alpha = alpha  # 학습률
        self.ns = np.zeros(action_size)  # 각 슬롯머신 선택 횟수

    # 소프트맥스 선택 확률 계산
    def get_action(self):
        exp_Qs = np.exp(self.Qs / self.tau)
        probabilities = exp_Qs / np.sum(exp_Qs)  # 각 슬롯머신의 선택 확률 계산
        action = np.random.choice(len(self.Qs), p=probabilities)  # 확률적으로 행동 선택
        return action

    # 슬롯머신의 가치 추정(업데이트)
    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += self.alpha * (reward - self.Qs[action])  # 그래디언트 업데이트

if __name__ == '__main__':
    steps = 1000

    bandit = Bandit()
    agent = Agent()
    total_reward = 0
    total_rewards = []  # 보상 합
    rates = []          # 승률

    for step in range(steps):
        action = agent.get_action()   # 행동 선택 (소프트맥스 기반)
        reward = bandit.play(action)  # 실제로 플레이하고 보상을 받음
        agent.update(action, reward)  # 행동과 보상을 통해 학습
        total_reward += reward

        total_rewards.append(total_reward)       # 현재까지의 보상 합 저장
        rates.append(total_reward / (step + 1))  # 현재까지의 승률 저장

    print(total_reward)

    # [그림 1-12] 단계별 보상 총합
    plt.ylabel('Total reward')
    plt.xlabel('Steps')
    plt.plot(total_rewards)
    plt.show()

    # [그림 1-13] 단계별 승률
    plt.ylabel('Rates')
    plt.xlabel('Steps')
    plt.plot(rates)
    plt.show()
