

#MLE PLOT (Question 1)

import numpy as np
import matplotlib.pyplot as plt


def simulation (trials, probHeads = 0.5):
  result = np.random.binomial(1,probHeads,trials)
  propHeads = np.mean(result)
  return propHeads

thetaMLE = 8/11

trials10 = simulation(10, probHeads = thetaMLE)
trials100 = simulation(100, probHeads = thetaMLE)

print(f"MLE for θ: {thetaMLE:.3f}")
print(f"Proportion of heads in 100 trials: {trials100:.3f}")
print(f"Proportion of heads in 10 trials: {trials10:.3f}")

#plotting
trials_x = [10,100]
prop_y = [trials10,trials100]

plt.plot(trials_x, prop_y, 'o-', label='Empirical Proportion of Heads')
plt.axhline(y=thetaMLE, color='r', label='MLE')
plt.xscale('log')
plt.xlabel('Number of Trials (log scale)')
plt.ylabel('Proportion of Heads')
plt.title('Comparison of MLE with Empirical Averages')
plt.legend()
plt.show()


#MAP PLOT (QUESTION 2)

import numpy as np
import matplotlib.pyplot as plt

def simulate_coin_tosses(trials, prob_heads=0.5):
    outcomes = np.random.binomial(1, prob_heads, trials)
    proportion_heads = np.mean(outcomes)
    return proportion_heads



trials10MAP = simulate_coin_tosses(10)
trials100MAP = simulate_coin_tosses(100)
trials1000MAP = simulate_coin_tosses(1000)

thetaMAP = 9/13

print(f"Proportion of heads in 10 trials: {trials10MAP:.3f}")
print(f"Proportion of heads in 100 trials: {trials100MAP:.3f}")
print(f"Proportion of heads in 1000 trials: {trials1000MAP:.3f}")


trials = [10,100,1000]
proportions = [trials10MAP, trials100MAP, trials1000MAP]
plt.axhline(y=thetaMAP, color='r', label='MAP Estimation')
plt.plot(trials, proportions, 'o-', label='Empirical Proportion of Heads')
plt.xscale('log')
plt.xlabel('Number of Trials (log scale)')
plt.ylabel('Proportion of Heads')
plt.title('Comparison of MAP with Empirical Averages')
plt.legend()
plt.show()

