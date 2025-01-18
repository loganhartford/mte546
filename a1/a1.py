import numpy as np
from scipy.stats import norm

apple_roughness = np.array([
    2.56, 3.01, 3.22, 3.44, 3.62, 3.03, 2.40, 2.70, 2.58, 3.94,
    2.76, 3.30, 2.92, 3.36, 2.69, 2.44, 2.43, 3.20, 2.93, 2.92
])
lemon_roughness = np.array([
    4.57, 4.12, 4.08, 4.64, 3.68, 4.28, 4.33, 3.90, 4.09, 3.53,
    3.54, 4.04, 4.29, 5.03, 3.73, 4.07, 3.97, 3.23, 3.82, 3.28
])

apple_mean = np.mean(apple_roughness)
apple_std = np.std(apple_roughness, ddof=1)   # ddof=1 for unbiased sample std
lemon_mean = np.mean(lemon_roughness)
lemon_std = np.std(lemon_roughness, ddof=1)

# 1. Liklihood
p_x_given_apple = lambda x: norm.pdf(x, apple_mean, apple_std)
p_x_given_lemon = lambda x: norm.pdf(x, lemon_mean, lemon_std)

p_apple = 1000 / (2000 + 1000)  # 1/3
p_lemon = 2000 / (2000 + 1000)  # 2/3

x_test = 3.4

likelihood_apple = p_x_given_apple(x_test)
likelihood_lemon = p_x_given_lemon(x_test)

classification_decision = "Apple" if likelihood_apple > likelihood_lemon else "Lemon"

print("1(a) Likelihood of x=3.4 given Apple:", round(likelihood_apple, 2))
print("1(b) Likelihood of x=3.4 given Lemon:", round(likelihood_lemon, 2))
print("1(c) Max Likelihood Classifier Decision:", classification_decision)

# 2. Bayesian Classifier
p_x = likelihood_apple * p_apple + likelihood_lemon * p_lemon
p_apple_given_x = (likelihood_apple * p_apple) / p_x
p_lemon_given_x = (likelihood_lemon * p_lemon) / p_x

bayesian_decision = "Apple" if p_apple_given_x > p_lemon_given_x else "Lemon"

print("2(a) Bayesian Probability of Apple given x=3.4:", round(p_apple_given_x, 2))
print("2(b) Bayesian Probability of Lemon given x=3.4:", round(p_lemon_given_x, 2))
print("2(c) Bayesian Classifier Decision:", bayesian_decision)

# 3. Color Information
# Now we also measure color = "yellow".
# The assignment says:
#   P(Lemon|yellow) = 0.75,  P(Apple|yellow) = 0.25
# but these are posterior probabilities if color were the *only* measurement.
#
# To fuse with roughness, we need p(yellow|Apple) and p(yellow|Lemon),
# which must satisfy the ratio that yields 25%-75% posterior if color alone is known.
# One valid choice is:
p_yellow_given_apple = 0.20
p_yellow_given_lemon = 0.30
# (Any pair with ratio = 2/3 will reproduce the 25%-75% posterior for color alone.)

# Because roughness and color are conditionally independent given the fruit,
# the joint likelihood p(x=3.4, yellow | fruit) = p(x=3.4 | fruit)*p(yellow | fruit).
# So we do one more Bayesian update:
p_x_and_yellow = (p_apple_given_x * p_yellow_given_apple
                  + p_lemon_given_x * p_yellow_given_lemon)

p_apple_final = (p_apple_given_x * p_yellow_given_apple) / p_x_and_yellow
p_lemon_final = (p_lemon_given_x * p_yellow_given_lemon) / p_x_and_yellow
final_decision = "Apple" if p_apple_final > p_lemon_final else "Lemon"

print("3(a) Final Probability of Apple given x=3.4 and yellow color:",
      round(p_apple_final, 2))
print("3(b) Final Probability of Lemon given x=3.4 and yellow color:",
      round(p_lemon_final, 2))
print("Final Classification with Color Information:", final_decision)
