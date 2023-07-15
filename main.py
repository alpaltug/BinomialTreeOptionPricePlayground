import math

# Parameters
S0 = 100      # spot stock price
K = 105       # strike
T = 1.0       # maturity 
r = 0.05      # risk free rate
v = 0.2       # volatility
N = 1000      # number of steps

# Precompute constants
dt = T/N                           # delta t
u = math.exp(v*math.sqrt(dt))       # up factor
d = 1/u                             # down factor
p = (math.exp(r*dt) - d) / (u - d)  # risk neutral probability

def binomial_tree(S0, K, T, r, v, N, option_type='call'):
    # Initialize the end nodes of the tree
    end_nodes = [S0 * u**j * d**(N - j) for j in range(N + 1)]
    
    # Calculate option prices
    if option_type == 'call':
        option_prices = [max(0, S - K) for S in end_nodes]
    else:
        option_prices = [max(0, K - S) for S in end_nodes]
    
    # Move to earlier times
    for i in range(N - 1, -1, -1):
        option_prices = [(p * option_prices[j + 1] + (1 - p) * option_prices[j]) * math.exp(-r * dt)
                         for j in range(i + 1)]
    return option_prices[0]

call_price = binomial_tree(S0, K, T, r, v, N, option_type='call')
put_price = binomial_tree(S0, K, T, r, v, N, option_type='put')

print("European call option price: ", round(call_price, 2))
print("European put option price: ", round(put_price, 2))
