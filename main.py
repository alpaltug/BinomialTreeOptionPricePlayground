import math
import math
from scipy.stats import norm

# Parameters
S0 = 100      # spot stock price
K = 105       # strike
T = 1.0       # maturity 
r = 0.05      # risk free rate
v = 0.2       # volatility
N = 1000      # number of steps

option_type = 'call'

time_to_maturity = T


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

def calculate_delta(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity, option_type):
    d1 = calculate_d1(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity)
    if option_type == 'call':
        delta = math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d1)
    elif option_type == 'put':
        delta = math.exp(-risk_free_rate * time_to_maturity) * (norm.cdf(d1) - 1)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    return delta

def calculate_gamma(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity):
    d1 = calculate_d1(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity)
    gamma = (math.exp(-risk_free_rate * time_to_maturity) * norm.pdf(d1)) / (stock_price * volatility * math.sqrt(time_to_maturity))
    return gamma

def calculate_vega(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity):
    d1 = calculate_d1(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity)
    vega = stock_price * math.exp(-risk_free_rate * time_to_maturity) * norm.pdf(d1) * math.sqrt(time_to_maturity)
    return vega

def calculate_rho(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity, option_type):
    d2 = calculate_d2(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity)
    if option_type == 'call':
        rho = strike_price * time_to_maturity * math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
    elif option_type == 'put':
        rho = -strike_price * time_to_maturity * math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    return rho

def calculate_implied_volatility(stock_price, strike_price, risk_free_rate, option_price, time_to_maturity, option_type):
    implied_volatility = 0.5  # Initial guess for implied volatility
    MAX_ITERATIONS = 100
    PRECISION = 1e-5

    for i in range(MAX_ITERATIONS):
        d1 = calculate_d1(stock_price, strike_price, risk_free_rate, implied_volatility, time_to_maturity)
        if option_type == 'call':
            option_price_calculated = stock_price * norm.cdf(d1) - strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d1 - implied_volatility * math.sqrt(time_to_maturity))
        elif option_type == 'put':
            option_price_calculated = strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d1 + implied_volatility * math.sqrt(time_to_maturity)) - stock_price * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")
        
        diff = option_price_calculated - option_price

        if abs(diff) < PRECISION:
            break

        vega = calculate_vega(stock_price, strike_price, risk_free_rate, implied_volatility, time_to_maturity)
        implied_volatility -= diff / vega

    return implied_volatility

def calculate_d1(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity):
    d1 = (math.log(stock_price / strike_price) + (risk_free_rate + (volatility ** 2) / 2) * time_to_maturity) / (volatility * math.sqrt(time_to_maturity))
    return d1

def calculate_d2(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity):
    d2 = calculate_d1(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity) - volatility * math.sqrt(time_to_maturity)
    return d2


call_price = binomial_tree(S0, K, T, r, v, N, option_type='call')
put_price = binomial_tree(S0, K, T, r, v, N, option_type='put')

d1 = calculate_d1(S0, K, r, v, time_to_maturity)
d2 = calculate_d2(S0, K, r, v, time_to_maturity)

if option_type == 'call':
    option_price = S0 * norm.cdf(d1) - K * math.exp(-r * time_to_maturity) * norm.cdf(d2)
elif option_type == 'put':
    option_price = K * math.exp(-r * time_to_maturity) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
else:
    raise ValueError("Invalid option type. Choose 'call' or 'put'.")

# Calculate option greeks
delta = calculate_delta(S0, K, r, v, time_to_maturity, option_type)
gamma = calculate_gamma(S0, K, r, v, time_to_maturity)
vega = calculate_vega(S0, K, r, v, time_to_maturity)
rho = calculate_rho(S0, K, r, v, time_to_maturity, option_type)
implied_volatility = calculate_implied_volatility(S0, K, r, option_price, time_to_maturity, option_type)

print("European call option price: ", round(call_price, 2))
print("European put option price: ", round(put_price, 2))

print("European", option_type, "option price:", option_price)
print("European", option_type, "delta value:", delta)
print("European option gamma value:", gamma)
print("European option vega value:", vega)
print("European", option_type, "option rho value:", rho)
print("European", option_type, "option implied volatility value:", implied_volatility)