import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100      # spot stock price
K = 105       # strike
T = 1.0       # maturity 
r = 0.05      # risk free rate
v = 0.2       # volatility
N = 1000      # number of steps

implied_volatilities = []
atm_implied_volatility = 0.20
increment_per_step = 0.001
for i in range(N + 1):
    strike_distance = i - N / 2  # Distance from the ATM strike
    implied_vol = atm_implied_volatility + strike_distance * increment_per_step
    implied_volatilities.append(implied_vol)

option_type = 'call'

# Precompute constants
dt = T/N                           # delta t
u = math.exp(v*math.sqrt(dt))       # up factor
d = 1/u                             # down factor
p = (math.exp(r*dt) - d) / (u - d)  # risk neutral probability

time_to_maturity_range = np.linspace(0.1, 1, 10)
volatility_range = np.linspace(0.1, 0.5, 10)


def binomial_tree(S0, K, T, r, implied_volatilities, N, option_type='call', dividend=0.0, div_dates=[], div_amounts=[]):
    dt = T / N
    p_values = []  # To store the risk-neutral probabilities

    # Calculate the up and down factors for each time step based on the implied volatilities
    up_factors = [math.exp(implied_vol * math.sqrt(dt)) for implied_vol in implied_volatilities]
    down_factors = [1 / up_factor for up_factor in up_factors]

    # Calculate the risk-neutral probabilities for each time step using the up and down factors
    for i in range(N):
        p_values.append((math.exp(r * dt) - down_factors[i]) / (up_factors[i] - down_factors[i]))

    # Calculate option prices at the end nodes of the binomial tree
    end_nodes = [S0 * up_factors[N - j] * down_factors[j] for j in range(N + 1)]
    if option_type == 'call':
        option_prices = [max(0, S - K) for S in end_nodes]
    else:
        option_prices = [max(0, K - S) for S in end_nodes]

    # Calculate adjusted option prices for dividend-paying assets
    if dividend > 0 and len(div_dates) == len(div_amounts) > 0:
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                div_adjustment = 1.0
                for div_date, div_amount in zip(div_dates, div_amounts):
                    if div_date * T <= j * dt:
                        div_adjustment *= (1 - div_amount / (end_nodes[j] * (1 - dividend)))  # Adjusted stock price
                option_prices[j] = (p_values[i] * option_prices[j + 1] + (1 - p_values[i]) * option_prices[j]) * math.exp(-r * dt) * div_adjustment

    else:
        # Calculate option prices at earlier nodes using backward induction
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                option_prices[j] = (p_values[i] * option_prices[j + 1] + (1 - p_values[i]) * option_prices[j]) * math.exp(-r * dt)

    return option_prices[0]


def calculate_d1(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity):
    d1 = (math.log(stock_price / strike_price) + (risk_free_rate + (volatility ** 2) / 2) * time_to_maturity) / (volatility * math.sqrt(time_to_maturity))
    return d1

def calculate_d2(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity):
    d1 = calculate_d1(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity)
    d2 = d1 - volatility * math.sqrt(time_to_maturity)
    return d2

def calculate_delta(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity, option_type):
    d1 = calculate_d1(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity)
    if option_type == 'call':
        delta = norm.cdf(d1)
    elif option_type == 'put':
        delta = -norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    return delta

def calculate_theta(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity, option_price, option_type):
    d1 = calculate_d1(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity)
    d2 = calculate_d2(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity)
    if option_type == 'call':
        theta = -(stock_price * norm.pdf(d1) * volatility / (2 * math.sqrt(time_to_maturity))) - (risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2))
    elif option_type == 'put':
        theta = -(stock_price * norm.pdf(d1) * volatility / (2 * math.sqrt(time_to_maturity))) + (risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2))
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    return theta

def calculate_gamma(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity):
    d1 = calculate_d1(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity)
    gamma = norm.pdf(d1) / (stock_price * volatility * math.sqrt(time_to_maturity))
    return gamma

def calculate_vega(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity):
    d1 = calculate_d1(stock_price, strike_price, risk_free_rate, volatility, time_to_maturity)
    vega = stock_price * norm.pdf(d1) * math.sqrt(time_to_maturity)
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
    implied_volatility = 0.5  
    MAX_ITERATIONS = 100
    PRECISION = 1e-5

    for i in range(MAX_ITERATIONS):
        d1 = calculate_d1(stock_price, strike_price, risk_free_rate, implied_volatility, time_to_maturity)
        d2 = calculate_d2(stock_price, strike_price, risk_free_rate, implied_volatility, time_to_maturity)
        if option_type == 'call':
            option_price_calculated = stock_price * norm.cdf(d1) - strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
        elif option_type == 'put':
            option_price_calculated = strike_price * math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")

        option_price_difference = option_price_calculated - option_price  
        if abs(option_price_difference) < PRECISION:
            return implied_volatility
        vega = calculate_vega(stock_price, strike_price, risk_free_rate, implied_volatility, time_to_maturity)
        implied_volatility -= option_price_difference / vega  
    return implied_volatility

def plot_delta_vs_theta(stock_price, strike_price, risk_free_rate, volatility_range, time_to_maturity, option_type):
    deltas = []
    thetas = []
    N = 1000
    for v in volatility_range:
        option_price = binomial_tree(stock_price, strike_price, time_to_maturity, risk_free_rate, implied_volatilities, v, N, option_type)
        delta = calculate_delta(stock_price, strike_price, risk_free_rate, v, time_to_maturity, option_type)
        theta = calculate_theta(stock_price, strike_price, risk_free_rate, v, time_to_maturity, option_price, option_type)
        deltas.append(delta)
        thetas.append(theta)
    plt.plot(deltas, thetas)
    plt.xlabel('Delta')
    plt.ylabel('Theta')
    plt.title('Delta vs Theta')
    plt.show()

def plot_theta_vs_vega(stock_price, strike_price, risk_free_rate, volatility_range, time_to_maturity, option_type):
    thetas = []
    vegas = []
    N = 1000
    for v in volatility_range:
        option_price = binomial_tree(stock_price, strike_price, time_to_maturity, risk_free_rate, implied_volatilities, v, N, option_type)
        theta = calculate_theta(stock_price, strike_price, risk_free_rate, v, time_to_maturity, option_price, option_type)
        vega = calculate_vega(stock_price, strike_price, risk_free_rate, v, time_to_maturity)
        thetas.append(theta)
        vegas.append(vega)
    plt.plot(thetas, vegas)
    plt.xlabel('Theta')
    plt.ylabel('Vega')
    plt.title('Theta vs Vega')
    plt.show()

def main():
    call_price = binomial_tree(S0=100, K=105, T=1, r=0.05, implied_volatilities=implied_volatilities, N=100, option_type='call', dividend=0.03, div_dates=[0.5], div_amounts=[2.0])
    put_price = binomial_tree(S0=100, K=105, T=1, r=0.05, implied_volatilities=implied_volatilities, N=100, option_type='put', dividend=0.03, div_dates=[0.5], div_amounts=[2.0])

    d1 = calculate_d1(S0, K, r, v, T)
    d2 = calculate_d2(S0, K, r, v, T)

    if option_type == 'call':
        option_price = S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

    # Calculate option greeks
    delta = calculate_delta(S0, K, r, v, T, option_type)
    gamma = calculate_gamma(S0, K, r, v, T)
    vega = calculate_vega(S0, K, r, v, T)
    rho = calculate_rho(S0, K, r, v, T, option_type)
    implied_volatility = calculate_implied_volatility(S0, K, r, option_price, T, option_type)

    option_price = binomial_tree(S0=100, K=105, T=1, r=0.05, implied_volatilities=implied_volatilities, N=100, option_type='call', dividend=0.03, div_dates=[0.5], div_amounts=[2.0])
    print("European call with dividents: ", option_price)

    print("European call option price: ", round(call_price, 2))
    print("European put option price: ", round(put_price, 2))

    print("European", option_type, "option price:", option_price)
    print("European", option_type, "delta value:", delta)
    print("European option gamma value:", gamma)
    print("European option vega value:", vega)
    print("European", option_type, "option rho value:", rho)
    print("European", option_type, "option implied volatility value:", implied_volatility)

    #plot_delta_vs_theta(S0, K, r, volatility_range, T, option_type)
    #plot_theta_vs_vega(S0, K, r, volatility_range, T, option_type)

if __name__ == "__main__":
    main()
