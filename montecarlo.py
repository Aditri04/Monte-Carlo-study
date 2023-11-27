import random
import numpy as np
import math


def sampleGen(no_of_samples, distribution, parameters, seed=20):
    random.seed(seed)
    samples = []
    match distribution:
        case 'arb-discrete':
            p = []
            for param in parameters:
                p.append(float(param))
            F = []
            for i in range(len(p)):
                F.append(sum(p[0:i+1]))
            if F[-1] != 1:
                return 'Invalid paramters, Probabilites do not add up to 1'
            for i in range(no_of_samples):
                sample = 0
                u = random.random()
                while F[sample] <= u:
                    sample = sample + 1
                samples.append(sample)
            return samples
        case 'bernoulli':
            if len(parameters) != 1:
                return 'Invalid number of parameters'
            p = parameters[0]
            for i in range(no_of_samples):
                u = random.random()
                if u < p:
                    samples.append(1)
                else:
                    samples.append(0)
            return samples
        case 'binomial':
            if len(parameters) != 2:
                return 'Invalid number of parameters'
            n = parameters[0]
            p = parameters[1]
            for i in range(no_of_samples):
                sample = 0
                for j in range(n):
                    u = random.random()
                    if u < p:
                        sample = sample + 1
                samples.append(sample)
            return samples
        case 'geometric':
            if len(parameters) != 1:
                return 'Invalid number of parameters'
            p = parameters[0]
            for i in range(no_of_samples):
                sample = 0
                u = random.random()
                while u > p:
                    sample = sample + 1
                    u = random.random()
                samples.append(sample)
            return samples
        case 'neg-binomial':
            if len(parameters) != 2:
                return 'Invalid number of parameters'
            k = parameters[0]
            p = parameters[1]
            for i in range(no_of_samples):
                sample = 0
                success = 0
                while success < k:
                    u = random.random()
                    if u > p:
                        success = success+1
                    sample = sample + 1
                samples.append(sample)
            return samples
        case 'poisson':
            if len(parameters) != 1:
                return 'Invalid number of parameters'
            lamdba_dist = parameters[0]
            for i in range(no_of_samples):
                temp = 1
                val = math.exp(-1 * lamdba_dist)
                sample = 0
                while temp >= val:
                    u = random.random()
                    temp = temp*u
                    sample = sample+1
                samples.append(sample-1)
            return samples
        case 'uniform':
            if len(parameters) != 2:
                return 'Invalid number of parameters'
            a = parameters[0]
            b = parameters[1]
            for i in range(no_of_samples):
                u = random.random()
                samples.append(a + (b-a)*u)
            return samples
        case 'exponential':
            if len(parameters) != 1:
                return 'Invalid number of parameters'
            lambda_dist = parameters[0]
            for i in range(no_of_samples):
                u = random.random()
                sample = -1 * (np.log(u))/lambda_dist
                samples.append(sample)
            return samples
        case 'gamma':
            if len(parameters) != 2:
                return 'Invalid number of parameters'
            alpha = parameters[0]
            lambda_dist = parameters[1]
            for i in range(no_of_samples):
                u = random.random()
                sample = (-1 / lambda_dist) * math.log(1 - u) ** (1 / alpha)
                samples.append(sample)
            return samples
        case 'normal':
            if len(parameters) != 2:
                return 'Invalid number of parameters'
            mean = parameters[0]
            std = parameters[1]
            for i in range(no_of_samples):
                u1 = random.random()
                u2 = random.random()
                z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
                sample = mean + std * z1
                samples.append(sample)
            return samples
        case _:
            return 'Invalid distribution'


def poisson_dist_comparison(lambda1, lambda2, no_of_samples):
    seed1 = 10
    sample1 = sampleGen(no_of_samples, 'poisson', [lambda1], seed1)
    seed2 = 50
    sample2 = sampleGen(no_of_samples, 'poisson', [lambda2], seed2)
    success = np.sum(np.array(sample1) > np.array(sample2))
    prob = success/no_of_samples
    return prob


def prob_estimate_royal_flush(no_of_samples):
    success = 0
    prob = 0
    seed = 5
    for i in range(no_of_samples):
        valid_hand = False
        while valid_hand == False:
            seed = seed + 10
            cards = sampleGen(5, 'uniform', [0, 1], seed)
            hand = np.floor(np.array(cards)*52)
            if np.unique(hand).size == 5:
                valid_hand = True
                isFlush = False
                sorted_hand = np.sort(hand)
                isFlush = np.array_equal(
                    sorted_hand, np.array([0, 1, 2, 3, 4])) or np.array_equal(sorted_hand, np.array(
                        [5, 6, 7, 8, 9])) or np.array_equal(sorted_hand, np.array([10, 11, 12, 13, 14])) or np.array_equal(sorted_hand, np.array([15, 16, 17, 18, 19]))
                if isFlush == True:
                    success = success + 1
    prob = success/no_of_samples
    return prob


def prob_estimate_oil_filter(p, time_to_change, no_of_samples):
    success = 0
    seed = 20
    for i in range(no_of_samples):
        seed = seed + 10
        u1 = sampleGen(1, 'uniform', [0, 1], seed)[0]
        if u1 < p:
            seed = seed + 10
            sample = sampleGen(1, 'exponential', [5], seed)[0]
            if sample > time_to_change:
                success = success + 1
        else:
            seed = seed + 10
            sample = sampleGen(1, 'exponential', [20], seed)[0]
            if sample > time_to_change:
                success = success + 1
    prob = success/no_of_samples
    return prob


"""
print(sampleGen(10, 'bernoulli', [0.4], 20))
print(sampleGen(10, 'binomial', [5, 0.4], 20))
print(sampleGen(10, 'geometric', [0.4], 20))
print(sampleGen(10, 'neg-binomial', [5, 0.3], 20))
print(sampleGen(10, 'uniform', [5, 9], 20))
print(sampleGen(10, 'exponential', [5], 20))
print(sampleGen(10, 'poisson', [5], 20))
print(sampleGen(10, 'normal', [5, 0.5], 20))
print(sampleGen(10, 'gamma', [0.25, 5], 20))
print(sampleGen(10, 'arb-discrete', [0.25, 0.25, 0.5], 20))
"""

# from 5.7(d) this guarantees the error not exceeding 0.005 with probability 0.95
n_samples = 38416

print('Estimated Probability P{X>Y} is ',
      poisson_dist_comparison(3, 5, n_samples))
print('Estimated Probability of a royal flush is ',
      prob_estimate_royal_flush(n_samples))
print('Estimated Probability that it will take more than 35 mins is ',
      prob_estimate_oil_filter(1/5, 35/60, n_samples))
