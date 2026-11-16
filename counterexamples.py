"""Reproduce Two Counterexamples for e-means, using the printed recipe."""

from collections import Counter
from fractions import Fraction
from itertools import combinations
from math import cos, isclose, log2, pi
from random import Random

X = [0, 0, 15, 29, 29, 29, 29, 29]


def parameters(n, T):
    """Printed batch and peak-step laws for k=2."""
    raw = 0.035 * 2**0.13 * n**0.88 * T**0.81
    B = min(n, 2 ** max(0, round(log2(raw))))
    alpha0 = min(1, 0.824 * 2**-0.46 * B**0.90 * n**-0.45 * T**-0.11)
    return B, alpha0


def stats(x, centers):
    counts, sums = [0, 0], [0, 0]
    for v in x:
        j = 0 if (v - centers[0])**2 <= (v - centers[1])**2 else 1
        counts[j] += 1
        sums[j] += v
    return counts, sums


def loss(x, centers):
    return sum(min((v - c)**2 for c in centers) for v in x)


def lloyd(x, centers):
    centers = list(centers)
    for _ in range(100):
        counts, sums = stats(x, centers)
        assert all(counts)
        updated = [sums[j] / counts[j] for j in range(2)]
        if updated == centers:
            return updated
        centers = updated
    raise RuntimeError("Lloyd did not converge")


def emeans(x, centers, T, B, alpha0, revival=True, prefix=()):
    """One-dimensional recipe with two centers and last-epoch means.

    prefix optionally prescribes the first few epoch permutations.
    """
    assert len(x) % B == 0
    N = [1e-10, 1e-10]
    S = [1e-10 * c for c in centers]
    M = T * (len(x) // B)
    rng = Random(0)
    history, revivals = [], []

    for epoch in range(T):
        order = list(prefix[epoch]) if epoch < len(prefix) else list(x)
        if epoch >= len(prefix):
            rng.shuffle(order)

        final_counts, final_sums = [0, 0], [0, 0]
        for start in range(0, len(x), B):
            centers = [S[j] / N[j] for j in range(2)]
            counts, sums = stats(order[start:start + B], centers)
            t = len(history)  # zero-based schedule index
            a = alpha0 * (0.55 + 0.45 * cos(pi * t / M))

            for j in range(2):
                N[j] = (1 - a) * N[j] + a * counts[j]
                S[j] = (1 - a) * S[j] + a * sums[j]
                final_counts[j] += counts[j]
                final_sums[j] += sums[j]

            if revival and (t + 1) % 5 == 0 and min(N) < 0.05 * sum(N):
                dead = 0 if N[0] < N[1] else 1
                donor = 1 - dead
                center, half = S[donor] / N[donor], N[donor] / 2
                delta = 1 / 1024
                N[donor] = N[dead] = half
                S[donor] = half * center * (1 - delta)
                S[dead] = half * center * (1 + delta)
                revivals.append(t + 1)

            history.append([S[j] / N[j] for j in range(2)])

    assert all(final_counts)
    final = [final_sums[j] / final_counts[j] for j in range(2)]
    return final, history, revivals


def kpp_probabilities(x):
    """Exact probabilities of unordered seed-value pairs under k-means++."""
    mass = Counter(x)
    probabilities = Counter()
    for u, nu in mass.items():
        denominator = sum(nv * (v - u)**2 for v, nv in mass.items())
        for v, nv in mass.items():
            if u != v:
                pair = tuple(sorted((u, v)))
                probabilities[pair] += (
                    Fraction(nu, len(x))
                    * Fraction(nv * (v - u)**2, denominator)
                )
    assert sum(probabilities.values()) == 1
    return probabilities


def revival_example():
    print("Revival: prescribed-seed losses and k-means++ expectations")
    for m in (10, 20, 50):
        D = m + 4
        x = [1] * m + [3] * m + [D]
        T = 10 * len(x)
        B, alpha0 = parameters(len(x), T)
        assert (B, alpha0) == (len(x), 1)

        probabilities = kpp_probabilities(x)
        expected_l = expected_e = Fraction(0)

        for seeds, probability in probabilities.items():
            c_l = lloyd(x, seeds)
            c_e, _, events = emeans(x, seeds, T, B, alpha0)
            c_off, _, _ = emeans(x, seeds, T, B, alpha0, revival=False)

            assert c_e == [1, 4]
            assert c_l == ([1, 4] if seeds == (1, 3) else [2, D])
            assert c_off == c_l
            assert events == ([5] if D in seeds else [])

            expected_l += probability * round(loss(x, c_l))
            expected_e += probability * round(loss(x, c_e))

        q = Fraction(4 * m*m, 2*m + 1) * (
            Fraction(1, (m + 3)**2 + 4*m)
            + Fraction(1, (m + 1)**2 + 4*m)
        )
        assert probabilities[(1, 3)] == q < Fraction(4, m)
        assert expected_l == 2*m + q * m * (m - 1)
        assert expected_l < 6*m
        assert expected_e == m * (m + 1)

        print(
            f"m={m}: Lloyd={2*m}, e-means={m*(m+1)}; "
            f"E[Lloyd]={float(expected_l):.10f}, E[e-means]={expected_e}"
        )


def forgetting_example():
    B, alpha0 = parameters(len(X), 25)
    assert B == 4
    prefix = [
        [15, 29, 29, 29, 0, 0, 29, 29],
        [0, 0, 29, 29, 15, 29, 29, 29],
    ]
    c_e, history, events = emeans(X, (15, 29), 25, B, alpha0, prefix=prefix)
    c_l = lloyd(X, (15, 29))

    assert history[1][0] > 1 > history[2][0]
    assert c_e == [0, 80 / 3]
    assert c_l == [5, 29]
    assert not events
    assert isclose(loss(X, c_e), 490 / 3)
    assert loss(X, c_l) == 150
    assert lloyd(X, c_e) == c_e

    print(f"\nForgetting: alpha0={alpha0:.12f}")
    print("Left centers:", [round(c[0], 6) for c in history[:3]])
    print("Lloyd:", c_l, "loss:", loss(X, c_l))
    print("e-means:", c_e, "loss:", loss(X, c_e))


def certificate():
    """Appendix A: outward-rounded intervals and exact event counts."""
    from mpmath import iv

    iv.dps = 45
    I = iv.mpf
    eta = I("1e-10")
    peak = (
        I("0.824") * I(2)**I("-0.46") * I(4)**I("0.90")
        * I(8)**I("-0.45") * I(25)**I("-0.11")
    )
    steps = [
        peak * (I("0.55") + I("0.45") * iv.cos(iv.pi * t / 50))
        for t in range(50)
    ]

    p, weights = eta, []
    for t, a in enumerate(steps, 1):
        p *= 1 - a
        weights = [(1 - a) * w for w in weights] + [a]
        low, high = I(0), I(0)

        for r in range(0, t, 2):
            u = weights[r]
            v = weights[r + 1] if r + 1 < t else I(0)
            # Require interval-separated weights before selecting min/max.
            assert u < v or v < u
            low += min(u, v)
            high += max(u, v)

        if t >= 4:
            assert 15 * (p + low) / (p + low + 2 * high) > I("1.0275")
        if t % 5 == 0:
            assert (
                (p + 2 * low) / (2 * p + 4 * sum(weights))
                > I("0.06410")
            )

    # Compress the 70 indexed first batches into (number of 15s, zeros).
    types = Counter(
        (sum(X[i] == 15 for i in batch), sum(X[i] == 0 for i in batch))
        for batch in combinations(range(8), 4)
    )
    counts = Counter()

    for (u, z), frequency in types.items():
        for (v, y), next_frequency in types.items():
            N, S, hit = eta, 15 * eta, 0
            batches = [(u, z), (1 - u, 2 - z), (v, y)]
            for t, (ones, zeros) in enumerate(batches, 1):
                a = steps[t - 1]
                N = (1 - a) * N + a * (ones + zeros)
                S = (1 - a) * S + a * 15 * ones
                center = S / N
                if center < 1:
                    hit = t
                    break
                assert center > 1

            predicted = (
                1 if u == 0 and z > 0 else
                3 if u == 1 and z <= 1 and v == 0 and y == 2 else 0
            )
            assert hit == predicted
            counts[hit] += frequency * next_frequency

    assert counts == Counter({1: 2100, 3: 300, 0: 2500})
    probability = Fraction(counts[1] + counts[3], sum(counts.values()))
    assert probability == Fraction(24, 49)
    print("\nInterval bounds certified at 45 decimal digits.")
    print(f"First crossing: batch 1={counts[1]}, batch 3={counts[3]}, none={counts[0]}")
    print("Conditional failure probability:", probability)


if __name__ == "__main__":
    revival_example()
    forgetting_example()
    certificate()
