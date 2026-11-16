# Two Counterexamples for e-means

Run with Python 3:

    python -m pip install -r requirements.txt
    python counterexamples.py

The script checks:

- Revival: losses 20 versus 110, 40 versus 420, and 100 versus 2550
  from the prescribed seeds.
- All three possible k-means++ seed-value pairs and their exact probabilities.
- Agreement with Lloyd when revival is disabled in the revival family.
- Forgetting: losses 150 versus 490/3 and the displayed center trajectory.
- Appendix A's interval bounds and the exact conditional failure
  probability 24/49.

The reproduction is an independent one-dimensional implementation of the
printed recipe for two clusters. The illustrative forgetting trajectory uses
the prescribed first two epoch permutations; subsequent epochs use seed 0.
The probability calculation is exact, conditional on initial centers (15, 29).

The certificate uses 45-digit outward-rounded interval arithmetic and treats
the printed decimal coefficients as exact.

Original e-means paper:
https://arxiv.org/abs/2611.09501
