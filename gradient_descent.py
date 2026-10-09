"""
Gradient descent from scratch, with learning-rate plots.

Goal: find the straight line that best predicts house price from house area,
      price = w * area + b
without using any machine-learning library. The "learning" is written by hand.

Run:  python gradient_descent.py
Needs: numpy and matplotlib   (pip install numpy matplotlib)
It prints a results table and saves three pictures next to this file.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Make some data: 100 houses
#    area  in thousand square feet (0.5 = 500 sq ft)
#    price in lakh rupees
#    The "true" rule is price = 40 * area + 15, plus random noise.
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
area = rng.uniform(0.5, 2.5, 100)
price = 40 * area + 15 + rng.normal(0, 8, 100)


# ---------------------------------------------------------------
# 2. The three pieces of gradient descent
# ---------------------------------------------------------------
def loss(w, b):
    """How wrong is the line? Mean squared error:
    average of (predicted price - real price) squared. Lower is better."""
    error = w * area + b - price
    return np.mean(error ** 2)


def gradient(w, b):
    """Which way is uphill? Returns the slope of the loss for w and for b.
    Positive slope means: increasing this number makes the loss worse."""
    error = w * area + b - price
    dw = 2 * np.mean(error * area)
    db = 2 * np.mean(error)
    return dw, db


def gradient_descent(learning_rate, steps=200):
    """Start with a bad guess (w=0, b=0) and repeatedly take a small step downhill.
    Returns the history of w, b and loss at every step."""
    w, b = 0.0, 0.0
    history = [(w, b, loss(w, b))]
    for _ in range(steps):
        dw, db = gradient(w, b)
        w = w - learning_rate * dw      # step downhill = move AGAINST the slope
        b = b - learning_rate * db
        history.append((w, b, loss(w, b)))
    return np.array(history)


# ---------------------------------------------------------------
# 3. Try several learning rates (step sizes)
# ---------------------------------------------------------------
LEARNING_RATES = [0.001, 0.01, 0.1, 0.25, 0.3]
runs = {lr: gradient_descent(lr) for lr in LEARNING_RATES}

# The exact best line from the textbook formula, to check our answer against
exact_w, exact_b = np.polyfit(area, price, 1)
exact_loss = loss(exact_w, exact_b)

print(f"Exact best line (formula): price = {exact_w:.2f} * area + {exact_b:.2f}, loss = {exact_loss:.2f}\n")
print(f"{'Learning rate':>13} | {'w':>10} | {'b':>10} | {'Loss after 200 steps':>20}")
print("-" * 64)
for lr, h in runs.items():
    w, b, final_loss = h[-1]
    print(f"{lr:>13} | {w:>10.2f} | {b:>10.2f} | {final_loss:>20,.2f}")


# ---------------------------------------------------------------
# 4. Pictures
# ---------------------------------------------------------------
# Picture 1: loss at every step, one curve per learning rate
plt.figure(figsize=(8, 5))
for lr, h in runs.items():
    plt.plot(h[:, 2], label=f"learning rate {lr}", linewidth=2)
plt.axhline(exact_loss, color="grey", linestyle="--", linewidth=1, label="best possible loss")
plt.yscale("log")
plt.xlabel("Step number")
plt.ylabel("Loss (mean squared error, log scale)")
plt.title("How fast each learning rate learns")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("1_loss_curves.png", dpi=150)

# Picture 2: the data and the lines found
plt.figure(figsize=(8, 5))
plt.scatter(area, price, s=18, color="grey", alpha=0.6, label="houses")
xs = np.array([0.5, 2.5])
for lr in [0.001, 0.01, 0.1]:
    w, b, _ = runs[lr][-1]
    plt.plot(xs, w * xs + b, linewidth=2, label=f"line after 200 steps, learning rate {lr}")
plt.plot(xs, exact_w * xs + exact_b, "k--", linewidth=1.5, label="exact best line")
plt.xlabel("Area (thousand sq ft)")
plt.ylabel("Price (lakh rupees)")
plt.title("The line each learning rate ended up with")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("2_fitted_lines.png", dpi=150)

# Picture 3: the path taken across the "loss landscape"
W, B = np.meshgrid(np.linspace(-5, 75, 200), np.linspace(-25, 55, 200))
L = np.array([[loss(w, b) for w in W[0]] for b in B[:, 0]])
plt.figure(figsize=(8, 6))
plt.contourf(W, B, np.log10(L), levels=30, cmap="Blues_r")
plt.colorbar(label="log10 of loss (darker = lower)")
for lr, colour in [(0.01, "tab:green"), (0.1, "tab:orange"), (0.25, "tab:red")]:
    h = runs[lr]
    plt.plot(h[:, 0], h[:, 1], "-o", color=colour, markersize=3, linewidth=1.2, label=f"learning rate {lr}")
plt.plot(exact_w, exact_b, "k*", markersize=16, label="lowest point")
plt.xlim(-5, 75)
plt.ylim(-25, 55)
plt.xlabel("w (price per thousand sq ft)")
plt.ylabel("b (base price)")
plt.title("The path each learning rate walks towards the lowest point")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("3_path_on_landscape.png", dpi=150)

print("\nSaved: 1_loss_curves.png, 2_fitted_lines.png, 3_path_on_landscape.png")
plt.show()
