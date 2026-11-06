
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# OGS 2D Fine Tracking: Horizontal and Vertical Axes
# Simplified computational model
# ============================================================

DT = 0.01
DURATION = 20.0
TAU = 0.08
MAX_COMMAND = 3.0

KP = 1.6
KI = 0.2
KD = 0.005

RUNS = 30

time = np.arange(0, DURATION, DT)
N = len(time)
START = int(1.0 / DT)


def pid_axis(disturbance, noise, dt=DT):
    """Simulate one axis with a PID controller and actuator."""
    actuator = np.zeros(len(disturbance))
    command = np.zeros(len(disturbance))
    residual = np.zeros(len(disturbance))

    integral = 0.0
    previous_error = 0.0

    for k in range(len(disturbance) - 1):
        residual[k] = disturbance[k] - actuator[k]
        measured = residual[k] + noise[k]

        integral += measured * dt
        derivative = (measured - previous_error) / dt

        raw_command = (
            KP * measured
            + KI * integral
            + KD * derivative
        )

        command[k] = np.clip(
            raw_command, -MAX_COMMAND, MAX_COMMAND
        )

        actuator[k + 1] = actuator[k] + (
            dt / TAU
        ) * (command[k] - actuator[k])

        previous_error = measured

    residual[-1] = disturbance[-1] - actuator[-1]
    command[-1] = command[-2]

    return residual, actuator, command


def run_simulation(seed, return_series=False):
    """Simulate both axes using the same PID gains."""
    rng = np.random.default_rng(seed)

    # Illustrative disturbances, not measured telescope data.
    dx = (
        0.8 * np.sin(2 * np.pi * 0.15 * time)
        + 1.0 * np.sin(2 * np.pi * 1.2 * time)
        + 0.20 * rng.normal(size=N)
    )

    dy = (
        0.6 * np.sin(2 * np.pi * 0.20 * time + 0.7)
        + 0.8 * np.sin(2 * np.pi * 1.7 * time)
        + 0.20 * rng.normal(size=N)
    )

    noise_x = 0.10 * rng.normal(size=N)
    noise_y = 0.10 * rng.normal(size=N)

    ex, ax, ux = pid_axis(dx, noise_x)
    ey, ay, uy = pid_axis(dy, noise_y)

    # Radial errors for open-loop and closed-loop systems
    open_radial = np.sqrt(dx**2 + dy**2)
    closed_radial = np.sqrt(ex**2 + ey**2)

    open_rms = np.sqrt(np.mean(open_radial[START:]**2))
    closed_rms = np.sqrt(np.mean(closed_radial[START:]**2))

    reduction = 100 * (1 - closed_rms / open_rms)

    saturation_x = np.mean(
        np.abs(ux[START:]) >= MAX_COMMAND
    )
    saturation_y = np.mean(
        np.abs(uy[START:]) >= MAX_COMMAND
    )

    result = (open_rms, closed_rms, reduction,
              saturation_x, saturation_y)

    if return_series:
        return result, (dx, dy, ex, ey, ax, ay, ux, uy)

    return result


# Run 30 independent simulations
results = np.array([
    run_simulation(seed=2000 + i)
    for i in range(RUNS)
])

print("\nOGS 2D Fine Tracking Results")
print("----------------------------")
print(f"Independent runs: {RUNS}")
print(f"Kp = {KP}, Ki = {KI}, Kd = {KD}")

for col, label in [
    (0, "Open-loop radial RMS"),
    (1, "Closed-loop radial RMS"),
    (2, "RMS reduction (%)"),
]:
    print(
        f"{label}: {np.mean(results[:, col]):.4f} "
        f"+/- {np.std(results[:, col], ddof=1):.4f}"
    )

print(
    "Mean X-axis saturation: "
    f"{100 * np.mean(results[:, 3]):.3f}%"
)
print(
    "Mean Y-axis saturation: "
    f"{100 * np.mean(results[:, 4]):.3f}%"
)

# Show one representative run
_, series = run_simulation(seed=2000, return_series=True)
dx, dy, ex, ey, ax, ay, ux, uy = series

plt.figure(figsize=(10, 5))
plt.plot(time, dx, label="Open-loop X disturbance")
plt.plot(time, ex, label="Closed-loop X error")
plt.plot(time, dy, label="Open-loop Y disturbance", alpha=0.8)
plt.plot(time, ey, label="Closed-loop Y error", alpha=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Position error (arbitrary units)")
plt.title("Two-Axis OGS Fine Tracking")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(8, 5))
plt.hist(
    results[:, 2],
    bins=10,
    edgecolor="black"
)
plt.xlabel("Radial RMS error reduction (%)")
plt.ylabel("Number of independent runs")
plt.title("Two-Axis Controller Performance Across Runs")
plt.grid(True)
plt.tight_layout()

plt.show()
