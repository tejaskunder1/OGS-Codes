
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# OGS PID Robustness Test
# Fixed PID gains; independent random test conditions
# ============================================================

DT = 0.01
DURATION = 20.0
TAU = 0.08
MAX_COMMAND = 3.0

KP = 1.6
KI = 0.2
KD = 0.005

time = np.arange(0, DURATION, DT)
N = len(time)
START = int(1.0 / DT)  # Exclude first second from steady RMS


def simulate(frequency, noise_std, seed):
    """Simulate open-loop and closed-loop tracking."""
    rng = np.random.default_rng(seed)

    # Repeatable physical disturbance for this test case
    drift = 0.8 * np.sin(2 * np.pi * 0.15 * time)
    vibration = 1.0 * np.sin(2 * np.pi * frequency * time)
    jitter = 0.20 * rng.normal(size=N)
    disturbance = drift + vibration + jitter

    # Independent measurement noise
    noise = noise_std * rng.normal(size=N)

    actuator = np.zeros(N)
    command = np.zeros(N)
    residual = np.zeros(N)

    integral = 0.0
    previous_error = 0.0

    for k in range(N - 1):
        residual[k] = disturbance[k] - actuator[k]
        measured = residual[k] + noise[k]

        integral += measured * DT
        derivative = (measured - previous_error) / DT

        raw_command = (
            KP * measured
            + KI * integral
            + KD * derivative
        )

        command[k] = np.clip(
            raw_command, -MAX_COMMAND, MAX_COMMAND
        )

        actuator[k + 1] = actuator[k] + (
            DT / TAU
        ) * (command[k] - actuator[k])

        previous_error = measured

    residual[-1] = disturbance[-1] - actuator[-1]
    command[-1] = command[-2]

    open_rms = np.sqrt(np.mean(disturbance[START:] ** 2))
    closed_rms = np.sqrt(np.mean(residual[START:] ** 2))
    reduction = 100 * (1 - closed_rms / open_rms)

    saturation = 100 * np.mean(
        np.abs(command[START:]) >= MAX_COMMAND
    )

    return open_rms, closed_rms, reduction, saturation, residual


# Test conditions
frequencies = [0.5, 1.2, 2.5]       # Hz
noise_levels = [0.05, 0.10, 0.25]   # Arbitrary position units
runs = 5

results = []

for frequency in frequencies:
    for noise_std in noise_levels:
        run_results = []

        for run in range(runs):
            # Different random seeds from the original baseline
            seed = 1000 + run
            result = simulate(frequency, noise_std, seed)
            run_results.append(result[:4])

        run_results = np.array(run_results)

        results.append({
            "frequency": frequency,
            "noise": noise_std,
            "open_rms": np.mean(run_results[:, 0]),
            "closed_rms": np.mean(run_results[:, 1]),
            "reduction": np.mean(run_results[:, 2]),
            "saturation": np.mean(run_results[:, 3]),
        })


# Print results
print("\nOGS PID robustness study")
print("Fixed gains:", KP, KI, KD)
print("-" * 79)
print(
    f"{'Freq (Hz)':>10} {'Noise':>9} {'Open RMS':>11} "
    f"{'PID RMS':>11} {'Reduction %':>13} {'Sat. %':>9}"
)

for r in results:
    print(
        f"{r['frequency']:10.2f} {r['noise']:9.2f} "
        f"{r['open_rms']:11.4f} {r['closed_rms']:11.4f} "
        f"{r['reduction']:13.2f} {r['saturation']:9.2f}"
    )


# Plot reduction in RMS error versus vibration frequency.
plt.figure(figsize=(9, 5))

for noise_std in noise_levels:
    subset = [r for r in results if r["noise"] == noise_std]
    plt.plot(
        [r["frequency"] for r in subset],
        [r["reduction"] for r in subset],
        marker="o",
        label=f"Measurement noise = {noise_std}"
    )

plt.axhline(0, linewidth=0.8)
plt.xlabel("Vibration frequency (Hz)")
plt.ylabel("RMS error reduction (%)")
plt.title("PID Robustness Under Different Disturbances")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
