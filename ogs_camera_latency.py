import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# OGS Camera-feedback Latency Study
# Simulation only - not measured telescope performance
# ============================================================

DT = 0.01
DURATION = 10.0
TAU = 0.15
MAX_COMMAND = 5.0
TRIALS = 30

KP = 0.8
KD = 0.005
KI = 0.03

NOISE_STD = 0.003
VIBRATION_FREQUENCY = 1.2

DELAYS_MS = [0, 50, 100, 200, 300]


def simulate(delay_ms, seed):
    rng = np.random.default_rng(seed)

    t = np.arange(0, DURATION, DT)
    n = len(t)

    # Repeatable external pointing disturbance.
    # The same seed is used for every delay condition.
    disturbance_rng = np.random.default_rng(seed)

    disturbance = (
        np.sin(2 * np.pi * VIBRATION_FREQUENCY * t)
        + 0.35 * np.sin(2 * np.pi * 0.23 * t + 0.4)
        + disturbance_rng.normal(0, 0.03, n)
    )

    # Convert latency in milliseconds into simulation time steps.
    delay_steps = round((delay_ms / 1000) / DT)

    actuator = 0.0
    integral = 0.0
    previous_error = 0.0

    residual = np.zeros(n)
    saturation_count = 0

    # Store camera measurements so delayed feedback can be used.
    measured_error_history = np.zeros(n)

    for k in range(1, n):

        # Current true tracking error.
        true_error = disturbance[k] - actuator

        # Camera measurement with measurement noise.
        measured_error_history[k] = (
            true_error + rng.normal(0, NOISE_STD)
        )

        # Use a past measurement to represent camera-feedback latency.
        measurement_index = max(0, k - delay_steps)
        measured_error = measured_error_history[measurement_index]

        # PID controller calculations.
        integral += measured_error * DT

        derivative = (
            measured_error - previous_error
        ) / DT

        command = (
            KP * measured_error
            + KD * derivative
            + KI * integral
        )

        # Limit the actuator command.
        clipped_command = np.clip(
            command,
            -MAX_COMMAND,
            MAX_COMMAND
        )

        if clipped_command != command:
            saturation_count += 1

        # First-order actuator dynamics.
        actuator += (DT / TAU) * (
            clipped_command - actuator
        )

        # Residual pointing error after actuator correction.
        residual[k] = disturbance[k] - actuator

        previous_error = measured_error

    # Exclude the first second to reduce startup effects.
    start = int(1.0 / DT)

    rms = np.sqrt(
        np.mean(residual[start:] ** 2)
    )

    saturation_pct = (
        100 * saturation_count / (n - 1)
    )

    return rms, saturation_pct


def main():

    print("OGS Camera Latency Study")
    print("------------------------")
    print(f"Trials per delay: {TRIALS}")
    print(f"Vibration frequency: {VIBRATION_FREQUENCY} Hz")
    print(f"PID gains: Kp={KP}, Ki={KI}, Kd={KD}")
    print()

    results = {}
    saturation_results = {}

    # --------------------------------------------------------
    # Run trials for each camera latency
    # --------------------------------------------------------

    for delay_ms in DELAYS_MS:

        rms_values = []
        saturation_values = []

        for trial in range(TRIALS):

            rms, saturation = simulate(
                delay_ms,
                seed=2000 + trial
            )

            rms_values.append(rms)
            saturation_values.append(saturation)

        results[delay_ms] = np.array(rms_values)
        saturation_results[delay_ms] = np.array(
            saturation_values
        )

        mean_rms = np.mean(rms_values)
        std_rms = np.std(rms_values, ddof=1)
        mean_saturation = np.mean(saturation_values)

        print(
            f"Delay: {delay_ms:>3} ms | "
            f"RMS: {mean_rms:.4f} "
            f"+/- {std_rms:.4f} | "
            f"Saturation: {mean_saturation:.3f}%"
        )

    # --------------------------------------------------------
    # Compare each delay against the zero-delay baseline
    # --------------------------------------------------------

    baseline = results[0]
    baseline_mean = np.mean(baseline)

    print()
    print("Comparison against zero-delay baseline")
    print("---------------------------------------")

    for delay_ms in DELAYS_MS:

        mean_rms = np.mean(results[delay_ms])

        change_pct = (
            100 * (mean_rms - baseline_mean)
            / baseline_mean
        )

        print(
            f"{delay_ms:>3} ms: "
            f"RMS change = {change_pct:+.2f}%"
        )

    # --------------------------------------------------------
    # Plot mean RMS pointing error against camera latency
    # --------------------------------------------------------

    mean_rms_values = [
        np.mean(results[delay_ms])
        for delay_ms in DELAYS_MS
    ]

    std_rms_values = [
        np.std(results[delay_ms], ddof=1)
        for delay_ms in DELAYS_MS
    ]

    plt.figure(figsize=(9, 5.5))

    plt.errorbar(
        DELAYS_MS,
        mean_rms_values,
        yerr=std_rms_values,
        marker="o",
        markersize=7,
        capsize=5,
        linewidth=2,
        linestyle="-"
    )

    plt.xlabel(
        "Camera-feedback latency (ms)",
        fontsize=12
    )

    plt.ylabel(
        "Closed-loop RMS pointing error (arbitrary units)",
        fontsize=12
    )

    plt.title(
        "Effect of Camera-feedback Latency on OGS Fine Tracking",
        fontsize=13
    )

    plt.xticks(DELAYS_MS)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Display graph only - do not save it.
    plt.show()

    print()
    print(
        "All results are simulated, "
        "not measured telescope performance."
    )


if __name__ == "__main__":
    main()
