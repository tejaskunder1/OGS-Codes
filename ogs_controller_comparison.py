
import numpy as np

# OGS Fine Tracking: P vs PD vs PID
# Simulation only - not measured telescope performance

DT = 0.01
DURATION = 10.0
TAU = 0.15
MAX_COMMAND = 5.0
TRIALS = 30

# Use the same proportional gain for a fair baseline.
KP = 0.8
KD = 0.005
KI = 0.03

FREQUENCIES = [0.5, 1.2, 2.5]
NOISE_LEVEL = 3.0

CONTROLLERS = ["P", "PD", "PID"]


def simulate(controller, frequency, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(0, DURATION, DT)
    n = len(t)

    # Simulated external pointing disturbance.
    disturbance = (
        1.0 * np.sin(2 * np.pi * frequency * t)
        + 0.35 * np.sin(2 * np.pi * 0.23 * t + 0.4)
        + rng.normal(0, 0.03, n)
    )

    # First-order actuator and tracking states.
    actuator = 0.0
    integral = 0.0
    previous_error = 0.0
    residual = np.zeros(n)
    saturation_count = 0

    for k in range(1, n):
        # Residual pointing error before correction.
        error = disturbance[k] - actuator

        # Simulated centroid measurement noise.
        measured_error = error + rng.normal(
            0, NOISE_LEVEL * 0.001
        )

        integral += measured_error * DT
        derivative = (measured_error - previous_error) / DT

        command = KP * measured_error

        if controller in ["PD", "PID"]:
            command += KD * derivative

        if controller == "PID":
            command += KI * integral

        clipped_command = np.clip(
            command, -MAX_COMMAND, MAX_COMMAND
        )

        if clipped_command != command:
            saturation_count += 1

        # First-order actuator response.
        actuator += (DT / TAU) * (
            clipped_command - actuator
        )

        residual[k] = disturbance[k] - actuator
        previous_error = measured_error

    rms = np.sqrt(np.mean(residual[int(1 / DT):] ** 2))
    saturation_pct = 100 * saturation_count / (n - 1)

    return rms, saturation_pct


def main():
    print("OGS Controller Comparison: P vs PD vs PID")
    print("------------------------------------------")
    print(f"Trials per controller/frequency: {TRIALS}")
    print(f"KP={KP}, KD={KD}, KI={KI}")
    print(f"Camera noise scale: {NOISE_LEVEL}")
    print()

    for frequency in FREQUENCIES:
        results = {name: [] for name in CONTROLLERS}
        saturation = {name: [] for name in CONTROLLERS}
        open_rms_values = []

        for trial in range(TRIALS):
            seed = 1000 + trial
            rng = np.random.default_rng(seed)

            t = np.arange(0, DURATION, DT)
            disturbance = (
                1.0 * np.sin(2 * np.pi * frequency * t)
                + 0.35 * np.sin(2 * np.pi * 0.23 * t + 0.4)
                + rng.normal(0, 0.03, len(t))
            )
            open_rms_values.append(
                np.sqrt(np.mean(disturbance[int(1 / DT):] ** 2))
            )

            # Each controller receives the same trial seed.
            for name in CONTROLLERS:
                rms, sat = simulate(name, frequency, seed)
                results[name].append(rms)
                saturation[name].append(sat)

        open_mean = np.mean(open_rms_values)
        print(f"Vibration frequency: {frequency:.1f} Hz")
        print(f"Open-loop RMS: {open_mean:.4f}")

        for name in CONTROLLERS:
            values = np.array(results[name])
            reduction = 100 * (open_mean - np.mean(values)) / open_mean

            print(
                f"{name:>3}: RMS = {np.mean(values):.4f} "
                f"+/- {np.std(values, ddof=1):.4f}, "
                f"reduction = {reduction:.2f}%, "
                f"saturation = {np.mean(saturation[name]):.3f}%"
            )

        print()


if __name__ == "__main__":
    main()
