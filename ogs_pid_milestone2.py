
import numpy as np
import matplotlib.pyplot as plt


dt = 0.01
duration = 20.0
time = np.arange(0, duration, dt)
N = len(time)


rng = np.random.default_rng(42)

slow_drift = 0.8 * np.sin(2 * np.pi * 0.15 * time)
mechanical_vibration = 1.0 * np.sin(2 * np.pi * 1.2 * time)
random_jitter = 0.20 * rng.normal(size=N)

disturbance = slow_drift + mechanical_vibration + random_jitter
measurement_noise = 0.10 * rng.normal(size=N)


tau = 0.08                 # actuator time constant (s)
max_command = 3.0          # actuator command limit (arbitrary units)


def rms(values):
    return np.sqrt(np.mean(values ** 2))


def simulate_pid(Kp, Ki, Kd):
    actuator = np.zeros(N)
    residual = np.zeros(N)
    measured = np.zeros(N)
    command = np.zeros(N)

    integral = 0.0
    previous_error = 0.0

    for k in range(N - 1):
        # Residual pointing error before correction
        residual[k] = disturbance[k] - actuator[k]

        # Noisy measurement available to controller
        measured[k] = residual[k] + measurement_noise[k]

        # Discrete PID terms
        integral += measured[k] * dt
        derivative = (measured[k] - previous_error) / dt

        raw_command = (
            Kp * measured[k]
            + Ki * integral
            + Kd * derivative
        )

        # Limit the requested actuator correction
        command[k] = np.clip(
            raw_command, -max_command, max_command
        )

        # First-order actuator response
        actuator[k + 1] = actuator[k] + (
            dt / tau
        ) * (command[k] - actuator[k])

        previous_error = measured[k]

    # Calculate the final sample consistently
    residual[-1] = disturbance[-1] - actuator[-1]
    measured[-1] = residual[-1] + measurement_noise[-1]
    command[-1] = command[-2]

    return residual, actuator, command



open_loop = disturbance.copy()
open_rms = rms(open_loop)


# These are trial values
Kp_values = [0.2, 0.5, 0.8, 1.2, 1.6]
Ki_values = [0.0, 0.05, 0.10, 0.20]
Kd_values = [0.0, 0.002, 0.005, 0.01]

best = None

for Kp in Kp_values:
    for Ki in Ki_values:
        for Kd in Kd_values:
            error, actuator, command = simulate_pid(Kp, Ki, Kd)

            # Ignore the first second when comparing steady behaviour.
            steady_error = error[int(1.0 / dt):]
            score = rms(steady_error)

            # Reject clearly divergent or non-finite simulations.
            if not np.all(np.isfinite(error)):
                continue
            if np.max(np.abs(error)) > 10 * max_command:
                continue

            if best is None or score < best["score"]:
                best = {
                    "Kp": Kp,
                    "Ki": Ki,
                    "Kd": Kd,
                    "score": score,
                    "error": error,
                    "actuator": actuator,
                    "command": command,
                }

if best is None:
    raise RuntimeError("No acceptable controller found in the gain grid.")

# 8. Evaluate the selected candidate
closed_loop = best["error"]
start = int(1.0 / dt)

open_steady_rms = rms(open_loop[start:])
closed_steady_rms = rms(closed_loop[start:])
reduction = 100 * (
    1 - closed_steady_rms / open_steady_rms
)

print("\nOGS PID simulation results")
print("--------------------------")
print(f"Best tested Kp: {best['Kp']}")
print(f"Best tested Ki: {best['Ki']}")
print(f"Best tested Kd: {best['Kd']}")
print(f"Open-loop steady RMS: {open_steady_rms:.4f}")
print(f"Closed-loop steady RMS: {closed_steady_rms:.4f}")
print(f"RMS reduction: {reduction:.2f}%")
print(
    "Maximum absolute residual error:",
    f"{np.max(np.abs(closed_loop)):.4f}"
)
print(
    "Fraction of commands at actuator limit:",
    f"{np.mean(np.abs(best['command']) >= max_command) * 100:.2f}%"
)
print("\nAll results are simulated, not measured telescope performance.")

# 9. Plot tracking errors
plt.figure(figsize=(10, 5))
plt.plot(time, open_loop, label="Open-loop")
plt.plot(time, closed_loop, label="PID closed-loop")
plt.axhline(0, linewidth=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Pointing error (arbitrary units)")
plt.title("OGS Fine Tracking: Open-Loop vs PID")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 10. Plot the actuator correction
plt.figure(figsize=(10, 4))
plt.plot(time, best["actuator"], label="Actuator correction")
plt.plot(time, best["command"], alpha=0.7, label="Command")
plt.xlabel("Time (s)")
plt.ylabel("Correction (arbitrary units)")
plt.title("Actuator Response and Command")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 11. Plot the error distribution
plt.figure(figsize=(8, 4))
plt.hist(open_loop[start:], bins=45, alpha=0.6, label="Open-loop")
plt.hist(closed_loop[start:], bins=45, alpha=0.6, label="PID closed-loop")
plt.xlabel("Pointing error (arbitrary units)")
plt.ylabel("Number of samples")
plt.title("Distribution of Steady Tracking Error")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
