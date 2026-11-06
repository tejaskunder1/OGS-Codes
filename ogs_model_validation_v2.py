
import numpy as np

# ============================================================
# OGS Thesis - Test 1: Model Validation
# ============================================================

KP = 1.6
KI = 0.2
KD = 0.005

TAU = 0.15
MAX_COMMAND = 5.0
DURATION = 20.0


def simulate(dt, use_controller=True):
    """Simulate two-axis pointing error with optional PID feedback."""

    time = np.arange(0, DURATION, dt)
    n = len(time)

    # Deterministic disturbances: identical for open and closed loop
    dx = (
        2.0 * np.sin(2 * np.pi * 0.15 * time)
        + np.sin(2 * np.pi * 1.2 * time)
    )

    dy = (
        1.5 * np.sin(2 * np.pi * 0.20 * time + 0.7)
        + 0.8 * np.sin(2 * np.pi * 1.7 * time)
    )

    ax = np.zeros(n)
    ay = np.zeros(n)

    ex = np.zeros(n)
    ey = np.zeros(n)

    ux = np.zeros(n)
    uy = np.zeros(n)

    ix = 0.0
    iy = 0.0
    prev_x = 0.0
    prev_y = 0.0

    for k in range(n):
        # Residual pointing error
        ex[k] = dx[k] - ax[k]
        ey[k] = dy[k] - ay[k]

        if k == n - 1:
            break

        if use_controller:
            ix += ex[k] * dt
            iy += ey[k] * dt

            der_x = (ex[k] - prev_x) / dt
            der_y = (ey[k] - prev_y) / dt

            ux[k] = np.clip(
                KP * ex[k] + KI * ix + KD * der_x,
                -MAX_COMMAND, MAX_COMMAND
            )

            uy[k] = np.clip(
                KP * ey[k] + KI * iy + KD * der_y,
                -MAX_COMMAND, MAX_COMMAND
            )

            prev_x = ex[k]
            prev_y = ey[k]

        # First-order actuator dynamics
        ax[k + 1] = ax[k] + dt / TAU * (ux[k] - ax[k])
        ay[k + 1] = ay[k] + dt / TAU * (uy[k] - ay[k])

    # Exclude initial transient from RMS calculation
    start = int(1.0 / dt)

    open_rms = np.sqrt(np.mean(
        dx[start:]**2 + dy[start:]**2
    ))

    closed_rms = np.sqrt(np.mean(
        ex[start:]**2 + ey[start:]**2
    ))

    reduction = 100 * (1 - closed_rms / open_rms)

    return {
        "time": time,
        "dx": dx,
        "dy": dy,
        "ax": ax,
        "ay": ay,
        "ex": ex,
        "ey": ey,
        "ux": ux,
        "uy": uy,
        "open_rms": open_rms,
        "closed_rms": closed_rms,
        "reduction": reduction,
    }


# ------------------------------------------------------------
# TEST 1: Actuator follows a positive command
# ------------------------------------------------------------

dt = 0.01
time = np.arange(0, 1.0, dt)
actuator = np.zeros(len(time))
command = np.ones(len(time))

for k in range(len(time) - 1):
    actuator[k + 1] = actuator[k] + (
        dt / TAU
    ) * (command[k] - actuator[k])

actuator_pass = (
    actuator[-1] > 0
    and actuator[-1] < 1.0
    and np.all(np.diff(actuator) >= 0)
)

print("TEST 1: Actuator step response")
print(f"Final actuator position: {actuator[-1]:.4f}")
print("PASS" if actuator_pass else "FAIL")



# ------------------------------------------------------------
# TEST 2: Feedback direction
# ------------------------------------------------------------

positive_error = 1.0
negative_error = -1.0

positive_command = KP * positive_error
negative_command = KP * negative_error

sign_pass = (
    positive_command > 0
    and negative_command < 0
    and np.isclose(positive_command, -negative_command)
)

print("\nTEST 2: Feedback direction")
print(f"Positive error: {positive_error:.2f}")
print(f"Positive command: {positive_command:.2f}")
print(f"Negative error: {negative_error:.2f}")
print(f"Negative command: {negative_command:.2f}")
print("PASS" if sign_pass else "FAIL")

# ------------------------------------------------------------
# TEST 3: Open-loop versus closed-loop on identical inputs
# ------------------------------------------------------------

open_result = simulate(dt, use_controller=False)
closed_result = simulate(dt, use_controller=True)

print("\nTEST 3: Paired RMS comparison")
print(f"Open-loop RMS:   {open_result['open_rms']:.4f}")
print(f"Closed-loop RMS: {closed_result['closed_rms']:.4f}")
print(f"RMS reduction:   {closed_result['reduction']:.2f}%")

same_disturbance = (
    np.allclose(open_result["dx"], closed_result["dx"])
    and np.allclose(open_result["dy"], closed_result["dy"])
)

rms_pass = (
    same_disturbance
    and closed_result["closed_rms"] < closed_result["open_rms"]
)

print("PASS" if rms_pass else "FAIL")


# ------------------------------------------------------------
# TEST 4: Sensitivity to the time step
# ------------------------------------------------------------

print("\nTEST 4: Time-step sensitivity")

reductions = []

for step in [0.02, 0.01, 0.005]:
    result = simulate(step, use_controller=True)
    reductions.append(result["reduction"])

    print(
        f"dt = {step:.3f} s | "
        f"Open RMS = {result['open_rms']:.4f} | "
        f"Closed RMS = {result['closed_rms']:.4f} | "
        f"Reduction = {result['reduction']:.2f}%"
    )

# This is a diagnostic, not a universal stability criterion.
step_difference = max(reductions) - min(reductions)

print(f"Range of reductions: {step_difference:.3f} percentage points")

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\nVALIDATION SUMMARY")
print("------------------")
print(f"Actuator response:       {'PASS' if actuator_pass else 'FAIL'}")
print(f"Feedback direction:      {'PASS' if sign_pass else 'FAIL'}")
print(f"Paired RMS comparison:   {'PASS' if rms_pass else 'FAIL'}")
print("Time-step sensitivity: inspect the values above.")
print("\nThis validates a simplified model, not real telescope performance.")
