
import numpy as np

# ============================================================
# OGS Thesis - Test 2: Camera Noise Robustness
# ============================================================

DT = 0.05
DURATION = 10.0
TIME = np.arange(0, DURATION, DT)
N = len(TIME)
START = int(1.0 / DT)

IMAGE_SIZE = 64
CENTER = (IMAGE_SIZE - 1) / 2

BACKGROUND = 20.0
SPOT_AMPLITUDE = 150.0
SPOT_SIGMA = 1.5

TAU = 0.15
MAX_COMMAND = 5.0

KP = 0.8
KI = 0.03
KD = 0.005

NOISE_LEVELS = [1.0, 3.0, 6.0]
RUNS = 30

yy, xx = np.indices((IMAGE_SIZE, IMAGE_SIZE))


def make_camera_image(x_pos, y_pos, noise_std, rng):
    """Generate a synthetic Gaussian beacon and noisy camera frame."""
    spot = SPOT_AMPLITUDE * np.exp(
        -((xx - x_pos)**2 + (yy - y_pos)**2)
        / (2 * SPOT_SIGMA**2)
    )

    return (
        BACKGROUND + spot
        + rng.normal(0, noise_std, (IMAGE_SIZE, IMAGE_SIZE))
    )


def find_centroid(image, noise_std):
    """Find the brightest spot and estimate its local centroid."""
    peak_y, peak_x = np.unravel_index(
        np.argmax(image), image.shape
    )

    radius = 7
    y1 = max(0, peak_y - radius)
    y2 = min(IMAGE_SIZE, peak_y + radius + 1)
    x1 = max(0, peak_x - radius)
    x2 = min(IMAGE_SIZE, peak_x + radius + 1)

    roi = image[y1:y2, x1:x2]

    threshold = BACKGROUND + 3 * noise_std
    weights = np.maximum(roi - threshold, 0)
    total = np.sum(weights)

    if total < 1e-9:
        return float(peak_x), float(peak_y)

    ry, rx = np.indices(roi.shape)

    cx = x1 + np.sum(rx * weights) / total
    cy = y1 + np.sum(ry * weights) / total

    return cx, cy


def simulate(noise_std, seed):
    """Run one two-axis tracking trial."""

    rng = np.random.default_rng(seed)

    # Fixed disturbance for this seed.
    # Identical across noise levels when the seed is identical.
    dx = (
        2.0 * np.sin(2 * np.pi * 0.15 * TIME)
        + np.sin(2 * np.pi * 1.2 * TIME)
    )
    dy = (
        1.5 * np.sin(2 * np.pi * 0.20 * TIME + 0.7)
        + 0.8 * np.sin(2 * np.pi * 1.7 * TIME)
    )

    ax = np.zeros(N)
    ay = np.zeros(N)

    ex = np.zeros(N)
    ey = np.zeros(N)

    centroid_sq = np.zeros(N)

    ix = iy = 0.0
    prev_x = prev_y = 0.0

    for k in range(N):
        # True residual pointing error
        ex[k] = dx[k] - ax[k]
        ey[k] = dy[k] - ay[k]

        true_x = CENTER + ex[k]
        true_y = CENTER + ey[k]

        # Camera image and measured centroid
        image = make_camera_image(
            true_x, true_y, noise_std, rng
        )

        cx, cy = find_centroid(image, noise_std)

        measured_x = cx - CENTER
        measured_y = cy - CENTER

        centroid_sq[k] = (
            (cx - true_x)**2 + (cy - true_y)**2
        )

        if k == N - 1:
            break

        ix += measured_x * DT
        iy += measured_y * DT

        der_x = (measured_x - prev_x) / DT
        der_y = (measured_y - prev_y) / DT

        ux = np.clip(
            KP * measured_x + KI * ix + KD * der_x,
            -MAX_COMMAND, MAX_COMMAND
        )
        uy = np.clip(
            KP * measured_y + KI * iy + KD * der_y,
            -MAX_COMMAND, MAX_COMMAND
        )

        prev_x = measured_x
        prev_y = measured_y

        # First-order actuator response
        ax[k + 1] = ax[k] + DT / TAU * (ux - ax[k])
        ay[k + 1] = ay[k] + DT / TAU * (uy - ay[k])

    # Evaluate after initial transient
    open_rms = np.sqrt(np.mean(
        dx[START:]**2 + dy[START:]**2
    ))

    closed_rms = np.sqrt(np.mean(
        ex[START:]**2 + ey[START:]**2
    ))

    reduction = 100 * (1 - closed_rms / open_rms)

    centroid_rms = np.sqrt(np.mean(centroid_sq[START:]))

    return open_rms, closed_rms, reduction, centroid_rms


# ------------------------------------------------------------
# Run 30 trials at each noise level
# ------------------------------------------------------------

print("OGS Camera Noise Robustness Study")
print("---------------------------------")
print(f"Runs per noise level: {RUNS}")
print(f"PID gains: Kp={KP}, Ki={KI}, Kd={KD}")
print()

for noise_std in NOISE_LEVELS:
    results = []

    for run in range(RUNS):
        # Reusing seeds pairs the disturbance and underlying
        # standard-normal noise draws across noise levels.
        result = simulate(noise_std, seed=1000 + run)
        results.append(result)

    results = np.array(results)

    open_mean = np.mean(results[:, 0])
    closed_mean = np.mean(results[:, 1])
    reduction_mean = np.mean(results[:, 2])
    reduction_sd = np.std(results[:, 2], ddof=1)

    centroid_mean = np.mean(results[:, 3])
    centroid_sd = np.std(results[:, 3], ddof=1)

    print(f"Camera read noise: {noise_std:.1f}")
    print(f"Mean open-loop RMS: {open_mean:.4f} px")
    print(
        f"Mean closed-loop RMS: {closed_mean:.4f} px"
    )
    print(
        f"RMS reduction: {reduction_mean:.2f} "
        f"+/- {reduction_sd:.2f} percentage points"
    )
    print(
        f"Centroid RMS: {centroid_mean:.4f} "
        f"+/- {centroid_sd:.4f} px"
    )
    print()
    
print("All results are simulated, not measured telescope performance.")
