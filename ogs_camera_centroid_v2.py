
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# OGS: Synthetic Camera + Beacon Centroid + PID Tracking
# This is a computational model, not real camera data.
# ============================================================

# Simulation settings
DT = 0.05
DURATION = 20.0
TIME = np.arange(0, DURATION, DT)
N = len(TIME)

# Camera settings
IMAGE_SIZE = 64
CENTER = (IMAGE_SIZE - 1) / 2
SPOT_SIGMA = 1.5
SPOT_AMPLITUDE = 150.0
BACKGROUND = 20.0
READ_NOISE = 3.0

# Simplified actuator model
TAU = 0.15
MAX_COMMAND = 5.0

# PID gains: initial values for this pixel-based model
KP = 0.8
KI = 0.03
KD = 0.005

rng = np.random.default_rng(42)

# Pixel coordinate grids
yy, xx = np.indices((IMAGE_SIZE, IMAGE_SIZE))


def make_camera_image(x_position, y_position, rng):
    """
    Generate a synthetic camera frame.

    x_position and y_position are beacon coordinates in pixels.
    The image contains a Gaussian spot, background and read noise.
    """
    spot = SPOT_AMPLITUDE * np.exp(
        -((xx - x_position) ** 2 + (yy - y_position) ** 2)
        / (2 * SPOT_SIGMA ** 2)
    )

    image = BACKGROUND + spot
    image += rng.normal(0, READ_NOISE, image.shape)

    return image



def find_centroid(image):
    """
    Estimate the beacon centroid using a local region of interest
    and a threshold based on the simulated background noise.
    """
    # Locate the brightest pixel
    peak_y, peak_x = np.unravel_index(
        np.argmax(image), image.shape
    )

    # Extract a local region around the peak
    radius = 7

    y1 = max(0, peak_y - radius)
    y2 = min(image.shape[0], peak_y + radius + 1)
    x1 = max(0, peak_x - radius)
    x2 = min(image.shape[1], peak_x + radius + 1)

    roi = image[y1:y2, x1:x2]

    # Ignore background fluctuations below 3 noise standard deviations
    threshold = BACKGROUND + 3 * READ_NOISE
    weights = np.maximum(roi - threshold, 0)

    total = np.sum(weights)

    if total < 1e-9:
        return float(peak_x), float(peak_y)

    roi_y, roi_x = np.indices(roi.shape)

    x_centroid = x1 + np.sum(roi_x * weights) / total
    y_centroid = y1 + np.sum(roi_y * weights) / total

    return x_centroid, y_centroid


def pid_axis(error, integral, previous_error):
    """Calculate one PID correction command."""
    integral += error * DT
    derivative = (error - previous_error) / DT

    command = (
        KP * error
        + KI * integral
        + KD * derivative
    )

    command = np.clip(command, -MAX_COMMAND, MAX_COMMAND)

    return command, integral


# Generate a sample image to visualise centroid detection
sample_x = CENTER + 5.0
sample_y = CENTER - 3.0

sample_image = make_camera_image(sample_x, sample_y, rng)
detected_x, detected_y = find_centroid(sample_image)

print("Sample beacon detection")
print("-----------------------")
print(f"True position:     ({sample_x:.3f}, {sample_y:.3f}) px")
print(f"Detected position: ({detected_x:.3f}, {detected_y:.3f}) px")
print(f"X detection error: {detected_x - sample_x:.3f} px")
print(f"Y detection error: {detected_y - sample_y:.3f} px")

# ------------------------------------------------------------
# Create repeatable two-axis disturbances
# ------------------------------------------------------------

disturbance_x = (
    2.0 * np.sin(2 * np.pi * 0.15 * TIME)
    + 1.0 * np.sin(2 * np.pi * 1.2 * TIME)
)

disturbance_y = (
    1.5 * np.sin(2 * np.pi * 0.20 * TIME + 0.7)
    + 0.8 * np.sin(2 * np.pi * 1.7 * TIME)
)

# Actual actuator corrections
actuator_x = np.zeros(N)
actuator_y = np.zeros(N)

command_x = np.zeros(N)
command_y = np.zeros(N)

true_error_x = np.zeros(N)
true_error_y = np.zeros(N)

measured_error_x = np.zeros(N)
measured_error_y = np.zeros(N)

detected_x_history = np.zeros(N)
detected_y_history = np.zeros(N)

integral_x = 0.0
integral_y = 0.0
previous_x = 0.0
previous_y = 0.0

# ------------------------------------------------------------
# Closed-loop simulation using camera-derived measurements
# ------------------------------------------------------------

for k in range(N):
    # Residual error after actual actuator correction
    true_error_x[k] = disturbance_x[k] - actuator_x[k]
    true_error_y[k] = disturbance_y[k] - actuator_y[k]

    # Beacon location in camera coordinates
    true_x = CENTER + true_error_x[k]
    true_y = CENTER + true_error_y[k]

    # Generate camera image and detect centroid
    image = make_camera_image(true_x, true_y, rng)
    cx, cy = find_centroid(image)

    detected_x_history[k] = cx
    detected_y_history[k] = cy

    # Measured error relative to image centre
    measured_error_x[k] = cx - CENTER
    measured_error_y[k] = cy - CENTER

    # Compute correction commands
    command_x[k], integral_x = pid_axis(
        measured_error_x[k], integral_x, previous_x
    )
    command_y[k], integral_y = pid_axis(
        measured_error_y[k], integral_y, previous_y
    )

    previous_x = measured_error_x[k]
    previous_y = measured_error_y[k]

    # Actuator response for the next frame
    if k < N - 1:
        actuator_x[k + 1] = actuator_x[k] + (
            DT / TAU
        ) * (command_x[k] - actuator_x[k])

        actuator_y[k + 1] = actuator_y[k] + (
            DT / TAU
        ) * (command_y[k] - actuator_y[k])


# ------------------------------------------------------------
# Calculate performance after the first second
# ------------------------------------------------------------

start = int(1.0 / DT)

open_radial = np.sqrt(
    disturbance_x ** 2 + disturbance_y ** 2
)

closed_radial = np.sqrt(
    true_error_x ** 2 + true_error_y ** 2
)

open_rms = np.sqrt(np.mean(open_radial[start:] ** 2))
closed_rms = np.sqrt(np.mean(closed_radial[start:] ** 2))

reduction = 100 * (1 - closed_rms / open_rms)

centroid_error_x = detected_x_history - (
    CENTER + true_error_x
)
centroid_error_y = detected_y_history - (
    CENTER + true_error_y
)

centroid_rms = np.sqrt(np.mean(
    centroid_error_x[start:] ** 2
    + centroid_error_y[start:] ** 2
))

print("\nCamera-based tracking results")
print("-----------------------------")
print(f"Open-loop radial RMS: {open_rms:.4f} px")
print(f"Closed-loop radial RMS: {closed_rms:.4f} px")
print(f"RMS error reduction: {reduction:.2f}%")
print(f"Centroid localisation RMS: {centroid_rms:.4f} px")
print("All results are simulated, not measured telescope performance.")

# ------------------------------------------------------------
# Plot 1: Synthetic camera frame and detected centroid
# ------------------------------------------------------------

plt.figure(figsize=(7, 6))
plt.imshow(sample_image, origin="lower", cmap="gray")
plt.scatter(
    [sample_x], [sample_y],
    marker="x", s=100, label="True position"
)
plt.scatter(
    [detected_x], [detected_y],
    marker="+", s=100, label="Detected centroid"
)
plt.xlabel("X pixel")
plt.ylabel("Y pixel")
plt.title("Synthetic Beacon Detection")
plt.colorbar(label="Pixel intensity")
plt.legend()
plt.tight_layout()

# Plot 2: True residual error and camera-measured error
plt.figure(figsize=(10, 5))
plt.plot(TIME, true_error_x, label="True X error")
plt.plot(TIME, measured_error_x, alpha=0.7,
         label="Measured X error")
plt.plot(TIME, true_error_y, label="True Y error")
plt.plot(TIME, measured_error_y, alpha=0.7,
         label="Measured Y error")
plt.xlabel("Time (s)")
plt.ylabel("Position error (pixels)")
plt.title("Camera-Based Tracking Error")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Plot 3: Radial tracking error
plt.figure(figsize=(10, 5))
plt.plot(TIME, open_radial, label="Open-loop radial error")
plt.plot(TIME, closed_radial, label="Closed-loop radial error")
plt.xlabel("Time (s)")
plt.ylabel("Radial error (pixels)")
plt.title("OGS Tracking with Camera-Based Feedback")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
