# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated 2D physics for the drone intercept game."""

from libc.math cimport sin, cos, sqrt, atan2, M_PI, fabs

# ── Constants ────────────────────────────────────────────────────────
cdef double GRAVITY = 300.0        # px/s²  (downward)
cdef double PLAYER_THRUST = 600.0  # px/s²
cdef double PLAYER_TURN = 4.0      # rad/s
cdef double DRAG = 0.98            # per-frame velocity damping
cdef double MAX_SPEED = 800.0

# ── Structs packed as C arrays for speed ─────────────────────────────
# Drone state: [x, y, vx, vy, angle, radius]
#               0  1   2   3    4      5

cpdef void player_update(double[:] s, double dt, int turn, int thrust) noexcept:
    """Update player drone state in-place.
    turn:  -1 left, 0 none, +1 right
    thrust: 1 if thrusting, else 0
    """
    # Rotation
    s[4] += PLAYER_TURN * turn * dt

    # Thrust (angle 0 = up)
    if thrust:
        s[2] += sin(s[4]) * PLAYER_THRUST * dt
        s[3] -= cos(s[4]) * PLAYER_THRUST * dt

    # Gravity
    s[3] += GRAVITY * dt

    # Drag
    s[2] *= DRAG
    s[3] *= DRAG

    # Speed cap
    cdef double speed = sqrt(s[2] * s[2] + s[3] * s[3])
    if speed > MAX_SPEED:
        s[2] = s[2] / speed * MAX_SPEED
        s[3] = s[3] / speed * MAX_SPEED

    # Position
    s[0] += s[2] * dt
    s[1] += s[3] * dt


cpdef void target_update(double[:] s, double dt,
                         double cx, double cy,
                         double orbit_radius, double orbit_speed,
                         double phase) noexcept:
    """Update target drone – flies a smooth figure-8-ish patrol."""
    phase_val = phase  # caller tracks phase
    s[0] = cx + orbit_radius * sin(phase_val)
    s[1] = cy + orbit_radius * 0.4 * sin(phase_val * 2.0)
    # Derive velocity for visual tilt
    s[2] = orbit_radius * cos(phase_val) * orbit_speed
    s[3] = orbit_radius * 0.8 * cos(phase_val * 2.0) * orbit_speed
    # Face direction of travel
    s[4] = atan2(s[2], -s[3])


cpdef double distance(double[:] a, double[:] b) noexcept:
    """Euclidean distance between two drone centres."""
    cdef double dx = a[0] - b[0]
    cdef double dy = a[1] - b[1]
    return sqrt(dx * dx + dy * dy)


cpdef bint check_collision(double[:] a, double[:] b) noexcept:
    """Circle-circle collision test."""
    cdef double dx = a[0] - b[0]
    cdef double dy = a[1] - b[1]
    cdef double dist = sqrt(dx * dx + dy * dy)
    return dist < (a[5] + b[5])


cpdef double relative_speed(double[:] a, double[:] b) noexcept:
    """Relative speed at moment of collision (for scoring)."""
    cdef double dvx = a[2] - b[2]
    cdef double dvy = a[3] - b[3]
    return sqrt(dvx * dvx + dvy * dvy)


cpdef void camera_project(double[:] player, double[:] target,
                          double fov, int cam_width,
                          double[:] out) noexcept:
    """Project the target into the interceptor's 1D forward-facing camera.

    The camera looks along the player's heading (angle, where 0 = up).
    Returns via `out`:
      out[0] = centre pixel (0..cam_width-1), or -1 if off-screen
      out[1] = width in pixels of the target's apparent size
      out[2] = signed bearing offset in radians (negative = target is left)

    Player heading convention: angle 0 points up (-Y), positive = clockwise.
    We convert to standard math angle for the projection.
    """
    cdef double dx = target[0] - player[0]
    cdef double dy = target[1] - player[1]
    cdef double dist = sqrt(dx * dx + dy * dy)

    if dist < 1.0:
        # Basically on top of the target
        out[0] = cam_width / 2.0
        out[1] = cam_width
        out[2] = 0.0
        return

    # Angle from player to target in world coords (atan2 gives angle from +X axis)
    cdef double angle_to_target = atan2(dy, dx)

    # Player's forward direction: heading angle 0 = up = -Y
    # Convert to standard: forward = (-sin(a), -cos(a))... but easier:
    # heading in standard-angle form = -(player_angle) + pi/2  ... wait let me think.
    # In the game: angle=0 means thrust goes (sin(0), -cos(0)) = (0, -1) = up.
    # So forward direction vector is (sin(angle), -cos(angle)).
    # Standard angle of that vector: atan2(-cos(angle), sin(angle))
    cdef double heading = atan2(-cos(player[4]), sin(player[4]))

    # Bearing: signed angle from heading to target, in [-pi, pi]
    cdef double bearing = angle_to_target - heading
    # Normalise to [-pi, pi]
    while bearing > M_PI:
        bearing -= 2.0 * M_PI
    while bearing < -M_PI:
        bearing += 2.0 * M_PI

    out[2] = bearing

    cdef double half_fov = fov / 2.0

    if fabs(bearing) > half_fov:
        out[0] = -1.0
        out[1] = 0.0
        return

    # Map bearing [-half_fov .. +half_fov] to pixel [0 .. cam_width]
    out[0] = (bearing / half_fov * 0.5 + 0.5) * cam_width

    # Apparent angular size: 2 * atan(radius / dist)
    cdef double angular_size = 2.0 * atan2(target[5], dist)
    # Convert to pixels
    out[1] = (angular_size / fov) * cam_width
    if out[1] < 1.0:
        out[1] = 1.0


cpdef void clamp_world(double[:] s, double h) noexcept:
    """Clamp drone vertically: bounce off ground, soft ceiling. No horizontal wrap."""
    if s[1] > h:
        s[1] = h
        s[3] = -fabs(s[3]) * 0.3  # gentle bounce off ground
    if s[1] < -200:  # soft ceiling
        s[1] = -200
        s[3] = fabs(s[3]) * 0.3
