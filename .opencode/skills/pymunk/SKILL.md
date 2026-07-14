---
name: pymunk
description: "2D physics simulation in Python using Pymunk (built on Chipmunk2D). Use when the user wants to simulate rigid body physics, collisions, joints, constraints, or game physics in 2D. Triggers on requests involving: physics simulation, rigid bodies, collision detection, joints, ragdolls, game physics, falling objects, bouncing, or any 2D physical interaction in Python. Install with `pip install pymunk`."
---

# Pymunk 2D Physics

Pymunk is a 2D rigid body physics library for Python, built on Chipmunk2D.

## Installation

```bash
pip install pymunk
```

## Core Concepts

### Space
The main simulation container. All bodies, shapes, and constraints belong to a space.

```python
import pymunk

space = pymunk.Space()
space.gravity = (0, -981)  # px/s², downward
```

### Bodies
Rigid bodies with mass, moment, and position.

```python
# Static body (infinite mass, doesn't move)
static_body = space.static_body

# Dynamic body (moves under forces)
body = pymunk.Body(mass=1, moment=pymunk.moment_for_box(mass=1, w=50, h=50))
body.position = (400, 500)

# Kinematic body (user-controlled velocity)
body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
body.velocity = (100, 0)

space.add(body)
```

### Shapes
Collision shapes attached to bodies. Define physical boundaries.

```python
# Circle
shape = pymunk.Circle(body, radius=25)
shape.elasticity = 0.8
shape.friction = 0.5
space.add(shape)

# Box/polygon
shape = pymunk.Poly.create_box(body, size=(50, 50))
space.add(shape)

# Segment (line)
shape = pymunk.Segment(static_body, (0, 0), (800, 0), 5)  # floor
space.add(shape)
```

### Constraints (Joints)
Connect bodies together.

```python
# Pin joint (fixed distance)
j = pymunk.PinJoint(body_a, body_b, (0, 0), (0, 0))
space.add(j)

# Pivot joint (rotate around point)
j = pymunk.PivotJoint(body_a, body_b, (400, 400))

# Slide joint (constrained range)
j = pymunk.SlideJoint(body_a, body_b, (0, 0), (0, 0), min=10, max=50)

# Groove joint (slide along a line)
j = pymunk.GrooveJoint(body_a, body_b, groove_a=(0, 0), groove_b=(100, 0), anchor_b=(0, 0))

# Damped spring
j = pymunk.DampedSpring(body_a, body_b, (0, 0), (0, 0), rest_length=50, stiffness=100, damping=10)

# Simple motor (rotates at constant speed)
j = pymunk.SimpleMotor(body_a, body_b, rate=2.0)
space.add(j)
```

### Collision Handling

```python
def collision_handler(arbiter, space, data):
    # Return True to process collision, False to ignore
    return True

handler = space.add_collision_handler(collision_type_a=1, collision_type_b=2)
handler.begin = collision_handler
handler.separate = lambda a, s, d: print("collision ended")

# Set collision types on shapes
shape.collision_type = 1
```

### Sensors and Filters

```python
# Sensor shape (detects overlap without physical response)
shape.sensor = True

# Collision filter (control what interacts)
filter = pymunk.ShapeFilter(group=1, categories=0b0011, mask=0b0001)
shape.filter = filter
```

## Simulation Loop

```python
import pygame

def run():
    space = pymunk.Space()
    space.gravity = (0, -981)
    
    # Setup bodies and shapes...
    
    dt = 1.0 / 60.0
    while running:
        space.step(dt)  # Advance simulation by dt seconds
        
        # Update visuals from body positions
        for body in space.bodies:
            x, y = body.position
            angle = body.angle
        
        # Render...

run()
```

## Debug Drawing

```python
import pymunk.pygame_util

draw_options = pymunk.pygame_util.DrawOptions(screen)
space.debug_draw(draw_options)
```

## Common Patterns

### Stacking Boxes

```python
def add_box(space, x, y, w=50, h=50):
    mass = 1
    moment = pymunk.moment_for_box(mass, w, h)
    body = pymunk.Body(mass, moment)
    body.position = (x, y)
    shape = pymunk.Poly.create_box(body, (w, h))
    shape.friction = 0.7
    space.add(body, shape)
```

### Ragdoll (Simple Chain)

```python
def create_ragdoll(space, x, y):
    segment_count = 5
    segment_length = 30
    bodies = []
    for i in range(segment_count):
        mass = 1
        moment = pymunk.moment_for_circle(mass, 0, 10)
        body = pymunk.Body(mass, moment)
        body.position = (x, y - i * segment_length)
        shape = pymunk.Circle(body, 10)
        space.add(body, shape)
        bodies.append(body)
        if i > 0:
            j = pymunk.PinJoint(bodies[i-1], body, (0, -segment_length//2), (0, segment_length//2))
            space.add(j)
```

### Mouse Interaction

```python
mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)

def update_mouse(pos):
    mouse_body.position = pos

# Constrain grabbed body to mouse
grab_joint = pymunk.PivotJoint(grabbed_body, mouse_body, (0, 0), (0, 0))
space.add(grab_joint)
```

## Sleep, Damping, and Performance

```python
space.sleep_time_threshold = 0.5  # Bodies sleep if idle (seconds)
space.idle_speed_threshold = 5.0  # Speed below threshold to sleep
space.damping = 0.8  # Velocity damping per step (1.0 = none)
space.collision_bias = 0.01  # Collision resolution bias
space.collision_slop = 0.01  # Allowed penetration
```

## Key Properties

```python
body.position          # (x, y)
body.velocity          # (vx, vy)
body.angle             # radians
body.angular_velocity  # rad/s
body.force             # accumulated force
body.torque            # accumulated torque
body.mass              # mass (settable)
body.moment            # moment of inertia
shape.elasticity       # bounciness (0.0 - 1.0)
shape.friction         # friction coefficient
shape.surface_velocity # conveyor belt effect
```
