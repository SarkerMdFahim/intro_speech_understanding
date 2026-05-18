import numpy as np
import matplotlib.pyplot as plt

def center_of_gravity(x):
    '''
    Find the center of gravity of a vector, x.
    '''
    
    indices = np.arange(len(x))
    c = np.dot(indices, x) / np.sum(x)
    
    return c

def matched_identity(x):
    '''
    Create an identity matrix that has the same number of rows as x has elements.
    '''
    
    I = np.eye(len(x))
    
    return I

def sine_and_cosine(t_start, t_end, t_steps):
    '''
    Create a time axis, and compute cosine and sine values.
    '''
    
    t = np.linspace(t_start, t_end, t_steps)
    x = np.cos(t)
    y = np.sin(t)
    
    return t, x, y


# =========================
# Example Inputs and Outputs
# =========================

# Example for center_of_gravity
x = np.array([1, 2, 3, 4])

c = center_of_gravity(x)

print("Input Vector:")
print(x)

print("\nCenter of Gravity:")
print(c)


# Example for matched_identity
I = matched_identity(x)

print("\nIdentity Matrix:")
print(I)


# Example for sine_and_cosine
t, cos_values, sin_values = sine_and_cosine(0, 2*np.pi, 100)

print("\nTime Axis:")
print(t)

print("\nCosine Values:")
print(cos_values)

print("\nSine Values:")
print(sin_values)


# Plotting the sine and cosine waves
plt.plot(t, cos_values, label='Cosine')
plt.plot(t, sin_values, label='Sine')

plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sine and Cosine Waves')

plt.legend()
plt.grid(True)

plt.show()
