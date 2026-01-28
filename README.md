# Learning Probability Density Functions using GAN (Assignment-2)

## Overview
This assignment focuses on learning an unknown probability density function (PDF) using data only, without assuming any analytical form. A Generative Adversarial Network (GAN) is used to implicitly learn the distribution of a transformed random variable derived from real-world air quality data.

---

## Dataset
- Feature used: NO2 concentration (denoted as x)
- Source: India Air Quality Dataset (Kaggle)
- The dataset contains real-valued measurements of NO2 concentration.

---

## Data Transformation
Each value of x is transformed into a new variable z using the following transformation:

z = Tr(x) = x + a_r * sin(b_r * x)

where:

a_r = 0.5 * (r mod 7)  
b_r = 0.3 * (r mod 5 + 1)

and r is the university roll number.

For roll number:

r = 102303803

we obtain:

r mod 7 = 6  
r mod 5 = 3  

a_r = 0.5 * 6 = 3.0  
b_r = 0.3 * (3 + 1) = 1.2  

Final transformation used:

z = x + 3 * sin(1.2 * x)

---

## Objective
- Assume that the transformed variable z is sampled from an unknown distribution
- Learn this distribution using a GAN
- Approximate the probability density function p_h(z) using generator samples

No parametric distribution (Gaussian, exponential, etc.) is assumed at any stage.

---

## GAN Architecture

### Generator
- Input: random noise sampled from N(0, 1)
- Fully connected neural network
- Output: generated samples z_f

### Discriminator
- Input: real samples z and fake samples z_f
- Output: probability that the input sample is real

The discriminator distinguishes between:
- Real samples: z
- Fake samples: z_f = G(error)

---

## Discriminator Noise Handling
To avoid discriminator overfitting and unstable GAN training, Gaussian noise is added to real samples during training:

z_real_noisy = z + ε  
ε ~ N(0, σ²)

This improves:
- Training stability
- Gradient flow to the generator
- Mode coverage of the learned distribution

---

## PDF Approximation
After training the GAN:
1. A large number of samples are generated from the generator
2. The probability density function p_h(z) is estimated using Kernel Density Estimation (KDE)

The resulting density represents the learned distribution of z.

---

## Observations

Mode Coverage:
- Multiple density regions are captured
- No significant mode collapse observed

Training Stability:
- Noise injection prevents discriminator dominance
- GAN training remains stable across epochs

Quality of Generated Distribution:
- Smooth and continuous estimated PDF
- No artificial spikes or discontinuities
- Distribution closely follows the empirical structure of transformed data

---

## Technologies Used
- Python
- Pandas
- NumPy
- TensorFlow / Keras
- SciPy
- Matplotlib  

---

## Plot
![image alt](https://github.com/pulkitsriv30/assign_2/blob/main/download.png)

---
## Colab Link 
https://colab.research.google.com/drive/1lCwoMvy9hgJ5Mev-g1FwHHGeuekJ4CYp?usp=sharing

## Conclusion
This assignment demonstrates that a Generative Adversarial Network can successfully learn an unknown probability density function using only sample data, without assuming any analytical distribution. Proper regularization of the discriminator plays a crucial role in achieving stable and meaningful results.
