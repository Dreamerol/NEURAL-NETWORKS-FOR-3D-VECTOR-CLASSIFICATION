<h1 align="center">🧠 NEURAL NETWORKS</h1>
## *Building Neural Networks*

![Neural Networks](https://github.com/Dreamerol/Dreamerol/blob/57256cdf74e94d8afc08a57d630287fa75743da4/!!!_NN_2.jpg)

---

# 🚀 Neural Network Lab: From Scratch to Smart Predictions

Welcome to **Neural Network Lab**, where math, code, and logic collide to create intelligent systems.  
This lab is not just coding – it’s understanding **how machines think** and **how neural networks learn from data**.  

---

## 🏆 Overview
The lab is divided into several main areas:  

- **🔹 Neural Network Design with TensorFlow** – classifying vectors in 2D & 3D.  
- **🔹 Implementing Neural Networks from Scratch** – mastering gradient descent, backpropagation, and weight optimization.  
- **🔹 Mathematical Modelling & Dynamic Systems** – applying NNs to SIR epidemiological models.  

Each section teaches **concepts, problem-solving, and visualization**, making abstract math tangible.  

**Badges:**  
`📊 TensorFlow` `🟢 Vector Classification` `⚙️ Backpropagation` `📈 Math Modelling` `💡 Scientific Method`  

---

## 📈 Key Tasks

### 1️⃣ Neural Network Design with TensorFlow
**Vector Classification:**  
- **2D Quadrants:** Determine which quadrant a vector belongs to.  
- **3D Octants:** Predict which octant a 3D vector lies in.  

**Function Prediction:**  
- Predict the behavior of the **sin(x)** function using neural networks.  

**Badges:**  
`📊 TensorFlow Expert` `🟢 Vector Classification` `📐 Function Approximation`  

---

### 2️⃣ Implementing Backpropagation from Scratch
- Calculated **derivatives manually** and applied **gradient descent**.  
- Traversed networks **layer by layer** to minimize error.  
- Experimented with **different architectures**: identity vs sigmoid.  

**Badges:**  
`⚙️ Backpropagation Pro` `💡 Gradient Descent Wizard` `🧮 Manual Calculations`  

---

### 3️⃣ Model Evaluation
- Used **accuracy & loss functions** to measure network performance.  
- Conducted three different tasks with varying NN architectures.  
- Applied **Mean Squared Error** to optimize weights.  

**Badges:**  
`✅ Model Evaluator` `📉 Loss Minimizer` `🤖 Neural Network Tester`  

---

### 4️⃣ Math Modelling & SIR Predictions
- Applied NNs to **dynamic systems**: predicting interactions among infected, recovered, and sustainable populations.  
- Learned to **map current values to previous ones**, enabling **time-based predictions**.  
- Plotted **SIR trajectory** to visualize epidemic evolution.  

**Badges:**  
`📊 SIR Modeller` `🌡️ Epidemic Predictor` `📈 Data Visualizer`  

### 📊 SIR Model Example

```python
# Mini example for README
days = np.arange(0, 50)
infected = np.sin(days/10) * 50 + 50
recovered = np.cos(days/15) * 30 + 30
susceptible = 100 - infected - recovered

