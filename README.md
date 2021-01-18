# LightNeuNet
A light-weight neural network framework built for academic use.

## Credits
LightNeuNet borrowed syntaxes from Keras.

## An example for Artificial Neural Network (ANN)
```python
from lnn.model import SequentialModel
from lnn.layers import Input, Dense, Output

# Initialize a fully connected neural network
model = SequentialModel()

# Add the input layer
model.add(Input(32))

# Add the first and second hidden layer
model.add(Dense(18, activation='tanh', 
    kernel_initializer='uniform', use_bias = True))
model.add(Dense(18, activation='tanh', 
    kernel_initializer = 'uniform', use_bias = True))

# Add the output layer
model.add(Output(num_output_units, activation = 'softmax', 
    use_bias = True))

# Compile the model
model.compile()

# The summary of the model built
model.summary()

# Train the model
for i in range(5000):
    model.fit(X_train, y_train, learning_rate = 0.01)
    
y_pred = model.predict(X_test)
```

## An example for Neural Networks optimized with Genetic Algorithm
```python
from lnn.genetic_algorithm.layers import GAInput, GADense, GAOutput
from lnn.genetic_algorithm.model import GAModel, GASequentialModel

# Create the model
model = GAModel(population=500)

# Add input layer to the model
model.add(GAInput(32))

# Add the first hidden layer to the model
model.add(GADense(24, activation='tanh', 
    use_bias=True, kernel_initializer='uniform'))

# Add the second hidden layer to the model
model.add(GADense(18, activation='tanh', 
    use_bias=True, kernel_initializer='uniform'))

# Add the output layer to the model
model.add(GAOutput(4, activation='softmax', use_bias=False))

# Generate the population
model.new_population()

def play_snake(model, params=(...)):
    # Code to play the snake
    ...

iters = 10000
for i in range(iters):
    model.simulate(play_snake, keep_rate=0.6, 
        mutate_rate=0.01, params=(False,))
```
