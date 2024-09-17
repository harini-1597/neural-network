## Kaggle Notebooks
- [NumPy & Math](https://www.kaggle.com/code/whatharini/neural-network-with-math)
- [TensorFlow](https://www.kaggle.com/code/whatharini/neural-network-with-tensorflow)
  
# Neural Networks
- How does the brain work? Recognize a 3 as 3. It recognizes patterns, like a loop under a loop is a 3.
- The neuron of, say a digit image, is 28x28 of the pictures dimensions and each node is lit up acc to the greyscale value of it. The dataset is composed of such pixel values.
    - 28x28 = 784 neurons is the first layer and the last layer is 10 neurons of digits
    - Hidden layers also exist. These add to the complexity of the code. This is also, say, 10 digits.
- Activation of one layer brings about activation in another layer
- Activation function - sigmoid function. Activate only when weighted sum above a certain number. 
- Bias - basically only be activated above a certain number.  x-10 (-10 is bias) x if only 11 or sth will be activated

Learning nasically involves finding the right weight and bias, by constant propogation.

