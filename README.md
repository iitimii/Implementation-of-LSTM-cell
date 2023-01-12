# Implementation-of-LSTM-cell
This is the implementation of the unit LSTM cell

Recurrent Neural Networks (RNN) are very effective for sequence tasks, such as language modelling, named entity recognition, music generation, translation and more. 
They are very effective at processing sequence data because they have some sort of memory.

LSTM (Long Short-Term Memory) networks are a type of recurrent neural network (RNN) that are able to process sequential data more effectively LSTMs are able to process data with long-term dependencies by introducing a "memory" cell and gates that control the flow of information in and out of the cell. This allows LSTMs to selectively keep or forget information from previous time steps, allowing them to better handle long sequences of data. LSTMs are widely used in various deep learning tasks, such as natural language understanding, speech recognition and time series forecasting

LSTMs are trained using optimization algorithms such as Gradient Descent to minimize the error between the predicted output and the true output, thus allowing the network to learn the optimal values of the parameters.

The gates in an LSTM network are used to control the flow of information into and out of the memory cell. There are three types of gates in an LSTM: the input gate, forget gate, and output gate.

The input gate, represented by the equation:
i(t) = sigmoid(W_i * [h(t-1), x(t)] + b_i)
controls the amount of information that is allowed to enter the memory cell. The input gate is a sigmoid function that takes in the previous hidden state, h(t-1), and the current input, x(t), and produces a value between 0 and 1.

The forget gate, represented by the equation:
f(t) = sigmoid(W_f * [h(t-1), x(t)] + b_f)
controls the amount of information that is allowed to be forgotten from the memory cell. Like the input gate, the forget gate is also a sigmoid function, but it produces a value between 0 and 1.

The output gate, represented by the equation:
o(t) = sigmoid(W_o * [h(t-1), x(t)] + b_o)
controls the amount of information that is allowed to be output from the memory cell. Like the input and forget gates, the output gate is also a sigmoid function.

Where W_i, W_f, W_o and b_i, b_f, b_o are the weights and biases for the input, forget, and output gates respectively.
h, x, t are the hidden states,  inputs and timesteps respectively.

The final state of the cell is given by :
c(t) = f(t) * c(t-1) + i(t) * tanh(W_c * [h(t-1), x(t)] + b_c)

Where W_c, b_c are the weights and biases for the cell state, and tanh is the hyperbolic tangent function.

The hidden state is given by :
h(t) = o(t) * tanh(c(t))

Where h(t) is the output of the lstm unit at time t.

In summary, LSTM gates allow to selectively keep or forget information from previous time steps, allowing them to better handle long sequences of data.
