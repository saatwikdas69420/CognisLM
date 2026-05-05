import random
import numpy as np

# Vocabulary and word encoding
words = [
    "hello", "how", "are", "you", "doing", "i", "am", "fine", "thanks", "bye", 
    "good", "nice", "see", "world", "love", "and",
    "happy", "sad", "excited", "bored", "hungry", "tired", "friend", "enemy", "peace",
    "war", "life", "death", "hope", "fear", "truth", "false", "light", "dark",
    "win", "lose", "strong", "weak", "fast", "slow", "high", "low", "bad", "cycle",
    "feelings", "emotions", "speed", "power", "altitude", "happiness", "winning", "losing", "control", "wins", "sadness", "victory", "failure", "strength", "darkness", "destruction", "creation", "end"
]

vocab_size = len(words)
word_to_idx = {word: idx for idx, word in enumerate(words)}
idx_to_word = {idx: word for idx, word in enumerate(words)}

# Function to encode a sentence into a list of word indices
def encode(sentence):
    return [word_to_idx[word] for word in sentence.split() if word in word_to_idx]

# Expanded training data
training_data = [
    ("hello how", "are"),
    ("how are", "you"),
    ("are you", "doing"),
    ("you doing", "i"),
    ("doing i", "am"),
    ("i am", "fine"),
    ("am fine", "thanks"),
    ("fine thanks", "bye"),
    ("thanks bye", "hello"),
    ("hello good", "nice"),
    ("good nice", "see"),
    ("nice see", "world"),
    ("see world", "love"),
    ("world love", "you"),
    ("love you", "doing"),
    ("i love", "you"),
    ("you love", "world"),
    ("i love see", "world"),
    ("i doing", "good"),
    ("how are you", "i"),
    ("i am", "fine"),
    ("you are", "happy"),
    ("i am", "tired"),
    ("world is", "dark"),
    ("light and", "hope"),
    ("truth is", "strong"),
    ("i am", "hungry"),
    ("fast and", "strong"),
    ("peace and", "war"),
    ("enemy and", "friend"),
    ("high and", "low"),
    ("win and", "lose"),
    ("life and", "death"),
    ("you are", "excited"),
    ("bored and", "tired"),
    ("fear and", "hope"),
    ("hello friend", "how"),
    ("you are friend", "good"),
    ("friend and enemy", "peace"),
    ("hope and", "life"),
    ("i am", "hungry"),
    ("fast is", "good"),
    ("slow is", "bad"),
    ("bad and", "good"),
    ("strong is", "nice"),
    ("fear is", "false"),
    ("truth and", "false"),
    ("light is", "high"),
    ("dark is", "low"),
    ("high and low", "win"),
    ("win is", "strong"),
    ("lose is", "weak"),
    ("strong is", "good"),
    ("weak is", "bad"),
    ("you win", "life"),
    ("you lose", "death"),
    ("peace and war", "truth"),
    ("truth is", "good"),
    ("fear is", "bad"),
    ("hunger and", "fear"),
    ("fear of", "death"),
    ("life is", "hope"),
    ("hope and", "peace"),
    ("war and", "death"),
    ("love and", "life"),
    ("dark and", "fear"),
    ("light and", "hope"),
    ("fast is better", "slow"),
    ("slow is worse", "fast"),
    ("strong and fast", "win"),
    ("weak and slow", "lose"),
    ("winning is", "good"),
    ("losing is", "bad"),
    ("good and nice", "happy"),
    ("bad and sad", "fear"),
    ("love is", "nice"),
    ("love and peace", "good"),
    ("hate and war", "bad"),
    ("enemy is", "bad"),
    ("friend is", "good"),
    ("life is", "good"),
    ("death is", "bad"),
    ("death and life", "cycle"),
    ("happy and sad", "feelings"),
    ("hope and fear", "emotions"),
    ("fast and slow", "speed"),
    ("strong and weak", "power"),
    ("high and low", "altitude"),
    ("you are strong", "win"),
    ("you are weak", "lose"),
    ("peace leads", "hope"),
    ("war leads", "death"),
    ("friendship brings", "happiness"),
    ("enemy brings", "fear"),
    ("life is about", "winning"),
    ("death is about", "losing"),
    ("light conquers", "dark"),
    ("fear can", "control"),
    ("truth always", "wins"),
    ("false leads", "fear"),
    ("win and peace", "happiness"),
    ("lose and war", "sadness"),
    ("fast and power", "victory"),
    ("slow and weak", "failure"),
    ("love creates", "hope"),
    ("hate creates", "fear"),
    ("happiness is", "good"),
    ("sadness is", "bad"),
    ("speed and power", "victory"),
    ("altitude affects", "strength"),
    ("hope brings", "light"),
    ("fear brings", "darkness"),
    ("win means", "strong"),
    ("lose means", "weak"),
    ("truth brings", "hope"),
    ("false brings", "fear"),
    ("life continues", "cycle"),
    ("dark and war", "destruction"),
    ("peace and light", "creation"),
]

# Neural Network Class (no external libraries)
class SimpleLanguageModel:
    def __init__(self, vocab_size, hidden_size, context_size=3):
        self.input_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.context_size = context_size  # Memory window size

        # Initialize weights and biases (randomly)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01  # Input to hidden weights
        self.b1 = np.zeros(self.hidden_size)  # Hidden biases
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01  # Hidden to output weights
        self.b2 = np.zeros(self.output_size)  # Output biases

    # Forward pass
    def forward(self, X):
        self.hidden = np.dot(X, self.W1) + self.b1  # Linear transformation
        self.hidden = np.tanh(self.hidden)  # Activation function (tanh)
        
        self.output = np.dot(self.hidden, self.W2) + self.b2  # Linear transformation to output
        self.output = self.softmax(self.output)  # Softmax activation for probabilities
        
        return self.output

    # Softmax function to normalize the output into probabilities
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)

    # Backward pass (training with gradient descent)
    def backward(self, X, Y, learning_rate=0.01):
        output_loss = self.output - Y  # Calculate loss derivative
        dW2 = np.outer(self.hidden, output_loss)  # Derivative w.r.t W2
        db2 = output_loss  # Derivative w.r.t b2

        hidden_loss = np.dot(self.W2, output_loss) * (1 - np.tanh(self.hidden) ** 2)  # Backpropagate to hidden layer
        dW1 = np.outer(X, hidden_loss)  # Derivative w.r.t W1
        db1 = hidden_loss  # Derivative w.r.t b1

        # Update weights and biases using gradient descent
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    # Train the model
    def train(self, training_data, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0
            for input_sentence, target_word in training_data:
                X = np.zeros(self.input_size)  # Input vector (one-hot encoded)
                for word in input_sentence.split():
                    if word in word_to_idx:  # Avoid words not in vocabulary
                        X[word_to_idx[word]] = 1  # One-hot encoding for each word

                Y = np.zeros(self.output_size)  # Target vector (one-hot encoded)
                Y[word_to_idx[target_word]] = 1  # Target word (one-hot)

                output = self.forward(X)  # Forward pass
                loss = -np.log(output[word_to_idx[target_word]])  # Cross-entropy loss
                total_loss += loss

                self.backward(X, Y, learning_rate)  # Backward pass (gradient descent)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss}")

    # Generate the next word (interactive and dynamic)
    def predict(self, input_sentence, max_length=3, temperature=1.0):
        context = input_sentence.split()  # Start with the user's input
        generated_text = input_sentence
        history_set = set(context)  # Keep track of previously used words

        for _ in range(max_length):  # Limit the number of words in the output
            X = np.zeros(self.input_size)
            for word in context:
                if word in word_to_idx:
                    X[word_to_idx[word]] = 1

            output = self.forward(X)

            # Temperature scaling (controls randomness in predictions)
            output = np.log(output) / temperature
            exp_output = np.exp(output - np.max(output))  # Subtract max for numerical stability
            output = exp_output / np.sum(exp_output)

            # Sample word based on the probability distribution
            predicted_idx = np.random.choice(np.arange(self.output_size), p=output)
            next_word = idx_to_word[predicted_idx]

            # Ensure no immediate repetition
            attempts = 0
            while next_word in history_set and attempts < 5:
                predicted_idx = np.random.choice(np.arange(self.output_size), p=output)
                next_word = idx_to_word[predicted_idx]
                attempts += 1

            # Stop if the predicted word is an end-of-sentence marker or some predefined stop word
            if next_word == "end":  # You can use a token like "end" to stop
                break

            generated_text += " " + next_word
            context.append(next_word)  # Add the predicted word to the context
            history_set.add(next_word)  # Add to history to avoid repeats
            if len(context) > self.context_size:
                context = context[1:]  # Limit context size based on context_size

        return generated_text


# Initialize and Train the Model
hidden_size = 10  # A small hidden layer size
context_size = 5  # Increase context size for more coherent responses
model = SimpleLanguageModel(vocab_size, hidden_size, context_size)

# Train the model on the training data
model.train(training_data, epochs=1000, learning_rate=0.01)

# Generate text based on user input
print("\nText generation:")
context = []
while True:
    input_sentence = input("Enter a sentence (or type 'exit' to stop): ")
    if input_sentence == "exit":
        break

    # Add the user's input to the context
    context.extend(input_sentence.split())
    
    # Limit context size based on context_size
    if len(context) > context_size:
        context = context[-context_size:]

    # Start with the user's input and generate the next word(s)
    generated_text = input_sentence
    next_word = model.predict(" ".join(context), max_length=2, temperature=1.0)  # Generate 2 words at a time
    print(f"Generated: {next_word}")


