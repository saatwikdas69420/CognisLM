import numpy as np

# ===============================
# Training Data
# ===============================
training_data = [

# Greetings (expanded)
("hello", "hi"),
("hi", "hello"),
("hey", "hello"),
("good morning", "morning"),
("good evening", "evening"),
("how are you", "i am fine"),
("how is it going", "it is going well"),
("nice to meet you", "nice to meet you too"),
("what is up", "not much"),
("are you there", "yes i am here"),

# Identity & Intent
("who are you", "i am an ai"),
("what are you", "i am a program"),
("what can you do", "i can respond"),
("are you real", "i am virtual"),
("why are you here", "to help"),
("what is your goal", "to assist"),
("can you think", "i simulate thinking"),
("do you learn", "i improve with data"),

# Feelings & States
("i am happy", "that is good"),
("i am sad", "that is bad"),
("i am tired", "you need rest"),
("i am hungry", "you need food"),
("i feel anxious", "try to relax"),
("i feel excited", "that is nice"),
("i feel angry", "calm down"),
("i am bored", "try something fun"),
("i feel confused", "let us think"),
("i feel great", "that is good"),

# Causal Reasoning
("i did not sleep", "you feel tired"),
("i did not eat", "you feel hungry"),
("it is raining", "ground gets wet"),
("i study hard", "i learn more"),
("i skip practice", "i get worse"),
("i run daily", "i get faster"),
("i eat too much", "i feel sick"),
("i save money", "i have more later"),
("i waste time", "i lose progress"),
("i train more", "i improve"),

# Conditional Logic
("if i study", "i pass"),
("if i sleep well", "i feel good"),
("if it rains", "i stay inside"),
("if i train", "i get strong"),
("if i eat healthy", "i feel better"),
("if i practice", "i improve"),
("if i try", "i succeed"),
("if i quit", "i fail"),
("if i focus", "i finish"),
("if i wait", "time passes"),

# Comparisons
("which is bigger cat or dog", "dog is bigger"),
("which is faster car or bike", "car is faster"),
("which is stronger man or child", "man is stronger"),
("which is heavier rock or feather", "rock is heavier"),
("which is smarter human or animal", "human is smarter"),
("which is hotter fire or ice", "fire is hotter"),
("which is colder ice or water", "ice is colder"),
("which is taller tree or grass", "tree is taller"),
("which is brighter sun or moon", "sun is brighter"),
("which is louder thunder or whisper", "thunder is louder"),

# Temporal Reasoning
("what comes after day", "night"),
("what comes after night", "day"),
("what comes before evening", "afternoon"),
("what comes before morning", "night"),
("i wake up", "then i eat"),
("i eat", "then i work"),
("i work", "then i rest"),
("i rest", "then i sleep"),
("today is monday", "tomorrow is tuesday"),
("today is friday", "tomorrow is saturday"),

# Goal-Oriented Reasoning
("i want to be strong", "train more"),
("i want to be smart", "study more"),
("i want to be fast", "practice running"),
("i want to win", "work hard"),
("i want to learn", "read books"),
("i want money", "work"),
("i want success", "stay consistent"),
("i want peace", "avoid conflict"),
("i want skills", "practice"),
("i want results", "take action"),

# Analogies
("bird is to sky as fish is to", "water"),
("car is to road as train is to", "track"),
("pen is to write as knife is to", "cut"),
("eye is to see as ear is to", "hear"),
("fire is to hot as ice is to", "cold"),
("sun is to day as moon is to", "night"),
("teacher is to teach as student is to", "learn"),
("doctor is to health as mechanic is to", "repair"),
("book is to read as music is to", "listen"),
("key is to lock as password is to", "access"),

# Math / Logic Patterns
("one plus one", "two"),
("two plus two", "four"),
("three plus one", "four"),
("five minus two", "three"),
("ten minus five", "five"),
("two times two", "four"),
("three times three", "nine"),
("ten divided by two", "five"),
("zero plus one", "one"),
("one minus one", "zero"),

# Contradictions
("i am full and hungry", "that is conflicting"),
("i am asleep and awake", "that is impossible"),
("it is hot and cold", "that is inconsistent"),
("i am winning and losing", "that is contradictory"),
("i am fast and slow", "that is conflicting"),
("i am happy and sad", "that is mixed"),
("light and dark together", "that is contrast"),
("true and false together", "that is contradiction"),
("yes and no", "that is conflict"),
("alive and dead", "that is impossible"),

# Basic Knowledge
("what is water", "water is liquid"),
("what is fire", "fire is hot"),
("what is ice", "ice is cold"),
("what is sun", "sun is star"),
("what is moon", "moon is satellite"),
("what is earth", "earth is planet"),
("what is air", "air is gas"),
("what is food", "food gives energy"),
("what is sleep", "sleep gives rest"),
("what is work", "work produces value"),

# Social Interaction
("thank you", "you are welcome"),
("thanks", "no problem"),
("sorry", "it is okay"),
("excuse me", "yes"),
("can you help", "yes i can"),
("please help me", "i will try"),
("i need help", "what do you need"),
("can we talk", "yes"),
("are you listening", "yes"),
("do you understand", "i understand"),

# Decision / Preference
("should i study or sleep", "sleep then study"),
("should i eat or work", "eat first"),
("should i run or rest", "rest if tired"),
("should i save or spend", "save money"),
("should i try or quit", "try"),
("should i go out or stay", "it depends"),
("should i learn or ignore", "learn"),
("should i act or wait", "act wisely"),
("should i speak or stay quiet", "think first"),
("should i help or ignore", "help others"),

# Ending
("bye", "goodbye"),
("goodbye", "bye"),
("see you", "see you later"),
("talk later", "okay"),
("exit", "goodbye"),
]

# ===============================
# Build Vocabulary
# ===============================
word_to_idx = {"<UNK>": 0, "<END>": 1}
idx_to_word = {0: "<UNK>", 1: "<END>"}

def add_word(w):
    if w not in word_to_idx:
        idx = len(word_to_idx)
        word_to_idx[w] = idx
        idx_to_word[idx] = w

for inp, out in training_data:
    for w in inp.split():
        add_word(w)
    for w in out.split():
        add_word(w)

vocab_size = len(word_to_idx)

# ===============================
# Model
# ===============================
class ReasoningLanguageModel:
    def __init__(self, vocab_size, hidden_size=64, embedding_dim=32, context_size=5):
        self.vocab_size = vocab_size
        self.embedding = np.random.randn(vocab_size, embedding_dim) * 0.01

        self.W1 = np.random.randn(embedding_dim, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.randn(hidden_size, vocab_size) * 0.01
        self.b2 = np.zeros(vocab_size)

        self.context_size = context_size

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def encode(self, sentence):
        return [word_to_idx.get(w, 0) for w in sentence.split()]

    def forward(self, word_indices):
        embedded = self.embedding[word_indices]
        X = np.mean(embedded, axis=0)

        self.hidden = np.tanh(np.dot(X, self.W1) + self.b1)
        self.output = self.softmax(np.dot(self.hidden, self.W2) + self.b2)

        return self.output, X

    def backward(self, X, word_indices, Y, lr):
        error = self.output - Y

        dW2 = np.outer(self.hidden, error)
        db2 = error

        hidden_error = np.dot(self.W2, error) * (1 - self.hidden**2)

        dW1 = np.outer(X, hidden_error)
        db1 = hidden_error

        # Update weights
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        # Update embeddings
        grad_X = np.dot(self.W1, hidden_error)
        for idx in word_indices:
            self.embedding[idx] -= lr * grad_X / len(word_indices)

    # ===============================
    # Full Sequence Training
    # ===============================
    def train(self, data, epochs=1500, lr=0.05):
        for epoch in range(epochs):
            total_loss = 0

            for inp, out in data:
                input_tokens = inp.split()
                target_tokens = out.split() + ["<END>"]

                context = input_tokens.copy()

                for target_word in target_tokens:
                    word_indices = [
                        word_to_idx.get(w, 0) for w in context
                    ]

                    Y = np.zeros(self.vocab_size)
                    Y[word_to_idx[target_word]] = 1

                    output, X = self.forward(word_indices)

                    loss = -np.log(output[word_to_idx[target_word]] + 1e-9)
                    total_loss += loss

                    self.backward(X, word_indices, Y, lr)

                    # Teacher forcing
                    context.append(target_word)
                    context = context[-self.context_size:]

            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    # ===============================
    # Improved Generation
    # ===============================
    def predict(self, input_sentence, max_length=6, temperature=0.7):
        context = input_sentence.split()
        result = []

        for _ in range(max_length):
            word_indices = [
                word_to_idx.get(w, 0) for w in context
            ]

            if not word_indices:
                break

            output, _ = self.forward(word_indices)

            # Temperature
            probs = np.log(output + 1e-9) / temperature
            probs = np.exp(probs - np.max(probs))
            probs /= np.sum(probs)

            # Repetition penalty
            for i, w in idx_to_word.items():
                if w in result:
                    probs[i] *= 0.5

            probs /= np.sum(probs)

            next_idx = np.random.choice(np.arange(self.vocab_size), p=probs)
            next_word = idx_to_word[next_idx]

            if next_word == "<END>":
                break

            result.append(next_word)

            # controlled context growth
            context.append(next_word)
            context = context[-self.context_size:]

        return " ".join(result)

# ===============================
# Train
# ===============================
model = ReasoningLanguageModel(vocab_size)
model.train(training_data, epochs=1500, lr=0.05)

# ===============================
# Chat
# ===============================
print("\nChat with AI (type 'exit' to quit):")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = model.predict(user_input)
    print("AI:", response)
