import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(42, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize Q-network
device = torch.device("cpu")
q_network = QNetwork().to(device)

# Convert board state to a flattened vector
def flatten_state(board):
    encoding = {'X': 1, 'O': -1, ' ': 0}
    numerical_board = [[encoding[cell] for cell in row] for row in board]
    return torch.tensor(numerical_board, dtype=torch.float32).view(-1).to(device)

# Epsilon-greedy action selection
def select_action(state):
    with torch.no_grad():
        q_values = q_network(state)
        return torch.argmax(q_values).item()

# Check for a winner in the Connect 4 board
def is_winner(board, player):
    for i in range(6):
        for j in range(7):
            if (
                j + 3 < 7 and all(board[i][j + k] == player for k in range(4))
                or i + 3 < 6 and all(board[i + k][j] == player for k in range(4))
                or i + 3 < 6 and j + 3 < 7 and all(board[i + k][j + k] == player for k in range(4))
                or i - 3 >= 0 and j + 3 < 7 and all(board[i - k][j + k] == player for k in range(4))
            ):
                return True
    return False

# Check for a draw in the Connect 4 board
def is_draw(board):
    return all(board[i][j] != ' ' for i in range(6) for j in range(7))

# Play the game against the computer
def play_game():
    epsilon = 0.1  # Fixed epsilon for exploration during gameplay
    board = [[' ' for _ in range(7)] for _ in range(6)]

    while True:  # Add a loop for multiple games
        for _ in range(42):  # Maximum 42 moves in Connect 4
            # Player move
            print_board(board)
            col = int(input("Enter the column (0 to 6): "))
            for row in range(5, -1, -1):
                if board[row][col] == ' ':
                    board[row][col] = 'X'
                    break

            if is_winner(board, 'X'):
                print_board(board)
                print("You win!")
                break

            if is_draw(board):
                print_board(board)
                print("It's a draw!")
                break

            # Computer move
            state = flatten_state(board)
            action = select_action(state)
            for row in range(5, -1, -1):
                if board[row][action] == ' ':
                    board[row][action] = 'O'
                    break

            if is_winner(board, 'O'):
                print_board(board)
                print("Computer wins!")
                break

            if is_draw(board):
                print_board(board)
                print("It's a draw!")
                break

        print_board(board)
        option = input("Choose an option:\n1. Play Again\n2. Change Model\n3. Exit\nEnter the option (1, 2, or 3): ")

        if option == '1':
            board = [[' ' for _ in range(7)] for _ in range(6)]
        elif option == '2':
            model_type = input("New ModelFile: ")
            load_model(model_type)
            board = [[' ' for _ in range(7)] for _ in range(6)]
        elif option == '3':
            break
        else:
            print("Invalid option. Please enter 1, 2, or 3.")
# Load Q-network parameters from a file
def load_model(model_type):
    q_network.load_state_dict(torch.load(model_type))
    q_network.eval()

# Print the Connect 4 board
def print_board(board):
    print("+-----------------------------+")
    for row in board:
        print("|", end=" ")
        for cell in row:
            print(cell, end=" | ")
        print("\n+-----------------------------+")

if __name__ == "__main__":
    model_type = input("ModelFile: ")
    load_model(model_type)
    play_game()
