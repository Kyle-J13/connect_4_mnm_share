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

# Initialize Q-networks, optimizers, and replay buffers for each player
device = torch.device("cpu")
q_network_player1 = QNetwork().to(device)
optimizer_player1 = optim.Adam(q_network_player1.parameters(), lr=0.001)

q_network_player2 = QNetwork().to(device)
optimizer_player2 = optim.Adam(q_network_player2.parameters(), lr=0.001)

replay_buffer_player1 = []
replay_buffer_player2 = []

# Q-learning parameters
gamma = 0.99
epsilon = 0.2
batch_size = 32

# Convert board state to a flattened vector
def flatten_state(board):
    encoding = {'X': 1, 'O': -1, ' ': 0}
    numerical_board = [[encoding[cell] for cell in row] for row in board]
    return torch.tensor(numerical_board, dtype=torch.float32).view(-1).to(device)

# Epsilon-greedy action selection
def select_action(state, q_network):
    if np.random.rand() < epsilon:
        return random.choice(range(7))  # Explore
    else:
        with torch.no_grad():
            q_values = q_network(state)
            return torch.argmax(q_values).item()  # Exploit

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

def play_game_with_models(model1, model2):
    board = [[' ' for _ in range(7)] for _ in range(6)]

    for _ in range(42):  # Maximum 21 moves for both players in Connect 4
        # Model 1 move
        state = flatten_state(board)
        action = select_action(state, q_network_player1)
        for row in range(5, -1, -1):
            if board[row][action] == ' ':
                board[row][action] = 'X'
                break

        if is_winner(board, 'X'):
            print_board(board) 
            return 'model1'  # Model 1 wins

        if is_draw(board):
            return 'draw'  # Draw

        # Model 2 move
        state = flatten_state(board)
        action = select_action(state, q_network_player2)
        for row in range(5, -1, -1):
            if board[row][action] == ' ':
                board[row][action] = 'O'
                break


        if is_winner(board, 'O'):
            print_board(board) 
            return 'model2'  # Model 2 wins

        if is_draw(board):
            print_board(board) 
            return 'draw'  # Draw
    
    print_board(board) 

    return 'draw'  # If no winner is determined in 21 moves, it's a draw

def print_board(board):
    print("+-----------------------------+")
    for row in board:
        print("|", end=" ")
        for cell in row:
            print(cell, end=" | ")
        print("\n+-----------------------------+")

# Rest of the code remains unchanged


def main():
    model1_file = input("ModelFile for Model 1(X): ")
    model2_file = input("ModelFile for Model 2(O): ")

    q_network_player1 = QNetwork().to(device)
    q_network_player1.load_state_dict(torch.load(model1_file))
    q_network_player1.eval()
    optimizer_player1 = optim.Adam(q_network_player1.parameters(), lr=0.001)

    q_network_player2 = QNetwork().to(device)
    q_network_player2.load_state_dict(torch.load(model2_file))
    q_network_player2.eval()
    optimizer_player2 = optim.Adam(q_network_player2.parameters(), lr=0.001)

    num_games = int(input("Enter the number of games to play: "))

    results = {'model1': 0, 'model2': 0, 'draw': 0}

    for _ in range(num_games):
        winner = play_game_with_models(q_network_player1, q_network_player2)

        if winner == 'model1':
            results['model1'] += 1
        elif winner == 'model2':
            results['model2'] += 1
        else:
            results['draw'] += 1

    print(f"Results after {num_games} games:")
    print(f"Model 1 wins: {results['model1']}")
    print(f"Model 2 wins: {results['model2']}")
    print(f"Draws: {results['draw']}")

if __name__ == "__main__":
    main()

