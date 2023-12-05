#Kyle Johnston
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
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

# Experience Replay Buffer
class ReplayBuffer:
    def init(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
        )

# Target Network
class TargetQNetwork(nn.Module):
    def __init__(self):
        super(TargetQNetwork, self).__init__()
        self.fc1 = nn.Linear(42, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Convert board state to a flattened vector
def flatten_state(board):
    encoding = {'X': 1, 'O': -1, ' ': 0}
    numerical_board = [[encoding[cell] for cell in row] for row in board]
    return torch.tensor(numerical_board, dtype=torch.float32).view(-1)

# Initialize Q-network and Target Q-network
device = torch.device("cpu")
q_network = QNetwork().to(device)
target_network = TargetQNetwork().to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

# Initialize Replay Buffer
replay_buffer = ReplayBuffer()
replay_buffer.init(capacity=10000)

# Initialize optimizer
optimizer = optim.Adam(q_network.parameters(), lr=0.0001)

# Q-learning parameters
gamma = 0.99
epsilon = 1.0  # Initial epsilon value
min_epsilon = 0.1
epsilon_decay = 0.995
batch_size = 32

# Other parameters
target_update_frequency = 10  # Update target network every 10 episodes

# Epsilon-greedy action selection
def select_action(state, epsilon, current_board):
    if np.random.rand() < epsilon:
        return random.choice(range(7))  # Explore
    else:
        with torch.no_grad():
            q_values = q_network(state)
            # Filter out actions for full columns
            valid_actions = [a for a in range(7) if current_board[0][a] == ' ']
            return torch.tensor(valid_actions)[torch.argmax(q_values[valid_actions])].item()

# Q-learning update
def q_learning_update(state, action, reward, next_state):
    replay_buffer.add_experience((state, action, reward, next_state))

    if len(replay_buffer.buffer) < batch_size:
        return  # Wait until enough experiences are in the replay buffer

    states, actions, rewards, next_states = replay_buffer.sample_batch(batch_size)

    state_action_values = q_network(states).gather(1, actions.view(-1, 1))
    next_state_values = target_network(next_states).max(1).values.detach()
    target = rewards + gamma * next_state_values

    loss = nn.MSELoss()(state_action_values.squeeze(), target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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

# Save Q-network parameters to a file
def save_model(save_file):
    torch.save(q_network.state_dict(), save_file)

# Main training loop
def train_main():
    epsilon = 1.0 
    min_epsilon = 0.1
    epsilon_decay = 0.99
    
    num_episodes = int(input("Enter number of episodes: "))
    save_file = input("Enter save_file.pth: ")

    logic_wins = 0
    model_wins = 0

    for episode in tqdm(range(num_episodes), desc="Training", unit="episode"):
        board = [[' ' for _ in range(7)] for _ in range(6)]
        
        for _ in range(21):  # Reduced to 21 to ensure both players make a move in each episode
            # Opponent move (random)
            action = get_computer_move(board)

            for row in range(5, -1, -1):
                if board[row][action] == ' ':
                    board[row][action] = 'O'
                    break

            if is_winner(board, 'O') or is_draw(board):
                logic_wins += 1
                break

            # Player move
            state = flatten_state(board)
            action = select_action(state, epsilon, board)

            for row in range(5, -1, -1):
                if board[row][action] == ' ':
                    board[row][action] = 'X'
                    break

            if is_winner(board, 'X') or is_draw(board):
                model_wins += 1
                break

            # Update Q-values
            next_state = flatten_state(board)
            reward = 1 if is_winner(board, 'X') else -1  # Reward for player's move
            q_learning_update(state, action, reward, next_state)

        if episode % target_update_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Print wins every 1000 episodes
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}: Logic Wins = {logic_wins}, Model Wins = {model_wins}")

    save_model(save_file)
    print("Training complete.")
    
    

def is_valid_move(board, col):
    return board[0][col] == ' '

def drop_piece(board, col, player):
    for row in range(5, -1, -1):
        if board[row][col] == ' ':
            board[row][col] = player
            break

def winning_move(board, player):
    
    # Check horizontal
    for row in range(6):
        for col in range(4):
            if board[row][col] == player and board[row][col + 1] == player and \
               board[row][col + 2] == player and board[row][col + 3] == player:
                return True

    # Check vertical
    for row in range(3):
        for col in range(7):
            if board[row][col] == player and board[row + 1][col] == player and \
               board[row + 2][col] == player and board[row + 3][col] == player:
                return True

    # Check positively sloped diagonal
    for row in range(3):
        for col in range(4):
            if board[row][col] == player and board[row + 1][col + 1] == player and \
               board[row + 2][col + 2] == player and board[row + 3][col + 3] == player:
                return True

    # Check negatively sloped diagonal
    for row in range(3):
        for col in range(3, 7):
            if board[row][col] == player and board[row + 1][col - 1] == player and \
               board[row + 2][col - 2] == player and board[row + 3][col - 3] == player:
                return True

    return False

def is_board_full(board):
    return not any(' ' in row for row in board)

def get_computer_move(board):
    for col in range(7):
        if is_valid_move(board, col):
            # Check if the move creates a winning move for the computer
            copy_board = np.copy(board)
            drop_piece(copy_board, col, 'O')
            if winning_move(copy_board, 'O'):
                return col

    for col in range(7):
        if is_valid_move(board, col):
            # Check if the move prevents the human player from winning
            copy_board = np.copy(board)
            drop_piece(copy_board, col, 'X')
            if winning_move(copy_board, 'X'):
                return col
    
    #Check for a future two way win
    for col in range(7):
        if is_valid_move(board, col):
            copy_board = np.copy(board)
            drop_piece(copy_board, col, 'O')

            for next_col in range(7):
                if is_valid_move(copy_board, next_col):
                    next_copy_board = np.copy(copy_board)
                    drop_piece(next_copy_board, next_col, 'X')
                    
                    if check_two_way_win(next_copy_board, 'X'):
                        return next_col

    # Check if the move can be strategically placed within 4 tiles from a previous piece
    for col in range(7):
        if is_valid_move(board, col):
            for row in range(6):
                if board[row][col] == 'O' or board[row][col] == 'X':
                    for i in range(max(0, row - 4), min(6, row + 5)):
                        if i != row and is_valid_move(board, col):
                            return col

    # If no winning, blocking, or strategic moves, choose a random valid move
    valid_moves = [col for col in range(7) if is_valid_move(board, col)]
    return np.random.choice(valid_moves)


def check_two_way_win(board, player):
    winning_moves_count = 0

    for col in range(7):
        if is_valid_move(board, col):
            copy_board = np.copy(board)
            drop_piece(copy_board, col, player)

            # Check if the move creates winning moves for the player
            if winning_move(copy_board, player):
                winning_moves_count += 1

            if winning_moves_count >= 2:
                return True

    return False

if __name__ == "__main__":
    train_main()
