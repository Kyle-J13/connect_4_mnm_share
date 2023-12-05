# Kyle Johnston
# Connect 4 torament WOOOOOOOOOOOOOOOOOOOoo
import asyncio
import websockets
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

board = None


def create_board():
    return np.full((6, 7), ' ', dtype=str)

def is_valid_move(board, col):
    return board[0][col] == ' '

def drop_piece(board, col, player):
    for row in range(5, -1, -1):
        if board[row][col] == ' ':
            board[row][col] = player
            break

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

def calculate_move(opponentCol, board):
  if opponentCol != -1:
    drop_piece(board, opponentCol, 'X')
    
  state = flatten_state(board)
  col = select_action(state)
  drop_piece(board, col, 'O')
  return col
  

async def gameloop(socket, created):
    global board  # Declare board as a global variable

    active = True

    while active:
        try:
            message = (await socket.recv()).split(':')
            print("Received message:", message)  # Debugging line

            match message[0]:
                case 'ERROR':
                    board = create_board()
                case 'GAMESTART' | 'OPPONENT':
                    if board is None:
                        board = create_board()
                    if len(message) == 1: 
                        opponentCol = -1
                    else:
                        opponentCol = int(message[1])  # Convert opponentCol to int
                    col = calculate_move(opponentCol, board)
                    print_board(board)
                    
                    await asyncio.sleep(1)
                    await socket.send(f'PLAY:{col}')
                case 'WIN' | 'LOSS' | 'DRAW' | 'TERMINATED':
                    print(message[0])
                    active = False
        except websockets.exceptions.ConnectionClosedOK as e:
            print("Connection closed:", e)
            active = False
        except Exception as e:
            print("Unexpected error:", e)
            active = False


def print_board(board):
    print("  0   1   2   3   4   5   6 ")
    print("|---|---|---|---|---|---|---|")
    for row in board:
        print(f'| {" | ".join(row)} |')
        print("|---|---|---|---|---|---|---|")
    print()


async def create_game (server):
  async with websockets.connect(f'ws://{server}/create') as socket:
    await gameloop(socket, True)

async def join_game(server, id):
  async with websockets.connect(f'ws://{server}/join/{id}') as socket:
    await gameloop(socket, False)

def load_model(model_type):
    q_network.load_state_dict(torch.load(model_type))
    q_network.eval()

if __name__ == '__main__':
  model_type = input("ModelFile: ")
  load_model(model_type)
  server = input('Server IP: ').strip()

  protocol = input('Join game or create game? (j/c): ').strip()

  match protocol:
    case 'c':
      asyncio.run(create_game(server))
    case 'j':
      id = input('Game ID: ').strip()

      asyncio.run(join_game(server, id))
    case _:
      print('Invalid protocol!')