# Kyle Johnston
# Connect 4 torament WOOOOOOOOOOOOOOOOOOOoo

import numpy as np
import random

def create_board():
    return np.full((6, 7), ' ', dtype=str)

def print_board(board):
    print("  0   1   2   3   4   5   6 ")
    print("|---|---|---|---|---|---|---|")
    for row in board:
        print(f'| {" | ".join(row)} |')
        print("|---|---|---|---|---|---|---|")
    print()
def is_valid_move(board, col):
    return board[0][col] == ' '

def drop_piece(board, col, player):
    for row in range(5, -1, -1):
        if board[row][col] == ' ':
            board[row][col] = player
            break

def winning_move(board, player):
    # Check horizontally
    for row in range(6):
        for col in range(4):
            if all(board[row][col + i] == player for i in range(4)):
                return True

    # Check vertically
    for row in range(3):
        for col in range(7):
            if all(board[row + i][col] == player for i in range(4)):
                return True

    # Check positively sloped diagonal
    for row in range(3, 6):
        for col in range(4):
            if all(board[row - i][col + i] == player for i in range(4)):
                return True

    # Check negatively sloped diagonal
    for row in range(3):
        for col in range(4):
            if all(board[row + i][col + i] == player for i in range(4)):
                return True

    return False

def is_board_full(board):
    return not any(' ' in row for row in board)

def play_connect4():
    while True:
        print("Connect 4")
        print("1. Play")
        print("2. Quit")

        choice = input("Enter your choice (1 or 2): ")

        if choice == '1':
            game_loop()
        elif choice == '2':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

def game_loop():
    board = create_board()
    current_player = 'X'

    while True:
        print_board(board)

        if current_player == 'X':
            while True:
                try:
                    col = int(input("Player X, enter your move (0-6): "))
                    if 0 <= col <= 6 and is_valid_move(board, col):
                        break
                    else:
                        print("Invalid move. Please enter a number between 0 and 6.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
        else:
            col = get_computer_move(board)

        drop_piece(board, col, current_player)

        if winning_move(board, current_player):
            print_board(board)
            if current_player == 'X':
                print("Player Wins!")
            else:
                print("Computer Wins!")
            break
        elif is_board_full(board):
            print_board(board)
            print("It's a draw!")
            break

        current_player = 'O' if current_player == 'X' else 'X'

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
    
    avoid_list = simulate_moves_and_avoid_player_wins(board, 'O')
    valid_moves = [col for col in range(7) if is_valid_move(board, col) and col not in avoid_list]
    
    #Check for a future two way win
    for col in range(7):
        if is_valid_move(board, col) and col not in avoid_list:
            copy_board = np.copy(board)
            drop_piece(copy_board, col, 'O')

            for next_col in range(7):
                if is_valid_move(copy_board, next_col):
                    next_copy_board = np.copy(copy_board)
                    drop_piece(next_copy_board, next_col, 'X')
                    
                    if check_two_way_win(next_copy_board, 'X'):
                        return next_col
                    
    col_2_future = check_2_future(board, 'O')
    if col_2_future is not None and col_2_future not in avoid_list:
        return col_2_future

    # Check if the move can be strategically placed within 4 tiles from a previous piece
    for col in range(7):
        if is_valid_move(board, col) and col not in avoid_list:
            for row in range(6):
                if board[row][col] == 'O' or board[row][col] == 'X':
                    for i in range(max(0, row - 4), min(6, row + 5)):
                        if i != row and is_valid_move(board, col):
                            return col
    
    if valid_moves:
        return random.choice(valid_moves)

    # If no winning, blocking, or strategic moves, choose a random valid move
    valid_moves = [col for col in range(7) if is_valid_move(board, col)]
    return np.random.choice(valid_moves)


def simulate_moves_and_avoid_player_wins(board, player):
    avoid_list = []
    for col in range(7):
        if is_valid_move(board, col):
            # Simulate placing computer's chip first
            computer_board = np.copy(board)
            drop_piece(computer_board, col, player)

            # Simulate placing player's chip
            for player_col in range(7):
                if is_valid_move(computer_board, player_col):
                    player_board = np.copy(computer_board)
                    drop_piece(player_board, player_col, 'X')

                    # Check if player can win in the next turn
                    if winning_move(player_board, 'X'):
                        avoid_list.append(col)
                        break

    return avoid_list

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


def check_2_future(board, player):
    for col in range(7):
        if is_valid_move(board, col):
            # Check if the move prevents the human player from winning in the next two moves
            copy_board = np.copy(board)
            drop_piece(copy_board, col, player)

            for next_col in range(7):
                if is_valid_move(copy_board, next_col):
                    next_copy_board = np.copy(copy_board)
                    drop_piece(next_copy_board, next_col, 'X')

                    for third_col in range(7):
                        if is_valid_move(next_copy_board, third_col):
                            third_copy_board = np.copy(next_copy_board)
                            drop_piece(third_copy_board, third_col, player)

                            if winning_move(third_copy_board, player):
                                return col

    return None

if __name__ == "__main__":
    play_connect4()
