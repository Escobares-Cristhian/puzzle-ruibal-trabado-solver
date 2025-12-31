import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import numba as nb

# ----- Configuration params -----
# "Trabado" puzzle solver: Ruibal version
initial_board = np.array(
    [
        [1, 2, 2, 3],
        [1, 2, 2, 3],
        [4, 5, 5, 6],
        [4, 7, 8, 6],
        [9, 0, 0, 10],
    ],
    dtype=np.int64,
)

# Max depth for safety
MAX_DEPTH = 1000

# ----- Functions -----
@nb.njit(cache=True)
def is_solved(board: np.ndarray) -> bool:
    # Goal: piece "2" must occupy positions (4,1) and (4,2)
    return board[4, 1] == 2 and board[4, 2] == 2


def plot_board(board_in: np.ndarray) -> None:
    board = board_in.copy().astype(float)
    board[board == 0] = np.nan

    plt.imshow(board, cmap="tab10")
    # Draw piece id numbers
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            if np.isnan(board[r, c]):
                continue
            plt.text(c, r, int(board[r, c]), ha="center", va="center", color="w")
    plt.show()


def get_pieces_adjacent_to_empty(board: np.ndarray) -> list[int]:
    """
    Return piece ids that are orthogonally adjacent to at least one empty cell (value 0).
    """
    empty_positions = np.argwhere(board == 0)
    adjacent_pieces: set[int] = set()

    for r, c in empty_positions:
        # Up / Down neighbors
        if r - 1 >= 0 and board[r - 1, c] != 0:
            adjacent_pieces.add(int(board[r - 1, c]))
        if r + 1 < board.shape[0] and board[r + 1, c] != 0:
            adjacent_pieces.add(int(board[r + 1, c]))

        # Left / Right neighbors
        if c - 1 >= 0 and board[r, c - 1] != 0:
            adjacent_pieces.add(int(board[r, c - 1]))
        if c + 1 < board.shape[1] and board[r, c + 1] != 0:
            adjacent_pieces.add(int(board[r, c + 1]))

    return list(adjacent_pieces)


@nb.njit(cache=True)
def move_piece(board: np.ndarray, piece: int, direction: int):
    """
    Move a piece by 1 cell if valid.

    direction codes:
      0 = up    (row - 1)
      1 = down  (row + 1)
      2 = left  (col - 1)
      3 = right (col + 1)

    Returns:
      (valid_move: bool, new_board: np.ndarray)
    """
    piece_rows, piece_cols = np.where(board == piece)

    # Compute target coordinates
    if direction == 0:  # up
        new_rows = piece_rows - 1
        new_cols = piece_cols
    elif direction == 1:  # down
        new_rows = piece_rows + 1
        new_cols = piece_cols
    elif direction == 2:  # left
        new_rows = piece_rows
        new_cols = piece_cols - 1
    else:  # 3 = right
        new_rows = piece_rows
        new_cols = piece_cols + 1

    # Validate bounds and collisions
    for i in range(len(new_rows)):
        nr = new_rows[i]
        nc = new_cols[i]
        if nr < 0 or nr >= board.shape[0] or nc < 0 or nc >= board.shape[1]:
            return False, board
        cell = board[nr, nc]
        if cell != 0 and cell != piece:
            return False, board

    # Clear current piece cells
    for i in range(len(piece_rows)):
        board[piece_rows[i], piece_cols[i]] = 0

    # Place piece in new cells
    for i in range(len(new_rows)):
        board[new_rows[i], new_cols[i]] = piece

    return True, board


def board_key(board: np.ndarray) -> bytes:
    # Fast and hashable representation (board shape is fixed)
    return board.tobytes()


def expand_unique_frontier(frontier_boards: list[np.ndarray], frontier_paths: list[str]):
    """
    Expand one BFS layer, keeping only unique boards within this expansion step.
    """
    next_boards: list[np.ndarray] = []
    next_paths: list[str] = []

    unique_next: set[bytes] = set()

    for i, board in enumerate(frontier_boards):
        path_so_far = frontier_paths[i]
        for piece in get_pieces_adjacent_to_empty(board):
            for direction_code, direction_char in enumerate(["u", "d", "l", "r"]):
                valid, moved = move_piece(board.copy(), piece, direction_code)
                if not valid:
                    continue

                k = board_key(moved)
                if k in unique_next:
                    continue

                unique_next.add(k)
                next_boards.append(moved.copy())
                next_paths.append(f"{path_so_far} {piece}{direction_char}".strip())

    return next_boards, next_paths


# ----- Main solver loop -----
if __name__ == "__main__":
    # Initial board plot
    plot_board(initial_board)
    
    # ----- Initialize BFS frontier (depth 1) -----
    visited_boards: list[np.ndarray] = [initial_board.copy()]
    visited_keys: set[bytes] = {board_key(initial_board)}

    frontier_boards: list[np.ndarray] = []
    frontier_paths: list[str] = []

    # Generate all valid moves from the initial board
    for piece in get_pieces_adjacent_to_empty(initial_board):
        for direction_code, direction_char in enumerate(["u", "d", "l", "r"]):
            valid, moved = move_piece(initial_board.copy(), piece, direction_code)
            if valid:
                k = board_key(moved)
                if k not in visited_keys:
                    visited_keys.add(k)
                    frontier_boards.append(moved.copy())
                    frontier_paths.append(f"{piece}{direction_char}")

    t_global_start = perf_counter()

    # ----- BFS Loop -----
    for depth in range(1, MAX_DEPTH + 1):
        t_depth_start = perf_counter()

        # Expand one layer, dedupe within the new layer
        expanded_boards, expanded_paths = expand_unique_frontier(frontier_boards, frontier_paths)

        # Filter out boards already visited across all previous layers
        new_frontier_boards: list[np.ndarray] = []
        new_frontier_paths: list[str] = []

        # Deduplication count and filtering
        removed_count = 0
        for b, p in zip(expanded_boards, expanded_paths):
            k = board_key(b)
            if k in visited_keys:
                removed_count += 1
                continue
            visited_keys.add(k)
            visited_boards.append(b.copy())
            new_frontier_boards.append(b.copy())
            new_frontier_paths.append(p)

        frontier_boards = new_frontier_boards
        frontier_paths = new_frontier_paths

        # Check if solved
        solved = False
        for idx, b in enumerate(frontier_boards):
            if is_solved(b):
                solved = True
                print("-" * 60)
                print("%" * 60)
                print("-" * 60)
                print(f"Solved at depth {depth}")
                plot_board(b)
                print("Moves:")
                print(frontier_paths[idx])
                print("-" * 60)
                print("%" * 60)
                print("-" * 60)
                break

        t_depth_end = perf_counter()

        print(
            f"Depth {depth:4d} --- "
            f"Frontier boards: {len(frontier_boards):6d} --- "
            f"Visited boards: {len(visited_boards):6d} --- "
            f"Removed (already visited): {removed_count:6d} --- "
            f"Depth time: {t_depth_end - t_depth_start:.4f}s"
        )

        # If frontier is empty, there are no more possible paths
        if len(frontier_boards) == 0:
            print("No solution.")
            break

        if solved:
            break

    t_global_end = perf_counter()
    print(f"Total runtime: {t_global_end - t_global_start:.4f}s")