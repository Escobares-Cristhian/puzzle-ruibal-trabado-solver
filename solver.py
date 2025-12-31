import os
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import numba as nb

# For movie export
import matplotlib.animation as animation


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

# Output folder
OUT_FOLDER = "output"


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


# ----- Reconstruct boards & save a movie efficiently -----
_DIR_TO_CODE = {"u": 0, "d": 1, "l": 2, "r": 3}


def boards_from_path(initial: np.ndarray, path: str) -> list[np.ndarray]:
    """
    Reconstruct the full board sequence from the initial board and a path like:
        "2u 3d 5l ..."
    Returns a list of boards: [initial, after_1, after_2, ...]
    """
    boards: list[np.ndarray] = [initial.copy()]
    if not path.strip():
        return boards

    current = initial.copy()
    for token in path.strip().split():
        piece = int(token[:-1])
        direction_char = token[-1]
        direction_code = _DIR_TO_CODE[direction_char]
        valid, current = move_piece(current.copy(), piece, direction_code)
        if not valid:
            raise ValueError(f"Invalid move in solution path: {token}")
        boards.append(current.copy())
    return boards


def save_solution_movie(
    boards: list[np.ndarray],
    out_path: str = "solution.mp4",
    fps: int = 8,
    dpi: int = 160,
    hold_last_seconds: float = 5.0,
) -> str:
    """
    Save a movie of the solution.

    - Uses FFmpeg for .mp4 if available (fast + compact).
    - Falls back to .gif (Pillow) if FFmpeg isn't available.
    """
    if len(boards) == 0:
        raise ValueError("No boards to render.")

    # Precompute frame arrays once (fast updates)
    fps = int(fps)
    if fps < 1:
        fps = 1
    frames = []
    for b in boards:
        arr = b.astype(float)
        arr[arr == 0] = np.nan
        frames.append(arr)

    # Hold the final frame for a fixed wall-clock duration (independent of fps choice)
    hold_frames = int(round(max(0.0, hold_last_seconds) * fps))
    total_frames = len(frames) + hold_frames

    # Colormap with explicit NaN color for consistent output
    cmap = plt.get_cmap("tab10")
    try:
        cmap = cmap.copy()
    except Exception:
        pass
    try:
        cmap.set_bad(color="white")
    except Exception:
        pass

    fig, ax = plt.subplots()
    ax.set_axis_off()

    im = ax.imshow(frames[0], cmap=cmap)

    # Text artists: one per cell, updated each frame (board is tiny -> very fast)
    nrows, ncols = frames[0].shape
    texts = []
    for r in range(nrows):
        for c in range(ncols):
            t = ax.text(c, r, "", ha="center", va="center", color="w")
            texts.append(t)

    def _update(frame_idx: int):
        # During the 'hold' tail, keep showing the last actual board
        idx = frame_idx if frame_idx < len(frames) else (len(frames) - 1)
        arr = frames[idx]
        im.set_data(arr)

        # Update numbers
        k = 0
        for r in range(nrows):
            for c in range(ncols):
                v = arr[r, c]
                if np.isnan(v):
                    texts[k].set_text("")
                else:
                    texts[k].set_text(str(int(v)))
                k += 1

        return (im, *texts)

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=total_frames,
        interval=int(1000 / max(1, fps)),
        blit=True,
        repeat=False,
    )

    # Pick writer
    lower = out_path.lower()
    if lower.endswith(".gif"):
        writer = animation.PillowWriter(fps=fps)
    else:
        if animation.writers.is_available("ffmpeg"):
            writer = animation.FFMpegWriter(
                fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]
            )
        else:
            # Fallback: save GIF instead
            out_path = out_path.rsplit(".", 1)[0] + ".gif"
            writer = animation.PillowWriter(fps=fps)

    ani.save(f"{OUT_FOLDER}/{out_path}", writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path


# ----- Main solver loop -----
if __name__ == "__main__":
    # Ensure folders exist
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)
    
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

    solved = False
    solved_path = ""

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
        for idx, b in enumerate(frontier_boards):
            if is_solved(b):
                solved = True
                solved_path = frontier_paths[idx]

                print("-" * 60)
                print("%" * 60)
                print("-" * 60)
                print(f"Solved at depth {depth}")
                plot_board(b)
                print("Moves:")
                print(solved_path)
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

    # ----- Save movie after finding a solution -----
    if solved:
        boards = boards_from_path(initial_board, solved_path)
        movie_path = save_solution_movie(
            boards, out_path="solution.mp4", fps=1, dpi=200, hold_last_seconds=5.0
        )
        print(f"Saved solution movie to: {movie_path}")