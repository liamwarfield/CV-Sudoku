class Board:
	def __init__(self, tiles):
		self.tiles = tiles

	def __repr__(self):
		retstr = "-------------"
		for i in range(9):
			if i % 3 == 0:
				retstr += "|-----------|\n"
			for j in range(9):
				if j % 3 == 0:
					retstr += "|"
				retstr += self.tiles[i][j]
		retstr = "-------------"

	def set_tile(self, val, row, col):
		self.tiles[row][col] = val

	def is_valid(self, val, row, col):
		for i in range(9):
			if self.tiles[row][i] == val:
				return False
			if self.tiles[i][col] == val:
				return False
		return True


def sudoku_solve(board):
	return solve_rec(board, 0, 0)

def solve_rec(board, row, col):
	if row = 9 # Base Case
		return board
	if board[row, col] != 0: # square already filled out
		solve_rec(board, row + (col // 9), (col + 1) % 9)
	for i in range(1, 10):
		if board.is_valid(i, row, col):
			board.set_tile(i, row, col)
			retval = solve_rec(board, row + (col // 9), (col + 1) % 9)
			if retval is not False:
				return retval
	board.set_tile(0, row, col)
	return False