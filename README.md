# Game-With-Naive-Bayes
# CA2 - Game

## Yasaman Jafari - 810195376

In this project, we have an adverserial search algorithm. We have a zero sum game and when an agent loses, the other wins. In this game we have 2 agents and when one of them can't move anymore it loses and the other one wins.

For adverserial search we can use Minimax search. It can be used for deterministic, zero sum games. Each agent tries to maximize its score and assumes that the other agent is intelligent and tries to minimize the first agent's score.(maximizes its own score)

In this algorithm, we do a tree-space search. The nodes of these trees can be either minimizer or maximizer. Maximizer chooses the state which leads to maximum possible score and minimizer chooses the minimum score. These two nodes play in turn.

Minimax algorithm is actually a backtracking algorithm which tries to find the optimal movement for each agent. We assign a value to each state.

Time Complexity: O($b^m$)

Space Complexity: O(bm)

Alpha-betha pruning is an optimization technique used on minimax algorithm. Using alpha-betha pruning we can do the search much faster as we identify nad eliminate the useless nodes faster.

Alpha is the best value that the maximizer currently can be sure of at current level or above.

Betha is the best value that the minimizer currently can be sure of at current level or above.

Using alpha-betha pruning the time complexity can drop to O($b^(m/2)$)

For each movement, if we try to search all the tree to the leaf nodes, there are so many different states and it takes a very long time to analyze all the states. In many games we cannot do this.

So, we use an evaluation function. We choose a maximum depth and up to that level, we analyze all the possible states. After we reach the maximum depth, instead of creating the actual next possible states, we use an evaluation function which tries to predict how good each of these steps are.


```python
import random
import copy
import time
import numpy as np
```


```python
minimaxTime = []
alphaBethaTime = []
```


```python
class GameError(AttributeError):
    pass
```


```python
class Game:

    def __init__(self, n):
        self.size = n
        self.half_the_size = int(n/2)
        self.reset()

    def reset(self):
        self.board = []
        value = 'B'
        for i in range(self.size):
            row = []
            for j in range(self.size):
                row.append(value)
                value = self.opponent(value)
            self.board.append(row)
            if self.size%2 == 0:
                value = self.opponent(value)

    def __str__(self):
        result = "  "
        for i in range(self.size):
            result += str(i) + " "
        result += "\n"
        for i in range(self.size):
            result += str(i) + " "
            for j in range(self.size):
                result += str(self.board[i][j]) + " "
            result += "\n"
        return result

    def valid(self, row, col):
        return row >= 0 and col >= 0 and row < self.size and col < self.size

    def contains(self, board, row, col, symbol):
        return self.valid(row,col) and board[row][col]==symbol

    def countSymbol(self, board, symbol):
        count = 0
        for r in range(self.size):
            for c in range(self.size):
                if board[r][c] == symbol:
                    count += 1
        return count

    def opponent(self, player):
        if player == 'B':
            return 'W'
        else:
            return 'B'

    def distance(self, r1, c1, r2, c2):
        return abs(r1-r2 + c1-c2)

    def makeMove(self, player, move):
        self.board = self.nextBoard(self.board, player, move)

    def nextBoard(self, board, player, move):
        r1 = move[0]
        c1 = move[1]
        r2 = move[2]
        c2 = move[3]
        next = copy.deepcopy(board)
        if not (self.valid(r1, c1) and self.valid(r2, c2)):
            raise GameError
        if next[r1][c1] != player:
            raise GameError
        dist = self.distance(r1, c1, r2, c2)
        if dist == 0:
            if self.openingMove(board):
                next[r1][c1] = "."
                return next
            raise GameError
        if next[r2][c2] != ".":
            raise GameError
        jumps = int(dist/2)
        dr = int((r2 - r1)/dist)
        dc = int((c2 - c1)/dist)
        for i in range(jumps):
            if next[r1+dr][c1+dc] != self.opponent(player):
                raise GameError
            next[r1][c1] = "."
            next[r1+dr][c1+dc] = "."
            r1 += 2*dr
            c1 += 2*dc
            next[r1][c1] = player
        return next

    def openingMove(self, board):
        return self.countSymbol(board, ".") <= 1

    def generateFirstMoves(self, board):
        moves = []
        moves.append([0]*4)
        moves.append([self.size-1]*4)
        moves.append([self.half_the_size]*4)
        moves.append([(self.half_the_size)-1]*4)
        return moves

    def generateSecondMoves(self, board):
        moves = []
        if board[0][0] == ".":
            moves.append([0,1]*2)
            moves.append([1,0]*2)
            return moves
        elif board[self.size-1][self.size-1] == ".":
            moves.append([self.size-1,self.size-2]*2)
            moves.append([self.size-2,self.size-1]*2)
            return moves
        elif board[self.half_the_size-1][self.half_the_size-1] == ".":
            pos = self.half_the_size -1
        else:
            pos = self.half_the_size
        moves.append([pos,pos-1]*2)
        moves.append([pos+1,pos]*2)
        moves.append([pos,pos+1]*2)
        moves.append([pos-1,pos]*2)
        return moves

    def check(self, board, r, c, rd, cd, factor, opponent):
        if self.contains(board,r+factor*rd,c+factor*cd,opponent) and \
           self.contains(board,r+(factor+1)*rd,c+(factor+1)*cd,'.'):
            return [[r,c,r+(factor+1)*rd,c+(factor+1)*cd]] + \
                   self.check(board,r,c,rd,cd,factor+2,opponent)
        else:
            return []

    def generateMoves(self, board, player):
        if self.openingMove(board):
            if player=='B':
                return self.generateFirstMoves(board)
            else:
                return self.generateSecondMoves(board)
        else:
            moves = []
            rd = [-1,0,1,0]
            cd = [0,1,0,-1]
            for r in range(self.size):
                for c in range(self.size):
                    if board[r][c] == player:
                        for i in range(len(rd)):
                            moves += self.check(board,r,c,rd[i],cd[i],1,
                                                self.opponent(player))
            return moves

    def playOneGame(self, p1, p2, show):
        self.reset()
        while True:
            if show:
                print(self)
                print("player B's turn")
            start_time = time.time()
            move = p1.getMove(self.board)
            passedTime = time.time() - start_time
            minimaxTime.append(passedTime)
            print("B : --- %s seconds ---" % passedTime)
            if move == []:
                print("Game over")
                return 'W'
            try:
                self.makeMove('B', move)

            except GameError:
                print("Game over: Invalid move by", p1.name)
                print(move)
                print(self)
                return 'W'
            if show:
                print(move)
                print(self)
                print("player W's turn")
            start_time = time.time()
            move = p2.getMove(self.board)
            passedTime = time.time() - start_time
            alphaBethaTime.append(passedTime)
            print("W : --- %s seconds ---" % passedTime)
            if move == []:
                print("Game over")
                return 'B'
            try:        
                self.makeMove('W', move)
            except GameError:
                print("Game over: Invalid move by", p2.name)
                print(move)
                print(self)
                return 'B'
            if show:
                print(move)

    def playNGames(self, n, p1, p2, show):
        first = p1
        second = p2
        for i in range(n):
            print("Game", i)
            winner = self.playOneGame(first, second, show)
            if winner == 'B':
                first.won()
                second.lost()
                print(first.name, "wins")
            else:
                first.lost()
                second.won()
                print(second.name, "wins")
#             first, second = second, first
```


```python
class Player:
    name = "Player"
    wins = 0
    losses = 0
    def results(self):
        result = self.name
        result += " Wins:" + str(self.wins)
        result += " Losses:" + str(self.losses)
        return result
    def lost(self):
        self.losses += 1
    def won(self):
        self.wins += 1
    def reset(self):
        self.wins = 0
        self.losses = 0

    def initialize(self, side):
        abstract()

    def getMove(self, board):
        abstract()
```


```python
class SimplePlayer(Game, Player):
    def initialize(self, side):
        self.side = side
        self.name = "Simple"
    def getMove(self, board):
        moves = self.generateMoves(board, self.side)
        n = len(moves)
        if n == 0:
            return []
        else:
            return moves[0]
```


```python
class RandomPlayer(Game, Player):
    def initialize(self, side):
        self.side = side
        self.name = "Random"
    def getMove(self, board):
        moves = self.generateMoves(board, self.side)
        n = len(moves)
        if n == 0:
            return []
        else:
            return moves[random.randrange(0, n)]
```


```python
class HumanPlayer(Game, Player):
    def initialize(self, side):
        self.side = side
        self.name = "Human"
    def getMove(self, board):
        moves = self.generateMoves(board, self.side)
        while True:
            print("Possible moves:", moves)
            n = len(moves)
            if n == 0:
                print("You must concede")
                return []
            index = input("Enter index of chosen move (0-"+ str(n-1) +
                          ") or -1 to concede: ")
            try:
                index = int(index)
                if index == -1:
                    return []
                if 0 <= index <= (n-1):
                    print("returning", moves[index])
                    return moves[index]
                else:
                    print("Invalid choice, try again.")
            except Exception as e:
                print("Invalid choice, try again.")
```

In order to know whether changing the evaluation function improves the algorithm, I created an extra agent with the previous evaluation function and had a competition with it.(The agents with most recent evaluation function and the previous one compete against each other.)


```python
class AnotherPlayer(Game, Player):
    def __init__(self, size, maxDepth=4, pruning=False):
        super().__init__(size)
        self.maxDepth = maxDepth
        self.pruning = pruning
        
    def initialize(self, side):
        self.side = side
        self.name = "Another"
        
    def getEvaluatedValue(self, currBoard, myMoves, opponentMoves, myMovablePieces, opponentMovablePieces, myPieceCount, opponentPieceCount):
        allCount = (myPieceCount - opponentPieceCount)
#         distance = self.getDistanceOfAllPlayersFromAllOpponents(currBoard)
        allMyMoves = len(myMoves)
        allOpponentMoves = len(opponentMoves)
        allPossibleMoves = (allMyMoves - allOpponentMoves)
        movablePieces = (myMovablePieces - opponentMovablePieces)
        return movablePieces + 200 * allPossibleMoves
    
    def getDistanceOfAllPlayersFromAllOpponents(self, board):
        distanceSum = 0
        for r in range(self.size):
            for c in range(self.size):
                if board[r][c] == self.side:
                    distanceSum += self.getDistanceFromAllOpponenets(r, c, board)
        return distanceSum
    
    def getSelfAndOpponentPieceCount(self, board):
        selfCount = 0
        opponentCount = 0
        opponent = self.opponent(self.side)
        selfCount = sum([row.count(self.side) for row in board])
        opponentCount = sum([row.count(self.opponent(self.side)) for row in board])
        return selfCount, opponentCount
    
    def getDistanceFromAllOpponenets(self, x, y, board):
        distanceSum = 0
        opponentSymbol = self.opponent(self.side)
        for r in range(self.size):
            for c in range(self.size):
                if board[r][c] == opponentSymbol:
                    distanceSum += self.distance(x, y, r, c)
        return distanceSum
    
    def getMovablePiecesCount(self, board):
        selfCount = 0
        opponentCount = 0
        opponentSymbol = self.opponent(self.side)
        allMyMoves = []
        allOpponentMoves = []
        rd = [-1,0,1,0]
        cd = [0,1,0,-1]
        currBoard = np.array(board)
        selfPos = np.where(currBoard == self.side)
        opponentPos = np.where(currBoard == opponentSymbol)
        
        for i in range(len(selfPos[0])):
            r = selfPos[0][i]
            c = selfPos[1][i]
            found = False
            for i in range(len(rd)):
                prevLen = len(allMyMoves)
                allMyMoves += self.check(board,r,c,rd[i],cd[i],1,
                                    self.opponent(self.side))
                if (len(allMyMoves) > prevLen):
                    found = True
            if found:
                selfCount += 1

        for i in range(len(opponentPos[0])):
            r = opponentPos[0][i]
            c = opponentPos[1][i]
            found = False
            for i in range(len(rd)):
                prevLen = len(allOpponentMoves)
                allOpponentMoves += self.check(board,r,c,rd[i],cd[i],1,
                                    self.opponent(opponentSymbol))
                if (len(allOpponentMoves) > prevLen):
                    found = True
            if found:
                opponentCount += 1

        return selfCount, opponentCount, allMyMoves, allOpponentMoves, len(selfPos[0]), len(opponentPos[0])

    def minimax(self, currDepth, isMaximizer, currBoard, alpha=-float('inf'), betha=float('inf')):
        myMovablePieces, opponentMovablePieces, myPossibleMoves, opponentPossibleMoves, myPieceCount, opponentPieceCount = self.getMovablePiecesCount(currBoard)
        
        if(currDepth >= self.maxDepth and not (myPossibleMoves == [] or opponentPossibleMoves == [])):
            return self.getEvaluatedValue(currBoard, myPossibleMoves, opponentPossibleMoves, myMovablePieces, opponentMovablePieces, myPieceCount, opponentPieceCount)
      
        if isMaximizer: 
            currBest = -float('inf')
            for move in myPossibleMoves:
                currBest = max(self.minimax(currDepth + 1, not isMaximizer, self.nextBoard(currBoard, self.side, move), alpha, betha),
                           currBest)
                if self.pruning and currBest >= betha:
                    return currBest
                alpha = max(alpha, currBest)

        else:
            currBest = float('inf')
            for move in opponentPossibleMoves:
                currBest = min(self.minimax(currDepth + 1, not isMaximizer, self.nextBoard(currBoard, self.opponent(self.side), move), alpha, betha),
                       currBest)
                if self.pruning and currBest <= alpha:
                    return currBest
                betha = min(betha, currBest)
                
        return currBest
    
    def getMove(self, board):
        moves = self.generateMoves(board, self.side)
        self.maxVal = -float('inf')
        self.nextMove = []
        if moves == []:
            return []
        for move in moves:
            moveScore = self.minimax(1, False, self.nextBoard(board, self.side, move)); 
            if moveScore > self.maxVal:
                self.nextMove = move 
                self.maxVal = moveScore 
        return self.nextMove
```

The final evaluation function consists of two features:
- The difference between movable pieces of the agent and its opponent's movable pieces
- The difference between the count of agent's possible movements and its opponent's possible movements

As the final evaluation function, I used the weighted sum of these two features and gave the secind one more weight.

I used the first feature because the game ends when one of the players cannot move anymore and each agent wants to have more movable pieces and tries to minimize this count for its opponent.
The second feature is chosen because each agent tries to be in a position which can have many choices of where to move and tries to minimize this count for its opponent.

I also tried other features such as difference between remaining pieces of agent and its opponent or the sum of distance of agant's pieces and opponent's pieces. But, these features are not always good. For example when the oppoenet has less pieces on the board, our movements is restricted also.


```python
class MinimaxPlayer(Game, Player):
    def __init__(self, size, maxDepth=4, pruning=False):
        super().__init__(size)
        self.maxDepth = maxDepth
        self.pruning = pruning
        
    def initialize(self, side):
        self.side = side
        self.name = "Minimax"
        
    def getEvaluatedValue(self, currBoard, myMoves, opponentMoves, myMovablePieces, opponentMovablePieces, myPieceCount, opponentPieceCount):
        allCount = (myPieceCount - opponentPieceCount)
#         distance = self.getDistanceOfAllPlayersFromAllOpponents(currBoard)
        allMyMoves = len(myMoves)
        allOpponentMoves = len(opponentMoves)
        allPossibleMoves = (allMyMoves - allOpponentMoves)
        movablePieces = (myMovablePieces - opponentMovablePieces)
        return movablePieces + 200 * allPossibleMoves
    
    def getDistanceOfAllPlayersFromAllOpponents(self, board):
        distanceSum = 0
        for r in range(self.size):
            for c in range(self.size):
                if board[r][c] == self.side:
                    distanceSum += self.getDistanceFromAllOpponenets(r, c, board)
        return distanceSum
    
    def getSelfAndOpponentPieceCount(self, board):
        selfCount = 0
        opponentCount = 0
        opponent = self.opponent(self.side)
        selfCount = sum([row.count(self.side) for row in board])
        opponentCount = sum([row.count(self.opponent(self.side)) for row in board])
        return selfCount, opponentCount
    
    def getDistanceFromAllOpponenets(self, x, y, board):
        distanceSum = 0
        opponentSymbol = self.opponent(self.side)
        for r in range(self.size):
            for c in range(self.size):
                if board[r][c] == opponentSymbol:
                    distanceSum += self.distance(x, y, r, c)
        return distanceSum
    
    def getMovablePiecesCount(self, board):
        selfCount = 0
        opponentCount = 0
        opponentSymbol = self.opponent(self.side)
        allMyMoves = []
        allOpponentMoves = []
        rd = [-1,0,1,0]
        cd = [0,1,0,-1]
        currBoard = np.array(board)
        selfPos = np.where(currBoard == self.side)
        opponentPos = np.where(currBoard == opponentSymbol)
        
        for i in range(len(selfPos[0])):
            r = selfPos[0][i]
            c = selfPos[1][i]
            found = False
            for i in range(len(rd)):
                prevLen = len(allMyMoves)
                allMyMoves += self.check(board,r,c,rd[i],cd[i],1,
                                    self.opponent(self.side))
                if (len(allMyMoves) > prevLen):
                    found = True
            if found:
                selfCount += 1

        for i in range(len(opponentPos[0])):
            r = opponentPos[0][i]
            c = opponentPos[1][i]
            found = False
            for i in range(len(rd)):
                prevLen = len(allOpponentMoves)
                allOpponentMoves += self.check(board,r,c,rd[i],cd[i],1,
                                    self.opponent(opponentSymbol))
                if (len(allOpponentMoves) > prevLen):
                    found = True
            if found:
                opponentCount += 1

        return selfCount, opponentCount, allMyMoves, allOpponentMoves, len(selfPos[0]), len(opponentPos[0])

    def minimax(self, currDepth, isMaximizer, currBoard, alpha=-float('inf'), betha=float('inf')):
        myMovablePieces, opponentMovablePieces, myPossibleMoves, opponentPossibleMoves, myPieceCount, opponentPieceCount = self.getMovablePiecesCount(currBoard)
        
        if(currDepth >= self.maxDepth and not (myPossibleMoves == [] or opponentPossibleMoves == [])):
            return self.getEvaluatedValue(currBoard, myPossibleMoves, opponentPossibleMoves, myMovablePieces, opponentMovablePieces, myPieceCount, opponentPieceCount)
      
        if isMaximizer: 
            currBest = -float('inf')
            for move in myPossibleMoves:
                currBest = max(self.minimax(currDepth + 1, not isMaximizer, self.nextBoard(currBoard, self.side, move), alpha, betha),
                           currBest)
                if self.pruning and currBest >= betha:
                    return currBest
                alpha = max(alpha, currBest)

        else:
            currBest = float('inf')
            for move in opponentPossibleMoves:
                currBest = min(self.minimax(currDepth + 1, not isMaximizer, self.nextBoard(currBoard, self.opponent(self.side), move), alpha, betha),
                       currBest)
                if self.pruning and currBest <= alpha:
                    return currBest
                betha = min(betha, currBest)
                
        return currBest
    
    def getMove(self, board):
        moves = self.generateMoves(board, self.side)
        self.maxVal = -float('inf')
        self.nextMove = []
        if moves == []:
            return []
        for move in moves:
            moveScore = self.minimax(1, False, self.nextBoard(board, self.side, move)); 
            if moveScore > self.maxVal:
                self.nextMove = move 
                self.maxVal = moveScore 
        return self.nextMove
```


```python
if __name__ == '__main__':
    game = Game(8)
    player1 = AnotherPlayer(8, 3, False)
    player1.initialize('B')
    player2 = MinimaxPlayer(8, 3, True)
    player2.initialize('W')
    game.playNGames(3, player1, player2, False)
```

    Game 0
    B : --- 0.004087924957275391 seconds ---
    W : --- 0.010550975799560547 seconds ---
    B : --- 0.01404118537902832 seconds ---
    W : --- 0.04054594039916992 seconds ---
    B : --- 0.08426713943481445 seconds ---
    W : --- 0.055090904235839844 seconds ---
    B : --- 0.13326072692871094 seconds ---
    W : --- 0.08658719062805176 seconds ---
    B : --- 0.10260367393493652 seconds ---
    W : --- 0.07641434669494629 seconds ---
    B : --- 0.14644813537597656 seconds ---
    W : --- 0.20324492454528809 seconds ---
    B : --- 0.3350651264190674 seconds ---
    W : --- 0.42127394676208496 seconds ---
    B : --- 1.4528329372406006 seconds ---
    W : --- 0.6019141674041748 seconds ---
    B : --- 1.4196138381958008 seconds ---
    W : --- 0.491102933883667 seconds ---
    B : --- 1.4616448879241943 seconds ---
    W : --- 0.5098860263824463 seconds ---
    B : --- 0.6045382022857666 seconds ---
    W : --- 0.15825891494750977 seconds ---
    B : --- 0.381119966506958 seconds ---
    W : --- 0.36721110343933105 seconds ---
    B : --- 0.9082221984863281 seconds ---
    W : --- 0.47905826568603516 seconds ---
    B : --- 0.6150338649749756 seconds ---
    W : --- 0.5408990383148193 seconds ---
    B : --- 0.9201450347900391 seconds ---
    W : --- 0.23739409446716309 seconds ---
    B : --- 0.3548159599304199 seconds ---
    W : --- 0.19902992248535156 seconds ---
    B : --- 0.3336958885192871 seconds ---
    W : --- 0.14397788047790527 seconds ---
    B : --- 0.4030110836029053 seconds ---
    W : --- 0.09320592880249023 seconds ---
    B : --- 0.1655890941619873 seconds ---
    W : --- 0.08314323425292969 seconds ---
    B : --- 0.21138787269592285 seconds ---
    W : --- 0.11118197441101074 seconds ---
    B : --- 0.16535496711730957 seconds ---
    W : --- 0.10544991493225098 seconds ---
    B : --- 0.07616519927978516 seconds ---
    W : --- 0.03316998481750488 seconds ---
    B : --- 0.03566884994506836 seconds ---
    W : --- 0.024629831314086914 seconds ---
    B : --- 0.007599830627441406 seconds ---
    Game over
    Minimax wins
    Game 1
    B : --- 0.003699064254760742 seconds ---
    W : --- 0.009709835052490234 seconds ---
    B : --- 0.01289820671081543 seconds ---
    W : --- 0.0380253791809082 seconds ---
    B : --- 0.09354710578918457 seconds ---
    W : --- 0.05116009712219238 seconds ---
    B : --- 0.12164902687072754 seconds ---
    W : --- 0.1125938892364502 seconds ---
    B : --- 0.1306319236755371 seconds ---
    W : --- 0.08674788475036621 seconds ---
    B : --- 0.2635529041290283 seconds ---
    W : --- 0.23293328285217285 seconds ---
    B : --- 0.39232468605041504 seconds ---
    W : --- 0.42174506187438965 seconds ---
    B : --- 1.0024268627166748 seconds ---
    W : --- 0.6245501041412354 seconds ---
    B : --- 1.188014268875122 seconds ---
    W : --- 0.52593994140625 seconds ---
    B : --- 0.9903531074523926 seconds ---
    W : --- 0.3862171173095703 seconds ---
    B : --- 0.5596950054168701 seconds ---
    W : --- 0.1395399570465088 seconds ---
    B : --- 0.31289076805114746 seconds ---
    W : --- 0.3275432586669922 seconds ---
    B : --- 0.8095691204071045 seconds ---
    W : --- 0.4495561122894287 seconds ---
    B : --- 0.6022450923919678 seconds ---
    W : --- 0.5048809051513672 seconds ---
    B : --- 0.7340519428253174 seconds ---
    W : --- 0.2030041217803955 seconds ---
    B : --- 0.32495903968811035 seconds ---
    W : --- 0.20872807502746582 seconds ---
    B : --- 0.31993699073791504 seconds ---
    W : --- 0.11978387832641602 seconds ---
    B : --- 0.47350192070007324 seconds ---
    W : --- 0.10360884666442871 seconds ---
    B : --- 0.16970181465148926 seconds ---
    W : --- 0.07976078987121582 seconds ---
    B : --- 0.18304800987243652 seconds ---
    W : --- 0.0961000919342041 seconds ---
    B : --- 0.14687013626098633 seconds ---
    W : --- 0.09612798690795898 seconds ---
    B : --- 0.06528997421264648 seconds ---
    W : --- 0.0403599739074707 seconds ---
    B : --- 0.03592395782470703 seconds ---
    W : --- 0.023525714874267578 seconds ---
    B : --- 0.010163068771362305 seconds ---
    Game over
    Minimax wins
    Game 2
    B : --- 0.006289243698120117 seconds ---
    W : --- 0.009766101837158203 seconds ---
    B : --- 0.015512228012084961 seconds ---
    W : --- 0.04851698875427246 seconds ---
    B : --- 0.11088395118713379 seconds ---
    W : --- 0.05059313774108887 seconds ---
    B : --- 0.10651087760925293 seconds ---
    W : --- 0.07325601577758789 seconds ---
    B : --- 0.09921097755432129 seconds ---
    W : --- 0.07542562484741211 seconds ---
    B : --- 0.156998872756958 seconds ---
    W : --- 0.2023921012878418 seconds ---
    B : --- 0.3402090072631836 seconds ---
    W : --- 0.42736291885375977 seconds ---
    B : --- 0.9701642990112305 seconds ---
    W : --- 0.4842410087585449 seconds ---
    B : --- 1.3562171459197998 seconds ---
    W : --- 0.5363850593566895 seconds ---
    B : --- 1.1106889247894287 seconds ---
    W : --- 0.4529280662536621 seconds ---
    B : --- 0.6413109302520752 seconds ---
    W : --- 0.16440701484680176 seconds ---
    B : --- 0.47205209732055664 seconds ---
    W : --- 0.38471198081970215 seconds ---
    B : --- 0.9581632614135742 seconds ---
    W : --- 0.7361950874328613 seconds ---
    B : --- 1.0860209465026855 seconds ---
    W : --- 0.616469144821167 seconds ---
    B : --- 0.6942310333251953 seconds ---
    W : --- 0.23272180557250977 seconds ---
    B : --- 0.30439090728759766 seconds ---
    W : --- 0.18497610092163086 seconds ---
    B : --- 0.35086607933044434 seconds ---
    W : --- 0.13258886337280273 seconds ---
    B : --- 0.4548370838165283 seconds ---
    W : --- 0.10281920433044434 seconds ---
    B : --- 0.1840040683746338 seconds ---
    W : --- 0.09034013748168945 seconds ---
    B : --- 0.25527501106262207 seconds ---
    W : --- 0.0991978645324707 seconds ---
    B : --- 0.15625905990600586 seconds ---
    W : --- 0.12386202812194824 seconds ---
    B : --- 0.24459481239318848 seconds ---
    W : --- 0.035047054290771484 seconds ---
    B : --- 0.033628225326538086 seconds ---
    W : --- 0.01980280876159668 seconds ---
    B : --- 0.00859212875366211 seconds ---
    Game over
    Minimax wins


The alpha-betha pruning has no effect on minimax value computed for the root as it only eliminates the nodes when it is sure that they are useless. For example it is sure that some node's children is never chosen by higher level nodes so it eliminates all those children. 
Note: The values of intermediate nodes might be wrong and not accurate but the result is not different.

### Timing


```python
minimaxTime = []
alphaBethaTime = []
player3 = MinimaxPlayer(8, 3, False)
player3.initialize('B')
player4 = MinimaxPlayer(8, 3, True)
player4.initialize('W')
game.playOneGame(player3, player4, False)
```

    B : --- 0.005924224853515625 seconds ---
    W : --- 0.015657901763916016 seconds ---
    B : --- 0.022505998611450195 seconds ---
    W : --- 0.0425410270690918 seconds ---
    B : --- 0.11636018753051758 seconds ---
    W : --- 0.05465221405029297 seconds ---
    B : --- 0.1372370719909668 seconds ---
    W : --- 0.1149759292602539 seconds ---
    B : --- 0.11743688583374023 seconds ---
    W : --- 0.09514904022216797 seconds ---
    B : --- 0.18293404579162598 seconds ---
    W : --- 0.23330187797546387 seconds ---
    B : --- 0.34970617294311523 seconds ---
    W : --- 0.5615649223327637 seconds ---
    B : --- 1.0207202434539795 seconds ---
    W : --- 0.5019409656524658 seconds ---
    B : --- 1.1362183094024658 seconds ---
    W : --- 0.5476517677307129 seconds ---
    B : --- 1.104362964630127 seconds ---
    W : --- 0.43600010871887207 seconds ---
    B : --- 0.7243282794952393 seconds ---
    W : --- 0.18148112297058105 seconds ---
    B : --- 0.38410425186157227 seconds ---
    W : --- 0.35708189010620117 seconds ---
    B : --- 1.0394420623779297 seconds ---
    W : --- 0.5461833477020264 seconds ---
    B : --- 0.7060651779174805 seconds ---
    W : --- 0.5517199039459229 seconds ---
    B : --- 0.7029330730438232 seconds ---
    W : --- 0.22762632369995117 seconds ---
    B : --- 0.3407890796661377 seconds ---
    W : --- 0.2791571617126465 seconds ---
    B : --- 0.40288305282592773 seconds ---
    W : --- 0.16011404991149902 seconds ---
    B : --- 0.4548819065093994 seconds ---
    W : --- 0.10566210746765137 seconds ---
    B : --- 0.1652066707611084 seconds ---
    W : --- 0.09270477294921875 seconds ---
    B : --- 0.1994309425354004 seconds ---
    W : --- 0.09614896774291992 seconds ---
    B : --- 0.16762590408325195 seconds ---
    W : --- 0.1127161979675293 seconds ---
    B : --- 0.08246994018554688 seconds ---
    W : --- 0.032618045806884766 seconds ---
    B : --- 0.03242325782775879 seconds ---
    W : --- 0.023233890533447266 seconds ---
    B : --- 0.011140108108520508 seconds ---
    Game over





    'W'




```python
print("MinimaxTime: ")
print(sum(minimaxTime)/len(minimaxTime))
print("MinimaxTime with Pruning: ")
print(sum(alphaBethaTime)/len(alphaBethaTime))
```

    MinimaxTime: 
    0.400297075510025
    MinimaxTime with Pruning: 
    0.23347319727358612


As you can see, the time is much less when using alpha-betha pruning as many nodes are not expanded when the algorithm recognizes them as useless.
(The order is compared at the beginning)


```python
minimaxTime = []
alphaBethaTime = []
player5 = MinimaxPlayer(8, 3, False)
player5.initialize('B')
player6 = MinimaxPlayer(8, 3, False)
player6.initialize('W')
start_time = time.time()
game.playOneGame(player5, player6, False)
passedTime = time.time() - start_time
```

    B : --- 0.004724979400634766 seconds ---
    W : --- 0.011758089065551758 seconds ---
    B : --- 0.019885778427124023 seconds ---
    W : --- 0.053381919860839844 seconds ---
    B : --- 0.0817720890045166 seconds ---
    W : --- 0.07191991806030273 seconds ---
    B : --- 0.09862780570983887 seconds ---
    W : --- 0.14351296424865723 seconds ---
    B : --- 0.10145020484924316 seconds ---
    W : --- 0.12634611129760742 seconds ---
    B : --- 0.14262604713439941 seconds ---
    W : --- 0.3134191036224365 seconds ---
    B : --- 0.36068010330200195 seconds ---
    W : --- 0.6848220825195312 seconds ---
    B : --- 0.8777248859405518 seconds ---
    W : --- 0.9706830978393555 seconds ---
    B : --- 1.0700068473815918 seconds ---
    W : --- 1.0503439903259277 seconds ---
    B : --- 1.0859758853912354 seconds ---
    W : --- 0.61277174949646 seconds ---
    B : --- 0.5444927215576172 seconds ---
    W : --- 0.2022240161895752 seconds ---
    B : --- 0.3579862117767334 seconds ---
    W : --- 0.648472785949707 seconds ---
    B : --- 0.8377740383148193 seconds ---
    W : --- 0.9804439544677734 seconds ---
    B : --- 0.6033980846405029 seconds ---
    W : --- 1.1928410530090332 seconds ---
    B : --- 0.616840124130249 seconds ---
    W : --- 0.3447089195251465 seconds ---
    B : --- 0.2719151973724365 seconds ---
    W : --- 0.4290938377380371 seconds ---
    B : --- 0.33709168434143066 seconds ---
    W : --- 0.2636427879333496 seconds ---
    B : --- 0.4755549430847168 seconds ---
    W : --- 0.190201997756958 seconds ---
    B : --- 0.1952049732208252 seconds ---
    W : --- 0.19866585731506348 seconds ---
    B : --- 0.19762802124023438 seconds ---
    W : --- 0.15379095077514648 seconds ---
    B : --- 0.33368587493896484 seconds ---
    W : --- 0.12562918663024902 seconds ---
    B : --- 0.0821220874786377 seconds ---
    W : --- 0.03990983963012695 seconds ---
    B : --- 0.03462386131286621 seconds ---
    W : --- 0.02180194854736328 seconds ---
    B : --- 0.010046005249023438 seconds ---
    Game over



```python
print("Game Time: ")
print(passedTime)
```

    Game Time: 
    17.58314800262451



```python
minimaxTime = []
alphaBethaTime = []
player7 = MinimaxPlayer(8, 3, True)
player7.initialize('B')
player8 = MinimaxPlayer(8, 3, True)
player8.initialize('W')
start_time = time.time()
game.playOneGame(player7, player8, False)
passedTime = time.time() - start_time
```

    B : --- 0.0038199424743652344 seconds ---
    W : --- 0.009564876556396484 seconds ---
    B : --- 0.01259303092956543 seconds ---
    W : --- 0.03747892379760742 seconds ---
    B : --- 0.04546189308166504 seconds ---
    W : --- 0.05243515968322754 seconds ---
    B : --- 0.06252694129943848 seconds ---
    W : --- 0.0885319709777832 seconds ---
    B : --- 0.07367610931396484 seconds ---
    W : --- 0.08241009712219238 seconds ---
    B : --- 0.08913993835449219 seconds ---
    W : --- 0.19874787330627441 seconds ---
    B : --- 0.22379112243652344 seconds ---
    W : --- 0.37114977836608887 seconds ---
    B : --- 0.48966407775878906 seconds ---
    W : --- 0.510059118270874 seconds ---
    B : --- 0.7165729999542236 seconds ---
    W : --- 0.535959005355835 seconds ---
    B : --- 0.5560760498046875 seconds ---
    W : --- 0.4508020877838135 seconds ---
    B : --- 0.27490782737731934 seconds ---
    W : --- 0.14910006523132324 seconds ---
    B : --- 0.1823749542236328 seconds ---
    W : --- 0.3475179672241211 seconds ---
    B : --- 0.45391011238098145 seconds ---
    W : --- 0.4966747760772705 seconds ---
    B : --- 0.3961949348449707 seconds ---
    W : --- 0.5662081241607666 seconds ---
    B : --- 0.3672819137573242 seconds ---
    W : --- 0.20758605003356934 seconds ---
    B : --- 0.20692801475524902 seconds ---
    W : --- 0.18879103660583496 seconds ---
    B : --- 0.21991205215454102 seconds ---
    W : --- 0.12433171272277832 seconds ---
    B : --- 0.24695110321044922 seconds ---
    W : --- 0.09645199775695801 seconds ---
    B : --- 0.14940190315246582 seconds ---
    W : --- 0.08425068855285645 seconds ---
    B : --- 0.1563560962677002 seconds ---
    W : --- 0.10636401176452637 seconds ---
    B : --- 0.10315108299255371 seconds ---
    W : --- 0.08855414390563965 seconds ---
    B : --- 0.05763578414916992 seconds ---
    W : --- 0.032032012939453125 seconds ---
    B : --- 0.028380155563354492 seconds ---
    W : --- 0.02146601676940918 seconds ---
    B : --- 0.007447957992553711 seconds ---
    Game over



```python
print("Game Time with pruning: ")
print(passedTime)
```

    Game Time with pruning: 
    9.984219789505005


The time of entire game is also shown above.
