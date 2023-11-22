from backgammon import backgammon
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 1. this can be used to read in teh model 
with open('B00832852_Model.pkl', 'rb') as file:
    agent = pickle.load(file)
    print("Success!")

ROC = []
WINNER = []

for i in range(1000):
    # Instantiate the backgammon object
    game = backgammon()

    top = []
    # Iterate over the moves until a winner is determined
    while game.get_winner() is None:
        # Iterate over the moves directly from the moves attribute
        scores = []
        white = game.get_board()[-2]==1
        for move in game.moves:
            # Randomly generate a scalar value
            if(white):
                score = agent.predict(move)

            else:
                score = random.random()
            scores.append(score)
            game.score_move(move, (score))
        
        if(white):
            best_score = max(scores, key=lambda x: x)
            top.append(best_score)


        # Check if a winner is determined after each iteration
        if game.get_winner() is not None:
            if(game.get_winner() == 'WHITE'):
                WINNER.append(1)
                ROC.append(top[-1])
            else:
                WINNER.append(0)
                ROC.append(top[-1])
            break

def win_rate(lst):
    if not lst:
        return 0.0

    count_ones = sum(1 for elem in lst if elem == 1)
    percentage = (count_ones / len(lst)) * 100

    return percentage

def evaluate_game_predictions(confidence, outcomes):
    if len(confidence) != len(outcomes):
        raise ValueError("Input arrays must have the same length.")

    correct_predictions = 0
    total_predictions = len(confidence)

    for i in range(total_predictions):
        if confidence[i] >= 0.5 and outcomes[i] == 1:
            correct_predictions += 1
        elif confidence[i] < 0.5 and outcomes[i] == 0:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions * 100
    return accuracy


def generate_roc_chart(confidence, outcomes):   
    fpr, tpr, thresholds = roc_curve(outcomes, confidence)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()



generate_roc_chart(ROC, WINNER)

accuracy = evaluate_game_predictions(ROC, WINNER)
print(f"Accuracy: {accuracy}%")
print(f"Win rate: {win_rate(WINNER)}%")
