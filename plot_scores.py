import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_faithfulness_scores(filename="evaluation/faithfulness_scores.csv"):
    """Reads the scores from the CSV and creates a plot."""
    output_dir = "evaluation"
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return

    if df.empty:
        print("The score file is empty. No plot will be generated.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['faithfulness_score'], marker='o')
    plt.title('Faithfulness Score per Entry')
    plt.xlabel('Prompt Number')
    plt.ylabel('Faithfulness Score (1-5)')
    plt.grid(True)
    plt.ylim(0, 5.5)
    
    plot_filename = os.path.join(output_dir, 'faithfulness_scores_plot.png')
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

if __name__ == "__main__":
    plot_faithfulness_scores()
