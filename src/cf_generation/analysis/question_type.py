from datasets import load_dataset
from collections import Counter
from tqdm import tqdm

import plotly.graph_objs as go
import plotly.io as pio

# BASE_PATH = "/home/sachdeva/projects/ukp/exp_calibration/"
BASE_PATH = "/storage/ukp/work/sachdeva/research_projects/exp_calibration/"


def topk_questions(questions, k=5):
    """
    Counts the occurrences of the first word in each question and returns the top k most common questions.

    Args:
        questions (list): A list of questions to count.
        k (int): The number of top questions to return. Default is 5.

    Returns:
        list: A list of the top k most common questions based on the first word.
    """
    tokens_to_remove = ["[", "'", '"', "]", "/"]
    # Create a translation table that maps each unwanted token to None
    translator = str.maketrans({token: None for token in tokens_to_remove})
    questions = [ques.translate(translator).strip() for ques in questions]
    # remove empty questions
    questions = [ques for ques in questions if ques]
    print(len(questions))
    try:
        first_words = [q.split()[0].lower() for q in questions]
    except:
        print(questions)
        raise
    counter = Counter(first_words)
    return [(q, count) for q, count in counter.most_common(k)]


def plot_question_types(questions, k=5, file_name=None):
    """
    Creates a bar chart based on the types of questions in the input list.

    Args:
        questions (list): A list of questions to analyze.
        k (int): The number of top questions to display. Default is 5.
        file_name (str): The name of the file to save the plotly figure as..
    """
    # Get the top k most common question types and their counts
    topk = topk_questions(questions, k=k)
    labels, counts = zip(*topk)

    # Calculate the total number of questions
    total = len(questions)

    # Calculate the percentage of each question type
    percentages = [count / total * 100 for count in counts]

    # Create the bar chart
    fig = go.Figure([go.Bar(x=labels, y=percentages)])

    # Set the chart title and axis labels
    # fig.update_layout(title='Distribution of Question Types', xaxis_title='Question Type', yaxis_title='Count')

    # Adjust the font size based on the chart height
    height = 300
    width = 400

    # Set the height of the figure
    fig.update_layout(height=height)

    font_size = int(14 * (fig.layout.height / height))
    fig.update_layout(font=dict(size=font_size))

    # Adjust the margin to remove the side margins
    fig.update_layout(margin=dict(l=5, r=5, t=30, b=60))
    # Add a white background to the chart
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")

    # Set the showline and line attributes for the xaxis and yaxis objects to create a border around the chart
    fig.update_xaxes(showline=True, linecolor="gray", linewidth=1, mirror=True)
    # showgrid=True, gridcolor='lightgray', gridwidth=1)
    fig.update_yaxes(
        showline=True,
        linecolor="gray",
        linewidth=1,
        mirror=True,
        showgrid=True,
        gridcolor="lightgray",
        gridwidth=1,
    )

    # Save the chart as an SVG file
    pio.write_image(fig, file_name, format="svg", width=width, height=height, scale=3)

    # Display the chart
    # fig.show()


if __name__ == "__main__":
    data_type = "counterfactuals"
    if data_type == "squad":
        # load squad data
        dataset = load_dataset("squad", "plain_text")
        train_data = dataset["train"]
        data = [
            sample
            for sample in tqdm(
                train_data, total=len(train_data), desc="Loading SQuAD data ... "
            )
        ]
    elif data_type == "counterfactuals":
        # read in counterfactual data from jsonl file
        import jsonlines

        with jsonlines.open(
            f"{BASE_PATH}src/data/squad/flan_ul2_collated_data_with_answers_processed.jsonl"
        ) as reader:
            data = [
                sample
                for sample in tqdm(reader, desc="Loading Counterfactual data ... ")
            ]
    questions = [sample["question"] for sample in data]
    # print(len(questions))
    top_questions = topk_questions(questions, k=20)
    print(top_questions)

    # plot_question_types(questions, k=10, file_name="question_types_cf.svg")
