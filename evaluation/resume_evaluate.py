"""
This script evaluates the model-generated responses in the "Instruct_correct.csv" file using a structured prompt and saves the evaluation results in "evaluation_scores.csv". 
It also implements a checkpointing mechanism to resume evaluation from the last processed row in case of interruptions.


"""

from transformers import pipeline
import torch
import pandas as pd
from tqdm import tqdm
import re
import json
import os

CHECKPOINT_FILE = "checkpoint.txt"


def get_last_row():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return int(f.read().strip())
    return 0


def save_last_row(idx):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(idx))


def main(file_path="qalb-Instruct_correct.csv"):
    # Your evaluation code here

    df = pd.read_csv(file_path)

    validation_results = []

    model_id = "openai/gpt-oss-20b"

    start_idx = get_last_row()
    print(f"Resuming from row {start_idx}")

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    for idx, row in tqdm(df.iloc[start_idx:].iterrows(), total=len(df) - start_idx):

        prompt = f"""You are an expert evaluator of Urdu reasoning and explanation quality. Your task is to assess the quality of a model-generated response against a given question and reference answer.
        ### Input
            **Question:**
            {row['question']}

            **Ground Truth Answer:**
            {row['answer']}

            **Model Response:**
            {row['response']}

        ### Evaluation Criteria
        
        Evaluate the model response on the following dimensions using a 5-point Likert scale, where:
            * **1 = Poor**
            * **2 = Fair**
            * **3 = Satisfactory**
            * **4 = Good**
            * **5 = Excellent**

        #### 1. Correctness
        
        Assess whether the response provides the correct answer and whether factual, mathematical, or logical conclusions are accurate.

        #### 2. Reasoning
        
        Evaluate the coherence, validity, and step-by-step progression of the reasoning process. The reasoning should follow logically from the premises to the conclusion.

        #### 3. UrduLanguageFluency
        
        Assess the grammatical correctness, naturalness, readability, and linguistic quality of the Urdu text.

        #### 4. Clarity
        
        Evaluate how clearly the reasoning and conclusions are communicated. The response should be easy to understand and free from ambiguity.

        #### 5. Completeness
        
        Assess whether the response sufficiently addresses all aspects of the question and includes all necessary reasoning steps.

        ### Scoring Instructions
        For each criterion:
            * Assign an integer score from 1 to 5.

        ### Output Format
            Return a valid JSON object only no text
            {{
                "correctness": 1-5,
                "reasoning": 1-5,
                "UrduLanguageFluency": 1-5,
                "clarity": 1-5,
                "completeness": 1-5,
            }}
        """

        messages = [{"role": "user", "content": prompt}]
        outputs = pipe(messages, max_new_tokens=4096)
        result = outputs[0]["generated_text"][-1]["content"]
        print(result)
        print(f"Processed row {idx}")
        print("\n\n")
        validation_results.append({"row_idx": idx, "result": result})
        # Save immediately after successful processing
        save_last_row(idx + 1)
        # Also save results incrementally
        data = []
        for x in validation_results:
            # Find JSON after assistantfinal
            match = re.search(r"assistantfinal\s*(\{.*?\})", x["result"], re.DOTALL)
            if match:
                dat = json.loads(match.group(1))
                dat["row_idx"] = x["row_idx"]
                data.append(dat)
        scores_df = pd.DataFrame(
            data,
        )
        scores_df.to_csv(
            "qalb_evaluation_scores.csv",
            mode="a",
            index=False,
            header=not os.path.exists("qalb_evaluation_scores.csv"),
        )
        validation_results = (
            []
        )  # Clear after saving to avoid duplicates in next iteration


if __name__ == "__main__":
    main()
