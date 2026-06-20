# write a docstring
"""This script evaluates"""

from transformers import pipeline
import torch
import pandas as pd
from tqdm import tqdm
import re
import json


def main(file_path="../results/aqal-instruct_correct.csv"):
    # Your evaluation code here

    df = pd.read_csv(file_path)

    validation_results = []

    model_id = "openai/gpt-oss-20b"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    for _, row in tqdm(df.iterrows()):

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
            Return a valid JSON object
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
        print("\n\n")
        validation_results.append(result)

    data = []
    for x in validation_results:
        # Find JSON after assistantfinal
        match = re.search(r"assistantfinal\s*(\{.*?\})", x, re.DOTALL)
        if match:
            dat = json.loads(match.group(1))
            data.append(dat)

    scores_df = pd.DataFrame(data)
    scores_df.to_csv("results/aqal_evaluation_scores.csv", index=False)


def summarize_llm_judge_results():
    
    files = {
        "Alif-1.0-8B-Instruct": "alif_evaluation_scores.csv",
        "Qalb-1.0-8B-Instruct": "qalb_evaluation_scores.csv",
        "Riazi-8B": "riazi_evaluation_scores.csv",
        "Llama-3.1-8B-Instruct": "llama_evaluation_scores.csv",
    }
    metrics = [
        "correctness",
        "reasoning",
        "UrduLanguageFluency",
        "clarity",
        "completeness",
    ]

    rows = []


    for model, file in files.items():
        df = pd.read_csv(f"../results/{file}")

        row = {"Models": model}

        for metric in metrics:
            mean = df[metric].mean()
            std = df[metric].std()

            row[metric] = f"{mean:.2f} ± {std:.2f}"

        rows.append(row)

    results = pd.DataFrame(rows)
    return results


if __name__ == "__main__":
    main()
