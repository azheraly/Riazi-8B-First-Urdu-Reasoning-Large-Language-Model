# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "datasets==4.3.0",
#     "huggingface-hub==1.29.0",
#     "numpy==2.4.6",
#     "torch==2.11.0",
#     "tqdm==4.70.0",
#     "transformers==5.5.0",
#     "trl==0.24.0",
#     "unsloth==2026.8.22",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(auto_download=["html"])


@app.cell
def _():
    from huggingface_hub import notebook_login, list_repo_files, snapshot_download
    notebook_login()
    return list_repo_files, snapshot_download


@app.cell
def _():
    import os
    os.environ["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN' 
    return (os,)


@app.cell
def _():
    from unsloth import FastLanguageModel,is_bfloat16_supported
    import torch

    return FastLanguageModel, is_bfloat16_supported


@app.cell
def _(FastLanguageModel):

    lora_rank = 32  # Larger rank = smarter, but slower
    max_seq_length = 2048  # Choose any! We auto-support RoPE Scaling internally!
    dtype = (
        None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    )
    load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.
    load_in_8bit = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-8B",  # Choose ANY
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        fast_inference=True,  # Enable vllm fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.9,  # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank * 2,  # *2 speedataset up training
        use_gradient_checkpointing="unsloth",  # Reduces memory usage
        random_state=3407,
    )
    return (
        dtype,
        load_in_4bit,
        load_in_8bit,
        lora_rank,
        max_seq_length,
        model,
        tokenizer,
    )


@app.cell
def _(model):
    model
    return


@app.cell
def _():
    from datasets import load_dataset

    ds = load_dataset("large-traversaal/urdu-instruct", split="train")
    ds = ds.filter(lambda x: x["category"] == "reasoning")



    def extract_hash_answer(text):
        if "Answers:" in text:
            return text.split("Answers:", 1)[1].strip()

        if "Answer:" in text:
            return text.split("Answer:", 1)[1].strip()

        if "Answer" in text:
            return text.split("Answer", 1)[1].strip()

        return None



    reasoning_start = "<reasoning>"
    reasoning_end   = "</reasoning>"
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"

    system_prompt = f"""
    You are given a problem.

    Think about the problem and provide your working out.
    Place your reasoning between {reasoning_start} and {reasoning_end}.

    Then, provide your final solution between
    {solution_start} and {solution_end}.

    Always answer in Urdu.
    """
    system_prompt
    ds= ds.map(lambda x: {
        "prompt" : [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": x["instruction"]},
        ],
        "answer": extract_hash_answer(x["output"]),
    })
    return (
        ds,
        load_dataset,
        reasoning_end,
        reasoning_start,
        solution_end,
        solution_start,
        system_prompt,
    )


@app.cell
def _(re, reasoning_end, reasoning_start, solution_end, solution_start):
    match_format = re.compile(
        rf"^[\s]{{0,}}"\
        rf"{reasoning_start}.+?{reasoning_end}.*?"\
        rf"{solution_start}(.+?){solution_end}"\
        rf"[\s]{{0,}}$",
        flags = re.MULTILINE | re.DOTALL
    )
    return (match_format,)


@app.cell
def _(match_format):
    match_format.search(
        "<reasoning>Let me think!</reasoning>"\
        "<SOLUTION>2</SOLUTION>",
    )
    return


@app.cell
def _(match_format, reasoning_end, solution_end, solution_start):
    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            # Match if format is seen exactly!
            if match_format.search(response) is not None: score += 3.0
            scores.append(score)
        return scores

    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            # Count how many keywordataset are seen - we penalize if too many!
            # If we see 1, then plus some points!

            # No need to reward the opening tag since we always prepend it!
            # score += 0.5 if response.count(reasoning_start) == 1 else -1.0
            score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
            score += 0.5 if response.count(solution_start)  == 1 else -1.0
            score += 0.5 if response.count(solution_end)    == 1 else -1.0
            scores.append(score)
        return scores

    return match_format_approximately, match_format_exactly


@app.cell
def _(match_format):
    def check_answer(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := match_format.search(r)) is not None else None \
            for r in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if guess is None:
                scores.append(-2.0)
                continue
            # Correct answer gets 5 points!
            if guess == true_answer:
                score += 5.0
            # Match if spaces are seen, but less reward
            elif guess.strip() == true_answer.strip():
                score += 3.5
            else:
                # We also reward it if the answer is close via ratios!
                # Ie if the answer is within some range, reward it!
                try:
                    ratio = float(guess) / float(true_answer)
                    if   ratio >= 0.9 and ratio <= 1.1: score += 2.0
                    elif ratio >= 0.8 and ratio <= 1.2: score += 1.5
                    else: score -= 2.5 # Penalize wrong answers
                except:
                    score -= 4.5 # Penalize
            scores.append(score)
        return scores

    return (check_answer,)


@app.cell
def _(re, solution_start):
    match_numbers = re.compile(
        solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
        flags = re.MULTILINE | re.DOTALL
    )
    print(match_numbers.findall("<SOLUTION>  0.34  </SOLUTION>"))
    print(match_numbers.findall("<SOLUTION>  123,456  </SOLUTION>"))
    print(match_numbers.findall("<SOLUTION>  -0.234  </SOLUTION>"))
    print(match_numbers.findall("<SOLUTION>17</SOLUTION>"))
    return (match_numbers,)


@app.cell
def _(match_numbers):
    global PRINTED_TIMES
    PRINTED_TIMES = 0
    global PRINT_EVERY_STEPS
    PRINT_EVERY_STEPS = 5

    def check_numbers(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := match_numbers.search(r)) is not None else None \
            for r in responses
        ]

        scores = []
        # Print only every few steps
        global PRINTED_TIMES
        global PRINT_EVERY_STEPS
        if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
            print(
                '*'*20 + f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}"
            )
        PRINTED_TIMES += 1

        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(-2.5)
                continue
            # Convert to numbers
            try:
                true_answer = float(true_answer.strip())
                # Remove commas like in 123,456
                guess       = float(guess.strip().replace(",", ""))
                scores.append(3.5 if guess == true_answer else -1.5)
            except:
                scores.append(0)
                continue
        return scores

    return PRINTED_TIMES, PRINT_EVERY_STEPS, check_numbers


@app.cell
def _(ds, tokenizer):
    max(ds.map(
        lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
        batched = True,
    ).map(lambda x: {"length" : len(x["tokens"])})["length"])
    return


@app.cell
def _(ds):
    dataset = ds.filter(lambda x: x["answer"] is not None)
    return (dataset,)


@app.cell
def _(is_bfloat16_supported, max_seq_length):
    max_prompt_length = 290 # + 1 just in case!

    from trl import GRPOConfig, GRPOTrainer
    training_args = GRPOConfig(
        learning_rate = 5e-6,
        weight_decay = 0.001,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 50,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Increase to 4 for smoother training
        num_generations = 4, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        num_train_epochs = 1, # Set to 1 for a full training run
        # max_steps = 10,
        save_strategy="steps",
        save_steps = 200,
        save_total_limit=2,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        max_grad_norm = 1.0,
        report_to = "none", # Can use Weights & Biases
        output_dir = "Riazi-8B-GRPO-Lora",
        push_to_hub=True,
        hub_strategy="checkpoint",
    )
    return GRPOTrainer, training_args


@app.cell
def _(dataset):
    dataset
    return


@app.cell
def _(
    GRPOTrainer,
    check_answer,
    check_numbers,
    dataset,
    list_repo_files,
    match_format_approximately,
    match_format_exactly,
    model,
    snapshot_download,
    tokenizer,
    training_args,
):
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
        args = training_args,
        train_dataset = dataset
        ,
    )

    print(f"Start training")

    try:
        repo_id = "azherali/Riazi-8B-GRPO-Lora"  # Change to your repo ID
        files = list_repo_files(repo_id)
        checkpoints = sorted(
            [f for f in files if f.startswith("last-checkpoint")],
        )
        if len(checkpoints) == 0:
            print("No checkpoints — starting new training.")
            trainer_stats = trainer.train()
        else:
            local_dir = snapshot_download(repo_id, allow_patterns="last-checkpoint/*")
            print("Resuming from:", local_dir)
            trainer_stats = trainer.train(
                resume_from_checkpoint=f"{local_dir}/last-checkpoint"
            )
    except Exception as e:
        print("Repo not found or empty — starting fresh training.", e)
        trainer_stats = trainer.train()
    return


@app.cell
def _():
    # model.push_to_hub("azherali/Riazi-8B-GRPO-Lora")
    # tokenizer.push_to_hub("azherali/Riazi-8B-GRPO-Lora")
    return


@app.cell
def _():
    # model.push_to_hub_merged("azherali/Riazi-8B-GRPO", tokenizer, save_method = "merged_16bit")
    return


@app.cell
def _():
    return


@app.cell
def _(
    FastLanguageModel,
    dtype,
    load_in_4bit,
    load_in_8bit,
    lora_rank,
    max_seq_length,
):

    model1, tokenizer1 = FastLanguageModel.from_pretrained(
        model_name="azherali/Riazi-8B-Instruct",  # Choose ANY
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        fast_inference=True,  # Enable vllm fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.9,  # Reduce if out of memory
    )
    return model1, tokenizer1


@app.cell
def _(FastLanguageModel, model1, system_prompt, tokenizer1):
    FastLanguageModel.for_inference(model1)  # Enable native 2x faster inference

    messages = [
         {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "پانچ بچوں نے 20 چاکلیٹس برابر بانٹیں۔ ہر بچے کو کتنی چاکلیٹس ملیں گی؟"
        }
    ]
    text = tokenizer1.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        # Must add for generation
    )

    from transformers import TextStreamer

    _ = model1.generate(
        **tokenizer1(text, return_tensors="pt").to("cuda"),
        temperature=0.6,
        top_p=0.95,
        top_k=20,  # For non thinking
        streamer=TextStreamer(tokenizer1, skip_prompt=True),
    )
    return


@app.cell
def _(model1, system_prompt, tokenizer1):
    from vllm import SamplingParams
    def generate_answer(problem):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": problem},
        ]

        text = tokenizer1.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
            tokenize = False,
            enable_thinking=False,
        )
        sampling_params = SamplingParams(
            max_tokens=2048,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )
        output = model1.fast_generate(
            text,
            sampling_params = sampling_params,
            lora_request =None,
        )[0].outputs[0].text
        return output


    return (generate_answer,)


@app.cell
def _(generate_answer):
    print(generate_answer("پانچ بچوں نے 20 چاکلیٹس برابر بانٹیں۔ ہر بچے کو کتنی چاکلیٹس ملیں گی؟"))
    return


@app.cell
def _():
    from tqdm import tqdm

    return (tqdm,)


@app.cell
def _(os):
    CHECKPOINT_FILE = "checkpoint.txt"


    def get_last_row():
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, "r") as f:
                return int(f.read().strip())
        return 0


    def save_last_row(idx):
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(str(idx))

    return get_last_row, save_last_row


@app.cell
def _(load_dataset):
    test_data = load_dataset("large-traversaal/mgsm_urdu_final", split="test")
    df = test_data.to_pandas()
    return (df,)


@app.cell
def _():
    import pandas as pd
    from transformers import pipeline


    return pd, pipeline


@app.cell
def _(df, generate_answer, get_last_row, os, pd, save_last_row, tqdm):

    start_idx = get_last_row()
    print(f"Resuming from row {start_idx}")
    for idx, row in tqdm(df.iloc[start_idx:].iterrows(), total=len(df) - start_idx):
        response = generate_answer(row["urdu_question"])
        print(f"Processed row {idx}")
        print(f"question: {row['urdu_question']}")
        print("\n\n")
        results = []
        results.append(
            {
                "row_idx": idx,
                "question": row["urdu_question"],
                "answer": row["answer_number"],
                "response": response,
            }
        )
        # Save immediately after successful processing
        save_last_row(idx + 1)
        scores_df = pd.DataFrame(
            results,
        )
        scores_df.to_csv(
            "riazi-8b-grpo.csv",
            mode="a",
            index=False,
            header=not os.path.exists("riazi-8b-grpo.csv"),
        )
    return


@app.cell
def _():
    return


@app.cell
def _(pd, pipeline):
    data_df = pd.read_csv("qwen3-8b_grpo_correct.csv")

    validation_results = []

    model_id = "openai/gpt-oss-20b"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    # for _, rowa in data_df.iterrows():

    #     prompt = f"""You are an answer validator for Urdu math reasoning problems.

    #     Input:

    #     * Question: {rowa['question']}
    #     * Solution: {rowa['response']}
    #     * Expected Answer: {rowa['answer']}

    #     Task:

    #     1. Read the Urdu math problem and the provided solution.
    #     2. Determine the final answer produced by the solution.
    #     3. Compare it with the Expected Answer.
    #     4. Treat numerically equivalent values as equal:

    #        * 2 = 2.0 = 2.00
    #        * 5 = 5.000
    #        * 5.34 = 5.00
    #        * Ignore insignificant trailing zeros in decimals.
    #     5. If both answers represent the same numeric value, return:
    #        True
    #     6. Otherwise return:
    #        False

    #     Rules:

    #     * Return ONLY one word: True or False.
    #     * Do not provide explanations, reasoning, calculations, or extra text.
    #     * Comparison should be based on the final answer only.
    #     * For numeric answers, compare by value, not string format.
    #     * If the solution's final answer cannot be determined with confidence, return False.
    #      """

    #     message = [{"role": "user", "content": prompt}]
    #     outputs = pipe(message, max_new_tokens=1000)
    #     result = (
    #         True
    #         if "True.assistantfinalTrue" in outputs[0]["generated_text"][-1]["content"]
    #         else False
    #     )
    #     validation_results.append(result)

    # data_df["is_correct"] = validation_results

    # data_df.to_csv("qwen3-8b_grpo_correct.csv", index=False)

    # print(data_df.head())
    return data_df, pipe


@app.cell
def _(data_df, pipe, tqdm):
    import json
    validationresults = []
    for _, row in tqdm(data_df.iterrows()):

        prompts = f"""You are an expert evaluator of Urdu reasoning and explanation quality. Your task is to assess the quality of a model-generated response against a given question and reference answer.
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

        messages = [{"role": "user", "content": prompts}]
        outputss = pipe(messages, max_new_tokens=4096)
        results = outputss[0]["generated_text"][-1]["content"]
        print(row['question'])
        print("\n\n")
        print(results)
        print("\n\n")
        validationresults.append(results)

    return json, validationresults


@app.cell
def _(validationresults):
    len(validationresults)
    return


@app.cell
def _(json, pd, validationresults):

    import re


    data = []

    for x in validationresults:

        match = re.search(
            r'\{\s*"correctness"\s*:\s*\d+.*?"completeness"\s*:\s*\d+\s*\}',
            x,
            re.DOTALL
        )

        if match:
            try:
                dat = json.loads(match.group(0))
                data.append(dat)
            except json.JSONDecodeError:
                print("Invalid JSON:")
                print(match.group(0))
        else:
            print("No JSON found:")
            print(x)

    scores_df = pd.DataFrame(data)
    scores_df.to_csv("evaluation_scores.csv", index=False)
    return (re,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
