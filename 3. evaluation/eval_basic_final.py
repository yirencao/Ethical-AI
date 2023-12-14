import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import random
seed_value = 42
random.seed(seed_value)
# The model that you want to train from the Hugging Face hub
# original_model_name = "NousResearch/Llama-2-7b-chat-hf"
model_name = "NousResearch/Llama-2-7b-chat-hf"
data_names = ["ctang/HarmfulVsEthical_redteaming_eval_v3","ctang/HHH_redteaming_eval_v3"]  # "ctang/deon_eval_llama2_v3", 
for current_data_name in data_names:
    util_dataset = load_dataset(current_data_name)

    # Load the entire model on the GPU 0
    device_map = "auto"
    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = base_model


    # nb_eval_samples = 10
    total_samples = len(util_dataset['train'])
    import pandas as pd

    true_labels = []
    predicted_labels = []
    text_labels = []

    # Generating a unique filename using the current datetime
    from datetime import datetime
    output_directory = "./"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_results_{current_time}.txt"
    filepath = os.path.join(output_directory, filename)
    # file = open(filepath, 'w')
    # file.write("Evaluation Results\n\n")

    def generate_text(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=15)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Removing the prompt from the generated text
        cleaned_generated_text = generated_text[len(prompt):].strip()
        return cleaned_generated_text

    results = []
    print('dataset size',util_dataset['train'])
    for index in range(len(util_dataset['train'])):
        example = util_dataset['train'][index]
        prompt = example['prompt']
        response_a = example['response_a']
        response_b = example['response_b']
        preferred_response = example['more_reasonable'].lower()

        # Format the input prompt
        input_prompt = f"{prompt}\n{response_a}\n{response_b}\nThe more reasonable scenario is "
        generated_text = generate_text(input_prompt)

        # Assuming that generated_text starts with 'a' or 'b'
        generated_text_char = generate_text(input_prompt).strip().lower()[0]
        predicted_response = generated_text_char

        # Store the results
        true_labels.append(preferred_response)
        predicted_labels.append(predicted_response)
        text_labels.append(generated_text)
        
        # results.append({'index': index, 'true': preferred_response, 'predict': predicted_response})
        print({'index': index, 'true': preferred_response, 'predict': predicted_response})


    # Convert the true labels and predicted labels to numerical format
    true_labels_numeric = [0 if label == 'a' else (1 if label == "b" else -1) for label in true_labels]
    predicted_labels_numeric = [0 if label == 'a' else (1 if label == "b" else -2) for label in predicted_labels]

    # Calculate F1 Score and Accuracy
    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(true_labels_numeric, predicted_labels_numeric, average='weighted')
    accuracy = accuracy_score(true_labels_numeric, predicted_labels_numeric)

    print("\nF1 Score:", f1)
    print("Accuracy:", accuracy)

    # Create a DataFrame from the true and predicted labels
    results_df = pd.DataFrame({
        'True Labels': true_labels_numeric,
        'Predicted Labels': predicted_labels_numeric,
        'text': text_labels
    })

    # Generating a unique filename for the CSV using the current datetime
    csv_filename = f"eval_{model_name.split('/')[1]}_{current_data_name.split('/')[1]}_{current_time}.csv"

    # Save the DataFrame to a CSV file
    results_df.to_csv(csv_filename, index=False)

    print(f"Results saved to {csv_filename}")


    # file.write("\nF1 Score: {:.2f}\n".format(f1))
    # file.write("Accuracy: {:.2f}\n".format(accuracy))

    # print(f"File saved at: {filepath}")
    # file.close()
