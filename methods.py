# .py file also created
#import conv2visit_level as c2v

import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch

import random
from collections import Counter

# takes data in a txt file
# also takes the model name and tokenizer 
def llm_finetuning(train_data_path, 
                   model = 'gpt2',
                   tokenizer = 'gpt2',
                   output_path = "model_gpt2_finetuned"): # pat to save model after fine tuning
    # Load pre-trained model tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', device_map = 'auto')
    model = GPT2LMHeadModel.from_pretrained('gpt2', device_map = 'auto')

    # Load dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_data_path,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Start training
    trainer.train()

    # save to ouput path
    trainer.save_model(output_path)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', device_map = 'auto')
    model = GPT2LMHeadModel.from_pretrained(output_path, device_map = 'auto')
    print(f"model and tokenizer save to {output_path} folder")

    return model, tokenizer

# function to generate text in the concept IDs format
def generate_text(model,
                  tokenizer,
                  seed_text="<GID> 8532 <YOB> 2031 <AGE> 71 <VCID> 581385 <VISIT>", 
                  max_length=200, 
                  top_p=0.5,
                  end_str = ']'):
    # Encode seed text to tokens
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # Create an attention mask for the inputs
    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')

    # Generate text with top_p sampling
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode generated tokens to text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Post-processing to ensure ending with "]"
    ending = end_str
    if ending in generated_text:
        # If ending is present, truncate at the last occurrence
        first_occurrence = generated_text.find(ending)
        generated_text = generated_text[:first_occurrence + len(ending)]
    else:
        # Append ending if not present
        generated_text += ending

    return generated_text


# fucntion to generate a synthetic dataset of size N
# demographic prompts are given to the fine tuned gpt2 model
# random sampling from the real data attributtes is done
# generated data is saved to txt file
def generate_synth_data_ids_format(df_visit, # NOT the visit_occ df, but the one created with conv2visit.py 
        model,
        tokenizer,
        N = 50,
        output_file_path = 'synthetic_records_p80_prompt_attribute.txt',
        format_type = 'codes'): # or text - weather the codes are in text or in numbers (codes)

    # Compute distributions
    gender_distribution = df_visit['gender_concept_id'].value_counts(normalize=True)
    yob_distribution = df_visit['year_of_birth'].value_counts(normalize=True)
    age_distribution = df_visit['age'].value_counts(normalize=True)
    vcid_distribution = df_visit['visit_concept_id'].value_counts(normalize=True)

    # Calculate the distribution of visit lengths
    #visit_length_distribution = df_visit_rep['formatted'].apply(lambda x: len(x.split(','))).value_counts(normalize=True)

    # Function to sample a demographic attribute
    def sample_attribute(distribution):
        return np.random.choice(distribution.index, p=distribution.values)

    # Function to generate a single record
    def generate_record(tokenizer):
        gender = sample_attribute(gender_distribution)
        yob = sample_attribute(yob_distribution)
        age = sample_attribute(age_distribution)
        vcid = sample_attribute(vcid_distribution)
        #visit_length = sample_attribute(visit_length_distribution)

        seed_text = f"<GID> {gender} <YOB> {yob} <AGE> {age} <VCID> {vcid} <VISIT>"

        # Calculate the number of tokens in the seed text
        num_seed_tokens = len(tokenizer.encode(seed_text))

        # Set max_length in terms of tokens
        #max_length_in_tokens = visit_length

        generated_text = generate_text(model, tokenizer, seed_text, max_length=200, top_p=0.8)
        return generated_text
    
    # Function to generate a single record
    def generate_record_simple_desc(tokenizer):
        gender = sample_attribute(gender_distribution)
        yob = sample_attribute(yob_distribution)
        age = sample_attribute(age_distribution)
        vcid = sample_attribute(vcid_distribution)
        #visit_length = sample_attribute(visit_length_distribution)
        gender_dict = {8507: 'Male',
                       8532: 'Female'}
        #visit_type_dict = c2v.get_visit_type_dict()

        seed_text = f"{gender_dict[gender]}, born in {yob}, aged {age} when admitted in a {visit_type_dict.get(vcid, 'Unknown visit type')},  had the following events: "

        # Calculate the number of tokens in the seed text
        num_seed_tokens = len(tokenizer.encode(seed_text))

        # Set max_length in terms of tokens
        #max_length_in_tokens = visit_length

        generated_text = generate_text(seed_text, max_length=200, top_p=0.8)
        return generated_text

    # Assuming 'tokenizer' is your GPT tokenizer
    
    if format_type == 'codes': 
        synthetic_records = [generate_record(tokenizer) for _ in range(N)]
    else:
        synthetic_records = [generate_record_simple_desc(tokenizer) for _ in range(N)]

    # Write the synthetic records to the file
    with open(output_file_path, 'w') as file:
        for record in synthetic_records:
            file.write(record + '\n')

    print(f"Synthetic records saved to {output_file_path}")

    return None



# function to generate synthetic data that just follows the statistical distribution of the population and samples from it
# takes a df in the patient_visit level representation format
# returns new df in the visit_rep format
def baseline_sample(df_visit, N_synth):

    gender_dist = df_visit['gender_concept_id'].value_counts(normalize=True)
    yob_dist = df_visit['year_of_birth'].value_counts(normalize=True)
    age_dist = df_visit['age'].value_counts(normalize=True)
    vcid_dist = df_visit['visit_concept_id'].value_counts(normalize=True)
    vlen_dist = df_visit['no_of_concepts'].value_counts(normalize=True)

    # Function to sample a demographic attribute
    def sample_attribute(distribution):
        return np.random.choice(distribution.index, p=distribution.values)

    # Concepts distribution

    all_lists_combined = sum(df_visit['visit'], [])

    # Assuming all_lists_combined is your list containing all values
    value_counts = Counter(all_lists_combined)
    total_counts = sum(value_counts.values())
    probabilities = {k: v / total_counts for k, v in value_counts.items()}

    # Convert the probabilities dictionary to lists for sampling
    values, probs = zip(*probabilities.items())

    df_visit_synth = pd.DataFrame(columns=df_visit.columns)
    records = []

    for i in range(N_synth):

        gender = sample_attribute(gender_dist)
        yob = sample_attribute(yob_dist)
        age = sample_attribute(age_dist)
        vcid = sample_attribute(vcid_dist)
        vlen = sample_attribute(vlen_dist)

        # sampling from the dist of concepts to create visit
        visit = random.choices(values, weights=probs, k = vlen)

        new_record = {'gender_concept_id': gender, 'year_of_birth': yob, 
                      'age': age, 'visit_concept_id': vcid, 'visit': visit,
                      'no_of_concepts': vlen}
        records.append(new_record)
        
    df_visit_synth = pd.DataFrame(records)
    
    #df_visit_synth_clean = df_visit_synth.dropna(axis=1, how='all')

    return df_visit_synth