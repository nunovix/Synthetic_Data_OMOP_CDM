# imports
import numpy as np
import pandas as pd
from datetime import timedelta
from collections import Counter
import re


# function to print names of null colluns in pandas df
def check_NaN(df):
    null_columns = df.columns[df.isnull().any()]

    # Calculate percentage of NaN values per column
    nan_percentage_per_column = df.isnull().mean() * 100

    print("Columns with null values:\n{}\n".format(null_columns.to_numpy()))
    print("Percentage of NaN values per column:\n{}".format(nan_percentage_per_column))

    return null_columns.to_numpy()


# convert dtype of columns from df to datetime if 'date' in name
def conv2dt(df):
    date_cols = df.filter(like='date').columns
    df[date_cols] = df[date_cols].apply(pd.to_datetime)
    return df


# takes path to folder with csv's of the OMOP-CDM format
# returns pandas df with visit level representation
def conv2visit_level_df(folder_path,
                        remove_empty_visits = True):
    # Load CSVs
    df_person = pd.read_csv(folder_path + 'person.csv')
    df_visit = pd.read_csv(folder_path + 'visit_occurrence.csv')
    df_drug = pd.read_csv(folder_path + 'drug_exposure.csv')
    df_condition = pd.read_csv(folder_path + 'condition_occurrence.csv')
    df_procedure = pd.read_csv(folder_path + 'procedure_occurrence.csv')

    # Convert date columns
    df_person = conv2dt(df_person)
    df_visit = conv2dt(df_visit)
    df_drug = conv2dt(df_drug)
    df_condition = conv2dt(df_condition)
    df_procedure = conv2dt(df_procedure)

    # Aggregate drug, condition, and procedure data
    agg_drug = df_drug.groupby('visit_occurrence_id')['drug_concept_id'].apply(list).reset_index()
    agg_condition = df_condition.groupby('visit_occurrence_id')['condition_concept_id'].apply(list).reset_index()
    agg_procedure = df_procedure.groupby('visit_occurrence_id')['procedure_concept_id'].apply(list).reset_index()

    # Merge visit and person details
    df_merged = df_visit.merge(df_person[['person_id', 'gender_concept_id', 'year_of_birth']], on='person_id', how='left')

    # Calculate age
    df_merged['age'] = df_merged['visit_start_date'].dt.year - df_merged['year_of_birth']

    # Merge aggregated data with df_merged
    df_merged = df_merged.merge(agg_drug, on='visit_occurrence_id', how='left')
    df_merged = df_merged.merge(agg_condition, on='visit_occurrence_id', how='left')
    df_merged = df_merged.merge(agg_procedure, on='visit_occurrence_id', how='left')

    # Replace NaNs with empty lists
    df_merged['drug_concept_id'] = df_merged['drug_concept_id'].apply(lambda d: d if isinstance(d, list) else [])
    df_merged['condition_concept_id'] = df_merged['condition_concept_id'].apply(lambda c: c if isinstance(c, list) else [])
    df_merged['procedure_concept_id'] = df_merged['procedure_concept_id'].apply(lambda p: p if isinstance(p, list) else [])

    # Combine concept IDs into a single 'visit' column
    df_merged['visit'] = df_merged.apply(lambda row: row['drug_concept_id'] + row['condition_concept_id'] + row['procedure_concept_id'], axis=1)

    if remove_empty_visits == True:
        # Remove rows with empty 'visit'
        df_merged = df_merged[df_merged['visit'].str.len() > 0]

    # Select specific columns
    final_df = df_merged[['person_id', 'visit_occurrence_id', 'gender_concept_id', 'year_of_birth', 'age', 'visit_concept_id', 'visit']]

    # Reset the index
    final_df = final_df.reset_index(drop=True)

    # add unique visit column and no of concepts and unique concepts
    final_df['no_of_concepts'] = final_df['visit'].apply(lambda x: len(x))
    final_df['unique_concepts'] = final_df['visit'].apply(lambda x: np.unique(np.array(x)))
    final_df['no_of_unique_concepts'] = final_df['unique_concepts'].apply(lambda x: len(x))

    return final_df

# convert id format to txt to be used in the gpt fine tuning
def conv2txt4training(df, 
                      output_file_path = 'formatted_data_for_training_patient_visit.txt',
                      visit_column = 'visit'):

    # Function to format a single record for GPT
    def format_record_visit(row):
        formatted = f"<GID> {row['gender_concept_id']} <YOB> {row['year_of_birth']} <AGE> {row['age']} <VCID> {row['visit_concept_id']} <VISIT> {row[visit_column]}"
        return formatted.replace("'", "")
    
    # Apply the formatting function to each row
    df['formatted'] = df.apply(format_record_visit, axis=1)

    # Write each formatted record to a new line in the output file
    with open(output_file_path, 'w') as file:
        for record in df['formatted']:
            file.write(record + "\n") 
    
    return df


# function to convert time delta to time tokens from 0 to 3 weeks, 1 to 11 months or Long Term (LT)
def delta2token(delta):
    if delta > timedelta(days = 365):
        token = 'LT'

    elif delta > timedelta(days = 28) and delta <= timedelta(days = 365):
        # Convert timedelta to months (approximation)
        average_days_per_month = 30.44
        months_approx = delta.total_seconds() / (average_days_per_month * 24 * 3600)
        # Round the approximate number of months to the nearest whole number
        token = 'M' + str(round(months_approx))

    elif delta < timedelta(days = 28):
        weeks_integer = delta.days // 7
        token = 'W' + str(weeks_integer)

    return token

# function to extract list of concepts that appear less than
# 'min_no_of_occ'
# to be used to filter te pat level view
def extract_list_of_concepts(folder_path,
                             min_no_of_occ = 10):

    # Load CSVs
    df_drug = pd.read_csv(folder_path + 'drug_exposure.csv')
    df_condition = pd.read_csv(folder_path + 'condition_occurrence.csv')
    df_procedure = pd.read_csv(folder_path + 'procedure_occurrence.csv')

    # Convert date columns
    df_drug = conv2dt(df_drug)
    df_condition = conv2dt(df_condition)
    df_procedure = conv2dt(df_procedure)

    concepts = []

    drug_list = df_drug['drug_concept_id'].tolist()
    condition_list = df_condition['condition_concept_id'].tolist()
    procedure_list = df_procedure['procedure_concept_id'].tolist()

    concepts.extend(drug_list)
    concepts.extend(condition_list)
    concepts.extend(procedure_list)

    # Assuming 'result' is your final list of integers
    counts = Counter(concepts)

    to_be_filtered = [num for num, count in counts.items() if count < min_no_of_occ]

    return to_be_filtered

# takes path to folder with csv's of the OMOP-CDM format
# returns pandas df with patient level representation
def conv2pat_level_df(folder_path,
                      remove_empty_visits = True,
                      filter_by = 10,
                      remove_empty_visits_filtered = True):
        
    # Load CSVs
    df_person = pd.read_csv(folder_path + 'person.csv')
    df_visit = pd.read_csv(folder_path + 'visit_occurrence.csv')
    df_drug = pd.read_csv(folder_path + 'drug_exposure.csv')
    df_condition = pd.read_csv(folder_path + 'condition_occurrence.csv')
    df_procedure = pd.read_csv(folder_path + 'procedure_occurrence.csv')

    # Convert date columns
    df_person = conv2dt(df_person)
    df_visit = conv2dt(df_visit)
    df_drug = conv2dt(df_drug)
    df_condition = conv2dt(df_condition)
    df_procedure = conv2dt(df_procedure)

    columns = ['person_id', 'gender_concept_id', 'year_of_birth', 'visits', 'no_of_visits', 
               'visit_types', 'no_of_concepts',
               'visit_with_unique_codes', 'no_of_concepts_in_visit_with_unique_codes',
               'visit_filtered', 'no_of_filtered_concepts',
               'filtered_visit_types',
               'visit_unique_filtered', 'no_of_filtered_unique_concepts',
               'no_filtered_visits']
    df_patient = pd.DataFrame(columns=columns)

    to_be_filtered = extract_list_of_concepts(folder_path, filter_by)

    # Sort the person IDs
    person_ids = df_person['person_id']
    data_list = []

    # Iterate through person_id values in sorted order
    for person_id in person_ids:

        # Search for matches in df_visit
        matches = df_visit[df_visit['person_id'] == person_id]

        # Sort the matches by visit_start_datetime
        matches = matches.sort_values(by='visit_start_datetime')

        # no. of visits
        no_visits = 0
        no_visits_filtered = 0

        # Create a list to store concept IDs for the current person
        concepts = ['VS']

        # Create a list to store UNIQUE concept IDs for the current person
        unique_concepts = ['VS']

        # no. of concepts per patient
        no_concepts = 0

        # no. of UNIQUE concepts per patient
        no_unique_concepts = 0

        # visit types
        visit_types = []
        filtered_visit_types = []

        # Iterate through each visit in matches
        for _, visit in matches.iterrows():
            
            time_diff = visit['visit_start_datetime'] 
            if concepts[-1] != 'VS':
                t_diff = visit['visit_start_datetime'] - e
                concepts.append(delta2token(t_diff))
                concepts.append('VS')

                unique_concepts.append(delta2token(t_diff))
                unique_concepts.append('VS')

                #tests
                #concepts.append(visit['visit_start_datetime'])
            #concepts.append(visit['visit_start_datetime'])
            
            visit_occurrence_id = visit['visit_occurrence_id']

            # Look up information in df_drug based on visit_occurrence_id
            drug_info = df_drug[df_drug['visit_occurrence_id'] == visit_occurrence_id]
            concepts.extend(drug_info['drug_concept_id'].tolist())
            no_concepts += len(drug_info)

            unique_drug = np.unique(np.array(drug_info['drug_concept_id'].tolist()))
            unique_concepts.extend(unique_drug)
            no_unique_concepts += len(unique_drug)
            
            # Look up information in df_condition based on visit_occurrence_id
            condition_info = df_condition[df_condition['visit_occurrence_id'] == visit_occurrence_id]
            concepts.extend(condition_info['condition_concept_id'].tolist())
            no_concepts += len(condition_info)

            unique_condition = np.unique(np.array(condition_info['condition_concept_id'].tolist()))
            unique_concepts.extend(unique_condition)
            no_unique_concepts += len(unique_condition)
            
            # Look up information in df_procedure based on visit_occurrence_id
            procedure_info = df_procedure[df_procedure['visit_occurrence_id'] == visit_occurrence_id]
            concepts.extend(procedure_info['procedure_concept_id'].tolist())
            no_concepts += len(procedure_info)

            unique_procedure = np.unique(np.array(procedure_info['procedure_concept_id'].tolist()))
            unique_concepts.extend(unique_procedure)
            no_unique_concepts += len(unique_procedure)

            after_filter_concepts = [item for item in concepts if item not in to_be_filtered]
            after_filter_unique_concepts = [item for item in unique_concepts if item not in to_be_filtered]

            if remove_empty_visits == False or concepts[-1] != 'VS':
                #concepts.append(visit['visit_end_datetime'])
                concepts.append('VE')
                unique_concepts.append('VE')
                e = visit['visit_end_datetime']
                no_visits += 1
                visit_types.append(visit['visit_concept_id'])
                #tests
                #concepts.append(e)
                #print(concepts)
            
            if remove_empty_visits_filtered == False or concepts[-1] != 'VS':
                #concepts.append(visit['visit_end_datetime'])
                after_filter_concepts.append('VE')
                after_filter_unique_concepts.append('VE')
                e = visit['visit_end_datetime']
                no_visits_filtered += 1
                filtered_visit_types.append(visit['visit_concept_id'])
        
        no_of_filtered_concepts = len([item for item in after_filter_concepts if not isinstance(item, str)])
        no_of_filtered_unique_concepts = len([item for item in after_filter_unique_concepts if not isinstance(item, str)])

        
        data = {'person_id': int(person_id), 
                'gender_concept_id': df_person.loc[df_person['person_id'] == person_id, 'gender_concept_id'].item(), 
                'year_of_birth': df_person.loc[df_person['person_id'] == person_id, 'year_of_birth'].item(), 
                'visits': concepts,
                'no_of_visits': no_visits,
                'visit_types': visit_types,
                'no_of_concepts': no_concepts,
                'visit_with_unique_codes': unique_concepts, 
                'no_of_concepts_in_visit_with_unique_codes': no_unique_concepts,
                'visit_filtered': after_filter_concepts, 
                'no_of_filtered_concepts': no_of_filtered_concepts,
                'filtered_visit_types': filtered_visit_types,
                'visit_unique_filtered': after_filter_unique_concepts, 
                'no_of_filtered_unique_concepts': no_of_filtered_unique_concepts,
                'no_filtered_visits': no_visits_filtered
                }

        data_list.append(data)

    df_patient = pd.concat([df_patient, pd.DataFrame(data_list)], ignore_index=True)

    return df_patient


# function to filter out concepts that happer less than 
# 'min_no_of_occ' times
def filter_visits(df, 
                  column = 'visit', # column to be filtered
                  min_no_of_occ = 10, # min no. of occ. for concept to be kept
                  remove_empty_visits = True):

    # Filtering out strings and concatenating lists
    result = [item for sublist in df['visit'] for item in sublist if isinstance(item, int)]

    # Assuming 'result' is your final list of integers
    counts = Counter(result)
    print(f"counts-->{counts}")

    to_be_filtered = [num for num, count in counts.items() if count < min_no_of_occ]
    print(f"to_be_filtered-->{to_be_filtered}")

    # Function to filter out values
    def filter_values(lst):
        return [item for item in lst if item not in to_be_filtered]

    # Apply the filter function to each list in the DataFrame column
    df['filtered_' + column] = df[column].apply(filter_values)

    # remove rows that end up with empty visits
    if remove_empty_visits:
        df = df[df['filtered_' + column].astype(bool)].copy()

    df['no_of_concepts_in_filtered_' + column] = df['filtered_' + column].apply(lambda x: len(x))

    df = df.reset_index(drop=True)

    return df


# extracts concepts dicts from ID's to descriptions
# from OHDSI from folder named
def get_concept_dict(concept_data_path = 'mimic_iv_OMOP_CDM/vocab/CONCEPT.csv'):

    df_concept = pd.read_csv(concept_data_path, on_bad_lines='skip', sep='\t')

    # dictionary to store the concept id's to concept names correspondence
    concept_dict = pd.Series(
    df_concept['concept_name'].values,
                index=df_concept['concept_id']
                ).to_dict()
    return concept_dict

# extracts domain dicts from ID's to domain name
# from OHDSI from folder named
def get_domain_dict(concept_data_path = 'mimic_iv_OMOP_CDM/vocab/CONCEPT.csv'):

    df_concept = pd.read_csv(concept_data_path, on_bad_lines='skip', sep='\t')

    # dictionary to store the concept id's to concept domain names correspondence
    domain_dict = pd.Series(
    df_concept['domain_id'].values,
                index=df_concept['concept_id']
                ).to_dict()
    return domain_dict

# visit type dictionary
#had to be crafted by hand due to problem with the convept vocab
# there are some missing values
def get_visit_type_dict():
    dict = {262:'Emergency Room and Inpatient Visit',
            8870:'Emergency Room - Hospital',
            8883:'Ambulatory Surgical Center',
            9201:'Inpatient Visit',
            9202: 'Outpatient Visit',
            9203: 'Emergency Room Visit',
            581385:'Observation Room',
            38004207:'Ambulatory Clinic / Center'}
    return dict

# convert visit column to format to be used by the llm
# using the concepts descriptions
def simple_format_descriptions(df, 
                               visit_column = 'visit',
                               output_file_path = 'trial_desc.txt' ):

    concept_dict = get_concept_dict()
    visit_type_dict = get_visit_type_dict()

    # Function to format a single record for GPT
    def format_record_visit(row):
        record = ''
        if row['gender_concept_id'] == 8507:
            record += 'Male'
        elif row['gender_concept_id'] == 8532:
            record += 'Female'
        
        record += ', born in '
        record += str(row['year_of_birth'])

        record += ', aged '
        record += str(row['age'])

        record += ' when admitted in a '
        record += str(visit_type_dict.get(row['visit_concept_id'], 'Unknown visit type'))

        record += ', had the following events: '
        record += str([concept_dict.get(item, 'Unknown') for item in row[visit_column]])
        #record.replace("[", "")
        #record.replace("]", "")
        record += '|'

        return record.replace("'", "")
    
    df['formatted'] = df.apply(format_record_visit, axis=1)
    #df['formatted'] = df['formatted'].apply(lambda x: x.replace("]", ""))
    #df['formatted'] = df['formatted'].apply(lambda x: x.replace("[", ""))

    df['No. of unknowns'] = df['formatted'].apply(lambda x: x.count('Unknown'))

    # Write each formatted record to a new line in the output file
    with open(output_file_path, 'w') as file:
        for record in df['formatted']:
            file.write(record + "\n") 

    return df

# reconvert from the simple description format
def reconvert_from_descriptions(file_path = 'trial_desc.txt'):

    def extract_data_from_line(line):
        gender_match = re.search(r'^(Male|Female)', line)
        year_of_birth_match = re.search(r'born in (\d+)', line)
        age_match = re.search(r'aged (\d+)', line)
        visit_type_match = re.search(r'when admitted in a (.*?),', line)

        if not all([gender_match, year_of_birth_match, age_match, visit_type_match]):
            return None

        gender = gender_match.group()
        year_of_birth = year_of_birth_match.group(1)
        age = age_match.group(1)
        visit_type = visit_type_match.group(1)

        # Adjusting event extraction to handle the new format
        events_part = re.search(r'\[(.*?)\]\|', line)
        if events_part:
            events = [event.strip() for event in events_part.group(1).split(',')]
        else:
            events = []

        return {
            'gender': gender,
            'year_of_birth': year_of_birth,
            'age': age,
            'visit_type': visit_type,
            'events': events
        }

    def invert_dictionary(dictionary):
        return {v: k for k, v in dictionary.items()}

    def apply_mappings(data, mappings):
        for key, mapping in mappings.items():
            if key in data:
                if isinstance(data[key], list):
                    data[key] = [mapping.get(event, event) for event in data[key]]
                else:
                    data[key] = mapping.get(data[key], data[key])
        return data

    # Replace 'your_file.txt' with the path to your actual text file
    extracted_data_list = []

    with open(file_path, 'r') as file:
        for line in file:
            extracted_data = extract_data_from_line(line)
            if extracted_data:
                extracted_data_list.append(extracted_data)

    # Example Dictionaries (Please replace with your actual dictionaries)
    gender_dict = {8507: 'Male', 8532: 'Female'}
    visit_type_dict = get_visit_type_dict()
    events_dict = get_concept_dict()

    # Inverting dictionaries
    inverted_gender_dict = invert_dictionary(gender_dict)
    inverted_visit_type_dict = invert_dictionary(visit_type_dict)
    inverted_events_dict = invert_dictionary(events_dict)

    # Applying mappings
    mapped_data_list = []
    for data in extracted_data_list:
        mappings = {
            'gender': inverted_gender_dict,
            'visit_type': inverted_visit_type_dict,
            'events': inverted_events_dict
        }
        mapped_data = apply_mappings(data, mappings)
        mapped_data_list.append(mapped_data)
    # Example of processed data
    for data in mapped_data_list[:5]:  # Display first 5 entries
        print(data)

    
    return mapped_data_list