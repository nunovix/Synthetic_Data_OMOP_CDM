# imports
import numpy as np
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import conv2visit_level as c2v
from sklearn.cluster import KMeans
from itertools import combinations
from collections import defaultdict
from scipy.stats import entropy


# Function to extract concept codes, GID, age, YOB, and visit length from a file
def extract_data_id_format(file_path):
    concept_codes = []
    gid_list = []
    age_list = []
    yob_list = []
    visit_length_list = []

    with open(file_path, 'r') as file:
        for line in file:
            if '<GID> 8507' in line or '<GID> 8532' in line:  # Your existing condition
                # Extracting GID
                gid_match = re.search(r'<GID>\s+(\d+)', line)
                if gid_match:
                    gid = int(gid_match.group(1))
                    gid_list.append(gid)

                # Extracting age
                age_match = re.search(r'<AGE>\s+(\d+)', line)
                if age_match:
                    age = int(age_match.group(1))
                    age_list.append(age)

                # Extracting YOB
                yob_match = re.search(r'<YOB>\s+(\d+)', line)
                if yob_match:
                    yob = int(yob_match.group(1))
                    yob_list.append(yob)

                # Extracting concept codes
                if '<VCID>' in line and '<VISIT>' in line:
                    codes = re.findall(r'\[([\d, ]+)\]', line)
                    codes = [code.strip() for c in codes for code in c.split(',')]
                    concept_codes.extend(codes)

                    # Calculating visit length
                    visit_length = len(codes)
                    visit_length_list.append(visit_length)

                vcid = re.findall(r"<VCID> (.*?) <VISIT>", line)
                vcid = [code.strip() for c in vcid for code in c.split(',')]
                concept_codes.extend(vcid)

    return gid_list, age_list, yob_list, concept_codes, visit_length_list


# Function to extract concept codes from a file
def extract_concept_codes_id_format(file_path):
    concept_codes = []
    with open(file_path, 'r') as file:
        for line in file:
            # Female Male condition
            if '<GID> 8507' in line or '<GID> 8532' in line:
                #print(f'line-->{line}')
                # Updated regex pattern to match multiple codes inside brackets
                codes = re.findall(r'\[([\d, ]+)\]', line)
                # Splitting multiple codes and stripping whitespace
                codes = [code.strip() for c in codes for code in c.split(',')]
                concept_codes.extend(codes)

                '''#adding vcid
                vcid = re.findall(r"<VCID> (.*?) <VISIT>", line)
                vcid = [code.strip() for c in vcid for code in c.split(',')]
                concept_codes.extend(vcid)'''
    return concept_codes

# function to save prevalance plots by domain to folder
# used here: https://www.ohdsi.org/2023showcase-503/
def prevalance_plots(df,
                     output_folder_path,
                     method_name):

    # Titles for the subplots
    titles = ['Procedure', 'Drug', 'Condition', 'Visit']

    # Specify the font size for axis labels
    label_fontsize = 12  # You can adjust this value as needed

    # Loop over each domain to create and save a separate scatter plot
    for i, domain in enumerate(titles):
        # Create a new figure for each subplot
        fig, ax = plt.subplots(figsize=(6, 6))

        # Filter the DataFrame for the current domain
        domain_df = df[df['domain'] == domain]

        # Create scatter plot
        ax.scatter(domain_df['Real_Prevalence'], domain_df['Synthetic_Prevalence'], alpha=0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Set limits for axes
        ax.set_xlim(1e-6, 1)
        ax.set_ylim(1e-6, 1)

        # Set labels with increased font size
        ax.set_xlabel('Real Data Prevalence', fontsize=label_fontsize)
        ax.set_ylabel('Synthetic Data Prevalence', fontsize=label_fontsize)

        # Grid and diagonal line
        ax.grid(True, which="both", ls="--")
        ax.plot([1e-6, 1], [1e-6, 1], 'k--')

        # Save the figure
        fig.savefig(f'{output_folder_path}{method_name}_prevalance_{domain}.png')
        print(f"{domain} plot saved to: {output_folder_path}{method_name}_prevalance_{domain}.png")
        # Close the figure to free memory
        plt.close(fig)
        print()

    return None
    
# function to calculate the log cluster metric 
# defined here: https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-020-00977-1
def log_cluster_metric(real_data, 
                       synthetic_data,
                       n_clusters = 6):
    
    # Convert them into Pandas DataFrames
    df_real = pd.DataFrame({'concepts': real_data})
    df_synthetic = pd.DataFrame({'concepts': synthetic_data})

    # Explode the concepts into separate rows for counting
    df_real_exploded = df_real.explode('concepts')
    df_synthetic_exploded = df_synthetic.explode('concepts')

    def create_feature_vectors(df):
        # Apply one-hot encoding and sum the counts
        feature_vectors = pd.get_dummies(df['concepts']).groupby(df.index).sum()
        return feature_vectors

    # Create feature vectors
    feature_vectors_real = create_feature_vectors(df_real_exploded)
    feature_vectors_synthetic = create_feature_vectors(df_synthetic_exploded)

    # Combine the feature vectors, adding a column to distinguish datasets
    combined_feature_vectors = pd.concat([
        feature_vectors_real.assign(dataset='real'),
        feature_vectors_synthetic.assign(dataset='synthetic')
    ]).fillna(0)  # Fill missing values with 0

    # Initialize k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    # Fit k-means on the data (excluding the 'dataset' label)
    clusters = kmeans.fit_predict(combined_feature_vectors.drop('dataset', axis=1))

    # Assign clusters back to the DataFrame
    combined_feature_vectors['cluster'] = clusters

    # Define the total counts needed for calculation
    total_real_samples = len(feature_vectors_real)
    total_samples = len(combined_feature_vectors)

    # Calculate c, the expected ratio of real samples
    c = total_real_samples / total_samples

    # Calculate the metric for each cluster
    cluster_metrics = []
    for cluster in range(n_clusters):
        cluster_subset = combined_feature_vectors[combined_feature_vectors['cluster'] == cluster]
        n_j = len(cluster_subset)
        n_j_R = len(cluster_subset[cluster_subset['dataset'] == 'real'])
        
        # Calculate the metric for this cluster and add it to the list
        if n_j > 0:
            cluster_metrics.append((n_j_R / n_j - c) ** 2)

    # Calculate the final Log Cluster metric
    U_c = np.log((1 / n_clusters) * sum(cluster_metrics))

    return U_c

# calculation of the KL divergence between co-occurrence matrices
# called in the all_stats function
def kl_co_occ_metric(real_data,
                     synthetic_data):
    
    def create_co_occurrence_matrix(visits, allowed_concepts):
        allowed_concepts = set(allowed_concepts)  # Ensure allowed_concepts is a set for O(1) lookups
        co_occurrence = defaultdict(lambda: defaultdict(int))

        for visit in visits:
            # Filter visit first
            filtered_visit = set(visit) & allowed_concepts
            for concept1, concept2 in combinations(filtered_visit, 2):
                co_occurrence[concept1][concept2] += 1
                co_occurrence[concept2][concept1] += 1  # Symmetric matrix

        # Convert to DataFrame
        co_occurrence_df = pd.DataFrame(co_occurrence).fillna(0)
        # Reindex to include all concepts
        co_occurrence_df = co_occurrence_df.reindex(index=allowed_concepts, columns=allowed_concepts, fill_value=0)
        
        return co_occurrence_df

    # Identify the concept IDs present in the real data
    real_concepts = set().union(*real_data)

    # Create co-occurrence matrices
    co_occurrence_real = create_co_occurrence_matrix(real_data, real_concepts)
    co_occurrence_synthetic = create_co_occurrence_matrix(synthetic_data, real_concepts)

    normalized_co_occurrence_real = co_occurrence_real / co_occurrence_real.sum().sum()
    normalized_co_occurrence_synthetic = co_occurrence_synthetic / co_occurrence_synthetic.sum().sum()

    # Flatten the matrices and add a small constant to avoid division by zero
    flat_real = normalized_co_occurrence_real.to_numpy().flatten() + 1e-9
    flat_synthetic = normalized_co_occurrence_synthetic.to_numpy().flatten() + 1e-9

    # Calculate KL-Divergence
    kl_divergence = entropy(flat_synthetic, flat_real)

    return kl_divergence

# function to save all the prevalance plots, calculate the log cluster and the kl divergence 
# metrics
# takes two txt files in the format with the concept ids
def all_stats_ids_format(real_data_path,
                         synthetic_data_path,
                         method_name,
                         output_folder_path,
                         prev_plots = True,
                         log_cluster = True,
                         KL_co_occ_div = True,
                         print_txt = True):
    
    # Extract concept codes from both datasets
    real_gid_list, real_age_list, real_yob_list, real_concept_codes, real_visit_length_list = extract_data_id_format(real_data_path)
    synthetic_gid_list, synthetic_age_list, synthetic_yob_list, synthetic_concept_codes, synthetic_visit_length_list = extract_data_id_format(synthetic_data_path)

    # Count the frequency of each concept code in both datasets
    real_code_counts = Counter(real_concept_codes)
    synthetic_code_counts = Counter(synthetic_concept_codes)

    # Creating a combined set of all unique codes from both datasets
    # this removes the fake codes that were generated
    all_codes = set(real_code_counts.keys())#.union(set(synthetic_code_counts.keys()))

    # Prepare the table with counts for each dataset
    concept_code_table = []
    for code in all_codes:
        real_count = real_code_counts.get(code, 0)
        synthetic_count = synthetic_code_counts.get(code, 0)
        concept_code_table.append([code, real_count, synthetic_count])

    # Convert the concept code table into a pandas DataFrame for easier plotting
    df = pd.DataFrame(concept_code_table, columns=['Concept_Code', 'Real_Count', 'Synthetic_Count'])

    # Calculate total number of concept codes in each dataset
    total_real_concepts = df['Real_Count'].sum()
    total_synthetic_concepts = df['Synthetic_Count'].sum()

    # Add normalized prevalence columns to the DataFrame
    df['Real_Prevalence'] = df['Real_Count'] / total_real_concepts
    df['Synthetic_Prevalence'] = df['Synthetic_Count'] / total_synthetic_concepts

    concept_dict = c2v.get_concept_dict()
    domain_dict = c2v.get_domain_dict()
    
    df['Concept_Code'] = df['Concept_Code'].apply(lambda x: int(x) if x != '' else None)
    df['concept_name'] = df['Concept_Code'].map(concept_dict, na_action='ignore')
    df['domain'] = df['Concept_Code'].map(domain_dict, na_action='ignore')

    if prev_plots == True:
        prevalance_plots(df, output_folder_path, method_name)
    
    # Extract concept codes from both datasets
    real_data = extract_concept_codes_id_format(real_data_path)
    synthetic_data = extract_concept_codes_id_format(synthetic_data_path)

    with open(f'{output_folder_path}{method_name}.txt', 'a') as file:
        file.write(f'{method_name}\n')

    if log_cluster == True:
        log_cluster_value = log_cluster_metric(real_data, synthetic_data)
        print(f"log_cluster_value-->{log_cluster_value}")
        with open(f'{output_folder_path}{method_name}.txt', 'a') as file:
            file.write(f"log_cluster_value-->{log_cluster_value}\n")

    if KL_co_occ_div == True:
        kl = kl_co_occ_metric(real_data, synthetic_data)
        print(f"kl-->{kl}")
        with open(f'{output_folder_path}{method_name}.txt', 'a') as file:
            file.write(f"kl-->{kl}\n")

    with open(f'{output_folder_path}{method_name}.txt', 'a') as file:
        file.write(f"\n---------------\n")

    return None