# Synthetic_Data_OMOP_CDM
Project developed for the Msc course Applications of Data Science and Engineering as part of an internship @ Hospital da Luz Learning Health.

## Abstract

The potential of synthetic data in healthcare analytics, especially as a solution to privacy and data scarcity concerns, is gaining recognition. This project explores and compares various synthetic data generation techniques within this context. We focused on two datasets in the Observational Medical Outcomes Partnership (OMOP) Common Data Model (CDM) format, utilising a single table representation at the visit level. Our exploration included Generative Adversarial Networks (GANs) and fine-tuned Generative Pre-trained Trans- formers (GPTs), selected for their ability to handle complex data structures. To assess the fidelity of the synthetic data, we employed Log-Cluster and Kullback-Leibler (KL) Divergence between co-occurrence matrices of pairs of concepts metrics. We used prevalence plots for a visual dataset-level comparison. The results with these plots and the Log-Cluster metric were positive and aligned with the literature. However the results regarding the relations between concepts were not as impressive. A noteworthy outcome was observed with a GPT-2 model fine- tuned on our single table format, achieving a KL divergence score of 1.82 and a Log-Cluster score of -6.24. These findings, while divergent from expectations, provide valuable insights into the efficacy of different synthetic data generation methods in healthcare analytics.

## Content

- [`synth_env.yml`](synth_env.yml): all the necessary dependencies to run the project using Conda.
- [`report.pdf`](report.pdf): Project's Report
- [`poster.pdf`](report.pdf): Project's Poster
- [`abstract.pdf`](report.pdf): Project's Abstract
- [`library_test.ipynb`](library_test.ipynb): use examples of most of the functions created.
- [`conv2visit_level.py`](conv2visit_level.py): functions to convert the data from csv files to the visit level representation used for the patients
- [`methods.py`](methods.py): functions for the methods used to generate synthetic data in the visit level representation
- [`synth_metrics.py`](synth_metrics.py): functions to calculate the metrics used to evaluate the synthetic data

