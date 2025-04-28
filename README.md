# Synthetic Data Generation for Learning Analytics
Bachelor thesis project of Tobias Hartel
- Supervisor: Dr. Jakub Kuzilek
- Reviewer 1: Prof. Dr. Niels Pinkwart
- Reviewer 2: Prof. Dr. Gergana Vladova

## Project Overview
In this thesis project six different synthetic data generation (SDG) methods are evaluated using the three-dimensional evaluation approach proposed by Liu et al. [1], that encompasses resemblance, utility and privacy assessment.

The selected methods include [Synthpop non-parametric](https://synthpop.org.uk/index.html) [2], [DataSynthesizer](https://pypi.org/project/DataSynthesizer/) [3] and four methods from the [Synthetic Data Vault](https://sdv.dev/) (SDV) [4], namely GaussianCopula, CopulaGAN, TVAE and CTGAN. To assess the SDG methods the evaluation is conducted using five differently sized educational datasets. For more information on the specific datasets see the [_original\_data_](https://gitlab.informatik.hu-berlin.de/cses_students/bt-tobias-hartel/-/tree/main/scripts/data/original_data?ref_type=heads) folder. For each dataset a distinct jupyter notebook is created to carry out the evaluation and the results are accumulated one by one. 

## Usage
To carry out the evaluation from scratch, the existing generated datasets and result CSV files need to be deleted.

## Repository Structure

- **resources/:**
contains all the papers that were used as references.

- **scripts/:**
contains the original datasets, synthetic datasets, the evaluation notebooks and all of the generated data
    - **data/:** contains both the real and synthetic datasets and the evaluation results
        - **original_data/:** contains the original datasets and the respective train and test splits
            - 1_university_of_jordan/
            - 2_fictional_students_perfomance/
            - 3_edge_hill_university/
            - 4_open_university/
            - 5_portuguese_school/
        - **results/:** contains the temporary results
            - **plots/**
                - dcr/
                - mia/
            - **tables/**
        - **synthetic_data/:** contains the generated synthetic data
            - 1_university_of_jordan/
            - 2_fictional_students_perfomance/
            - 3_edge_hill_university/
            - 4_open_university/
            - 5_portuguese_school/
    - **final_results/:** contains the final results of all datasets after being merged
        - **combined_tables/**
        - **plots/**
            - **dcr/** 
            - **mia/**
    - **notebooks/:** contains the jupyter notebooks for each dataset and one notebook for merging the final results
    - **src/:** contains python scripts for evaluation and SDG
        - **evaluation/**
        - **generation/**

## References
[1] Qinyi Liu, Mohammad Khalil, Ronas Shakya, and Jelena Jovanovic. 2024.
Scaling While Privacy Preserving: A Comprehensive Synthetic Tabular
Data Generation and Evaluation in Learning Analytics. In The 14th Learning
Analytics and Knowledge Conference (LAK ’24), March 18–22, 2024, Kyoto,
Japan. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3636555.
3636921

[2] Nowok, B., G.M. Raab & C. Dibben (2016), synthpop: Bespoke creation of synthetic data in R. Journal of Statistical Software, 74:1-26; DOI:10.18637/jss.v074.i11. Available at: https://www.jstatsoft.org/article/view/v074i11

[3] Haoyue Ping, Julia Stoyanovich, and Bill Howe. 2017. DataSynthesizer:
Privacy-Preserving Synthetic Datasets. In Proceedings of SSDBM ’17, Chicago,
IL, USA, June 27-29, 2017, 5 pages.
DOI: http://dx.doi.org/10.1145/3085504.3091117

[4] N. Patki, R. Wedge and K. Veeramachaneni, "The Synthetic Data Vault," 2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA), Montreal, QC, Canada, 2016, pp. 399-410, doi: 10.1109/DSAA.2016.49.