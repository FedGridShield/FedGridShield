# FedGridShield🛡️

In this repo, we introduce a scalable framework, FedGridShield, for studying attacks and defenses in federated learning for smart grid systems.

## Dataset Describption

We provide two datasets in our framework as examples.

One task is Electricity Theft Detection. We use data from the paper "Privacy-Preserving Electricity Theft Detection based on Blockchain," published in IEEE Transactions on Smart Grid in 2023. Please refer to [their repository](https://github.com/Lanren9/Electricity-Theft-Detection) to download the `balance_data.csv` file and place it under the `your_path_to_repo/FedGridShield/data/dataset/electricity_theft_detection` directory. This is a tabular dataset with 48 features, and the binary label defines whether electricity theft is detected.

Another task is Generator Defect Classification. We use a dataset of generator images in perfect and defective conditions. This dataset is designed for training an image classifier to indicate the generator's condition, which can help with system maintenance. The training data contains 766 images of generators in perfect condition and 744 images of generators in defective condition. The test data contains 275 and 300 images for these conditions, respectively. You can download the dataset from [this link](https://drive.google.com/file/d/1fZ37PSSCYjQdJkE8Ak4ToSCtnyRxpX2s/view?usp=sharing) for full-size data (1024x1024) or [the link](https://drive.google.com/file/d/1sXlAJw7GioanVht27Uw0O5hhmHT9il2b/view?usp=sharing) for resized data (64x64). Place the downloaded data under the `your_path_to_repo/FedGridShield/data/dataset/generator_defect_classification` directory.
## How to use

Use `requirements.yml` to install necessary packages

```python
# Run a demo
python main.py 
```

```bash
# or use the shell scripts (change gpu ids accordingly) to run the whole experiments
bash run_theft_detection.sh
bash run_defect_classification.sh
```

## Performance

Electricity Theft Detection

|  Attack/Defense     | no defense | trmean | median | krum   | bulyan | flame  | dpfed  | sparsefed |
|---------------------|------------|--------|--------|--------|--------|--------|--------|-----------|
| no attack           | 90.65      | 88.30  | 90.65  | 87.96  | 88.31  | 50.29  | 51.13  | 85.24     |
| Fang (Attack=20%)   | 50.25      | 86.17  | 50.25  | 87.10  | 88.18  | 50.29  | 50.25  | 50.25     |
| AGR (Attack=20%)    | 50.25      | 85.44  | 50.25  | 87.42  | 88.21  | 50.25  | 50.25  | 50.25     |
| Fang (Attack=40%)   | 87.29      | 73.61  | 87.29  | 53.19  | 88.53  | 64.51  | 51.33  | 71.17     |
| AGR (Attack=40%)    | 75.69      | 86.17  | 75.69  | 87.63  | 86.86  | 64.82  | 51.44  | 76.35     |

Generator Defect Classification

|                     | no defense | trmean | median | krum   | bulyan  | flame  | dpfed  | sparsefed |
|---------------------|------------|--------|--------|--------|---------|--------|--------|-----------|
| no attack           | 75.82      | 77.56  | 89.36  | 78.72  | 76.40   | 55.31  | 53.19  | 90.13     |
| Fang (Attack=20%)   | 53.19      | 81.04  | 53.19  | 77.75  | 76.40   | 53.19  | 53.19  | 53.19     |
| AGR (Attack=20%)    | 53.19      | 80.27  | 53.19  | 77.75  | 76.01   | 53.19  | 53.19  | 53.19     |
| Fang (Attack=40%)   | 66.53      | 48.35  | 66.53  | 53.19  | 53.19   | 55.70  | 53.19  | 53.19     |
| AGR (Attack=40%)    | 53.19      | 53.19  | 53.19  | 54.93  | 64.41   | 53.96  | 53.19  | 53.19     |

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).  
You are free to use, modify, and share it, but must provide appropriate credit.  
For citation, please refer to the following:

```
@article{zhang2024federated,
  title={Federated Learning for Smart Grid: A Survey on Applications and Potential Vulnerabilities},
  author={Zhang, Zikai and Rath, Suman and Xu, Jiaohao and Xiao, Tingsong},
  journal={arXiv preprint arXiv:2409.10764},
  year={2024}
}
```