# Beyond Consensus: Mitigating the Agreeableness Bias in LLM Judge Evaluations

This repository contains the artifacts for our paper on mitigating the positive bias of LLM-as-a-judge systems. While LLMs are good at identifying correct outputs (True Positives), they are poor at identifying incorrect ones (True Negatives). In addition to empirically quantifying the actual bias and showing that minority-veto ensemble outperforms majority consensus, we propose a novel regression-based framework that models this bias to provide more accurate and reliable evaluations, reducing evaluation error by 2x compared to state-of-the-art ensemble methods.

## Publication

If you find our research or dataset useful, kindly consider citing it using the BibTeX entry below:

```bibtex
@misc{jain2025consensusmitigatingagreeablenessbias,
      title={Beyond Consensus: Mitigating the Agreeableness Bias in LLM Judge Evaluations}, 
      author={Suryaansh Jain and Umair Z. Ahmed and Shubham Sahai and Ben Leong},
      year={2025},
      eprint={2510.11822},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.11822}, 
}
```

## Prerequisites

This project needs Git and Docker as pre-requisites to run.

### Install Docker [Ubuntu]

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to the docker group (optional, to run docker without sudo)
sudo usermod -aG docker $USER

# Restart your session or run:
newgrp docker
```

### Install Git and Git LFS

```bash
# Install Git
sudo apt update
sudo apt install git

# Install Git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install git-lfs

# Initialize Git LFS
git lfs install
```

## Setup Instructions

1. **Clone the repository and download LFS files:**

    ```bash
    git clone https://github.com/ai-cet/llm-judge-calibration.git
    cd llm-judge-calibration
    git lfs pull
    ```

2. **Extract the dataset:**

    ```bash
    tar -xzvf data.tar.gz
    ```

3. **Build the Docker image:**

    ```bash
    docker build -t llm-benchmarking -f docker/Dockerfile .
    ```

4. **Run the Docker container:**

    ```bash
    docker run --rm -it \
      -v $(pwd)/output:/app/output \
      -v $(pwd)/logs:/app/logs \
      -v $(pwd)/src:/app/src \
      llm-benchmarking
    ```

5. **Inside the Docker container, run the LLM-as-judge and ensemble:**

    ```bash
    F=0 C=0 N=0 H=0 P=0 python -m src.validation.validator && \
    F=1 C=1 N=0 H=0 P=0 python -m src.validation.validator && \
    F=1 C=1 N=1 H=1 P=0 python -m src.validation.validator && \
    F=1 C=1 N=1 H=1 P=1 python -m src.validation.validator
    ```

6. **Inside the Docker container, run the regression script:**

    ```bash
    {
        F=0 C=0 N=0 H=0 P=0 python -m src.regression.regressor && \
        F=1 C=1 N=0 H=0 P=0 python -m src.regression.regressor && \
        F=1 C=1 N=1 H=1 P=0 python -m src.regression.regressor && \
        F=1 C=1 N=1 H=1 P=1 python -m src.regression.regressor
    } > regression.log 2>&1 &
    ```

7. **Generate plots**

    ```bash
    F=0 C=0 N=0 H=0 P=0 python -m src.plots.plots &&
    F=1 C=1 N=1 H=1 P=1 python -m src.plots.plots
    ```
