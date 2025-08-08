# Project Title

Stock Price Time Series Generation using GAN

# Description

This project implements a Generative Adversarial Network (GAN) to generate synthetic stock price time series data based on historical data.

# Setup

To run this notebook, you need to have Python installed along with the libraries listed in the first code cell (pandas, numpy, matplotlib, seaborn, tensorflow, sklearn). Additionally, you will need to mount your Google Drive to access the `stock_data.csv` file.

# Data

The project uses historical stock price data stored in a CSV file named `stock_data.csv`. The key column used for time series generation is the 'Close' price.

## Code Explanation

### Preprocessing

The 'Close' price data is scaled using MinMaxScaler. The `prepare_data` function is used to create sequences of stock prices with a defined `time_step` (60) for training.

### Model

The GAN consists of a Generator and a Discriminator. The Generator is a sequential model with Dense, LeakyReLU, and BatchNormalization layers, outputting a time series sequence. The Discriminator is a sequential model with Dense and LeakyReLU layers and a sigmoid output layer for binary classification.

### Training

The GAN is trained in an adversarial manner. The Discriminator is trained to distinguish between real and generated stock price sequences, while the Generator is trained to produce sequences that can fool the Discriminator. Training is done iteratively on batches of data.

### Evaluation

The generated stock price sequences are evaluated visually by plotting them against real stock price sequences. The Mean Squared Error (MSE) is calculated between a real and a generated sequence to provide a quantitative measure of the difference.

# Results

Visual comparison of real and generated stock prices shows that the generated sequences capture some of the characteristics of real price movements. The Mean Squared Error (MSE) between a sample real and generated sequence is 3087.7745.

# Future Work

Potential future work includes experimenting with different GAN architectures (e.g., using LSTMs or GRUs in the generator and discriminator), tuning hyperparameters for better performance, and exploring different evaluation metrics for time series generation.
