# OperatorLearningBG
Operator Learning for Bubble Growth Dynamics

# File Structure

Sayan - LSTM + GRU analysis
Tarun - Seq2Seq analysis
Vivek - DeepONet analysis

Each folder has arch_m-value folders with respective analysis with varying m values.

For plots, each arch_m-value folder (eg: gru_20) has `predictions` folder which has 
plots for different l values (length scale of the Gaussian Random Field).

# How to run

To run DeepOnet:

```python
    python train_don.py
```

To run Seq2Seq:

```python
    python train_seq.py
```

To run LSTM/GRU:

```python
    python train.py LSTM 20
    python train.py GRU 20
```

To run analysis on the trained data:

For DeepONet
```python
    jupyter analyze.ipynb
```

For Seq2Seq
```python
    python analyze_seq.py
```

For LSTM/GRU
```python
    python analyze.py LSTM 20
    python analyze.py GRU 20
```