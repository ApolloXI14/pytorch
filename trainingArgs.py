from argspace import Namespace
"""
"Namespace" nicely encapsulates a property dictionary and works well
with static analyzers, and supports command-line training with
"ArgumentParser" 
"""
args = Namespace(
    #Data and path info
    frequency_cutoff=25,
    model_state_file="",
    save_dir="FILE_PATH",
    vectorizer_file="Vectorizer.py",
    #Model hyper params
    hidden_dim=300,
    #Training params
    batch_size=128,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=100,
    seed=1337
)
