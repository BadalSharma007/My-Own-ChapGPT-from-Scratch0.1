My Own ChatGPT from Scratch — v0.1

This repository contains a custom-built generative language model inspired by ChatGPT, developed and trained entirely from scratch using PyTorch. The training data comprises the public domain Sherlock Holmes corpus by Sir Arthur Conan Doyle. This project is a hands-on implementation of a lightweight GPT-style architecture aimed at understanding the foundational building blocks of large language models (LLMs).




🎯 Project Objective

The primary objective of this project is to:

Implement a transformer-based autoregressive language model (similar to GPT)
Train the model on a classical English text corpus
Understand the entire pipeline: preprocessing, model design, training, inference, and evaluation
Enable text generation based on user-provided prompts in the style of Sherlock Holmes




📂 Repository Structure

🔹 sherlock_holmes_canon.txt
A cleaned corpus consisting of multiple works from the Sherlock Holmes series.
Used as the training dataset.
Source: Public domain via Project Gutenberg
🔹 My_own_LLM1.ipynb
Contains:
Data preprocessing and tokenization
Transformer-based model architecture
Training loop and loss tracking
Model checkpoint saving
Framework: PyTorch
🔹 Text_generation_using_trained_model.ipynb
Demonstrates:
Loading of the trained model
Prompt-based text generation
Decoding strategies such as temperature sampling and top-k sampling





⚙️ How It Works

1. Preprocessing
Lowercasing, cleaning, and tokenizing the text into fixed-size sequences.
Character-level tokenization (or configurable based on implementation).
2. Model Architecture
A lightweight GPT-style transformer decoder.
Includes embedding layers, positional encoding, multi-head self-attention, and feed-forward layers.
3. Training
Optimizer: Adam
Loss: Cross-Entropy
Dataset: Sherlock Holmes corpus
Training tracks loss and saves checkpoints
4. Text Generation
Load model and tokenizer
Generate text one token at a time
Adjustable parameters:
temperature: Controls randomness
top_k: Filters most likely next tokens





🚀 Getting Started

🔧 Setup Instructions
Clone the repository
git clone https://github.com/BadalSharma007/My-Own-ChapGPT-from-Scratch0.1
cd My-Own-ChapGPT-from-Scratch0.1
Install dependencies
pip install torch numpy
Run training
Open My_own_LLM1.ipynb in Jupyter Notebook or VS Code
Execute all cells to train the model
Generate text
Open Text_generation_using_trained_model.ipynb
Enter your prompt and run to see model output




📌 Example Use Case

Prompt: "Sherlock Holmes walked into the dimly lit room and noticed..."
Model Output:
"...a faint trace of cigar ash on the carpet, a detail that would have escaped the average observer. Holmes knelt down, his fingers tracing the outline of a boot heel that pointed toward the open window..."





🧭 Future Enhancements

Integrate Byte Pair Encoding (BPE) or WordPiece tokenization
Expand dataset to include diverse authors and domains
Add attention weight visualization
Integrate HuggingFace model conversion for deployment
Explore Reinforcement Learning with Human Feedback (RLHF)



📄 License

This project uses only public domain content and is freely available for research and educational purposes.



👤 Author

Badal Kr. Sharma
B.Tech | AI/ML Research Enthusiast
