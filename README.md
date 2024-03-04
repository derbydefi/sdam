# sdam
Sparse Distributed Associative Memory (SDAM) Model
Overview
This repository hosts an experimental implementation of a Sparse Distributed Associative Memory (SDAM) model. This project is inspired by foundational concepts in SDM and Hebbian learning theories, aiming to explore and enhance the handling of semantic relationships and episodic data through sparse and distributed representations.

The current model is a conceptual exploration, intending to push the boundaries of how memory systems can be designed to mimic certain aspects of human cognition, specifically the encoding, storage, and retrieval of semantic and episodic information.

Inspiration
The inspiration for this project comes from the rich domain of neural and memory models, particularly:

Sparse Distributed Memory (SDM): A theoretical framework for how memories might be stored and retrieved in the brain, emphasizing the distributed nature of memory storage across neural networks.
Hebbian Learning Principles: A fundamental concept in neural network theory, encapsulating the idea that neurons that fire together wire together, forming the basis for learning and memory association.
Our model attempts to marry these concepts with modern computational techniques and data structures, recognizing the limitations of current models and the potential for new discoveries in the realm of artificial memory systems.

Experimental Status
This model is experimental and in its early stages of development. It currently utilizes basic numpy operations for its implementation, with considerations for future optimizations and migration to more efficient data structures and processing frameworks (e.g., PyTorch for GPU acceleration or sparse matrix representations).

Key Features:

Ternary logic representation for addresses and contents, including 'don't care' states.
Dynamic adjustment of Hamming radii for memory retrieval, balancing specificity and generalization.
Decay mechanisms to simulate forgetting, emphasizing the transient nature of episodic memory.
Experimental encoding of semantic links and sequence-based information to explore the structure of complex memory relationships.
Goals for Sharing
The primary goal of sharing this project is to foster collaboration and discussion within the research community. We are particularly interested in:

Feedback on the model’s design and its alignment with cognitive theories.
Ideas for optimization and scalability, making the model more practical for large-scale experiments.
Exploration of potential applications, from simulating aspects of human memory to novel machine learning tasks.
Contributing
As this is a research project, contributions of all forms are welcomed, from theoretical insights and literature suggestions to code optimizations and experimental use cases. Please feel free to fork this repository, experiment with the model, and share your findings or suggestions through issues and pull requests.

Future Directions
The next steps for this project involve:

Optimization and efficiency improvements, considering Cython or GPU-accelerated processing.
Exploration of sparse data structures for more efficient memory representation.
Extended experiments to validate and refine the model’s capabilities in simulating memory processes.
We invite researchers, enthusiasts, and critics alike to engage with this project, challenge its assumptions, and contribute to its evolution.
