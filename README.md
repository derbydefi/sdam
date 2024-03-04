# Sparse Distributed Associative Memory (SDAM) Model

## Overview

This repository hosts an experimental implementation of a Sparse Distributed Associative Memory (SDAM) model. This project is inspired by foundational concepts in SDM and Hebbian learning theories, aiming to explore and enhance the handling of semantic relationships and episodic data through sparse and distributed representations.

The current model is a conceptual exploration, intending to push the boundaries of how memory systems can be designed to mimic certain aspects of human cognition, specifically the encoding, storage, and retrieval of semantic and episodic information.

## Inspiration

The inspiration for this project comes from the rich domain of neural and memory models, particularly:
- **Sparse Distributed Memory (SDM):** A theoretical framework for how memories might be stored and retrieved in the brain, emphasizing the distributed nature of memory storage across neural networks.
- **Hebbian Learning Principles:** A fundamental concept in neural network theory, encapsulating the idea that neurons that fire together wire together, forming the basis for learning and memory association.

Our model attempts to marry these concepts with modern computational techniques and data structures, recognizing the limitations of current models and the potential for new discoveries in the realm of artificial memory systems.

## Experimental Status

This model is experimental and in its early stages of development. It currently utilizes basic numpy operations for its implementation, with considerations for future optimizations and migration to more efficient data structures and processing frameworks (e.g., PyTorch for GPU acceleration or sparse matrix representations).

**Key Features:**
- Ternary logic representation for addresses and contents, including 'don't care' states.
- Dynamic adjustment of Hamming radii for memory retrieval, balancing specificity and generalization.
- Decay mechanisms to simulate forgetting, emphasizing the transient nature of episodic memory.
- Experimental encoding of semantic links and sequence-based information to explore the structure of complex memory relationships.

## Goals for Sharing

The primary goal of sharing this project is to foster collaboration and discussion within the research community. We are particularly interested in:
- Feedback on the model’s design and its alignment with cognitive theories.
- Ideas for optimization and scalability, making the model more practical for large-scale experiments.
- Exploration of potential applications, from simulating aspects of human memory to novel machine learning tasks.

## Contributing

As this is a research project, contributions of all forms are welcomed, from theoretical insights and literature suggestions to code optimizations and experimental use cases. Please feel free to fork this repository, experiment with the model, and share your findings or suggestions through issues and pull requests.

## Future Directions

The next steps for this project involve:
- Optimization and efficiency improvements, considering Cython or GPU-accelerated processing.
- Exploration of sparse data structures for more efficient memory representation.
- Extended experiments to validate and refine the model’s capabilities in simulating memory processes.

We invite researchers, enthusiasts, and critics alike to engage with this project, challenge its assumptions, and contribute to its evolution.

## References and Further Reading

Below are some key resources that have inspired and informed the development of this experimental SDAM model:

- [Lecture on SDM to get quick intuition](https://science.slc.edu/jmarshall/courses/2002/fall/cs152/lectures/SDM/sdm.html): A lecture providing a quick and intuitive understanding of Sparse Distributed Memory.
- [Original conception of SDM by Kanerva](https://redwood.berkeley.edu/wp-content/uploads/2020/08/KanervaP_SDMrelated_models1993.pdf): The seminal paper by Pentti Kanerva that introduces the concept of Sparse Distributed Memory.
- [Comparison of attention mechanism in transformers to SDM](https://arxiv.org/abs/2111.05498): A paper that explores the similarities between the attention mechanisms in transformers and the principles of Sparse Distributed Memory.
- [Research on the ternary logic and how it can help with semantic relationships](https://digitalcommons.memphis.edu/cgi/viewcontent.cgi?article=1115&context=ccrg_papers): This paper discusses the benefits of using ternary logic for capturing and processing semantic relationships.
- [General info on SDM and cognitive architectures](https://www.frontiersin.org/articles/10.3389/fnhum.2014.00222/full): An article providing a general overview of Sparse Distributed Memory within the context of cognitive architectures.

These resources provide a foundation for understanding the theoretical and practical aspects that motivate the design and implementation of the SDAM model.

## Credits

This project was developed by [derby](https://github.com/derbydefi) as an exploration into the possibilities of Sparse Distributed Associative Memory (SDAM) and its applications in understanding and simulating cognitive processes related to memory. 

Special thanks to all the researchers and authors whose works have inspired and informed this project. 


## MIT License
Copyright (c) [2024] [derby]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



