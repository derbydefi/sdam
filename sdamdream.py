import numpy as np
import random


# this code should probably be optimized a ton for real usage like cython or pytorch gpu/batch processing, for now lets not worry too much
# also migration to actual sparse data structure (torch sparse matrices?) 
class SparseDistributedAssociativeMemory:
    """
    A model of Sparse Distributed Associative Memory (SDAM) enhanced for better handling of semantic relationships
    and episodic data, using sparse and distributed representations. This version introduces improved handling of
    'don't care' states and weighted sequences for semantic linking, inspired by modifications proposed in research on
    transient episodic memory systems.
    
    Parameters:
    - address_space_size (int): Number of memory slots available for storing patterns.
    - address_dim (int): Length of binary vectors representing the addresses and contents in memory.
    - initial_hamming_radius (float): Initial threshold for activating memory addresses based on Hamming distance.
    - vocab (dict, optional): Mapping from words to indices for encoding.
    - reverse_vocab (dict, optional): Mapping from indices to words for decoding.
    """
    
    def __init__(self, address_space_size, address_dim, initial_hamming_radius, vocab=None, reverse_vocab=None):
        self.address_space_size = address_space_size
        self.address_dim = address_dim
        self.global_hamming_radius = initial_hamming_radius
        self.local_hamming_radii = np.full(address_space_size, initial_hamming_radius, dtype=np.float32)
        # Initialize addresses with ternary logic (-1 represents 'don't care')
        self.addresses = np.random.randint(3, size=(address_space_size, address_dim)) - 1
        self.contents = np.zeros((address_space_size, address_dim), dtype=np.float32)
        self.access_frequency = np.zeros(address_space_size, dtype=np.float32)
        self.sequence_pointers = {}
        self.decay_rate = 0.01
        self.initial_decay_rate = self.decay_rate
        self.global_activity_level = 0.0
        self.local_decay_rates = np.full(address_space_size, self.initial_decay_rate, dtype=np.float32)
        self.radius_adjustment_factor = 0.05
        self.vocab = vocab
        self.reverse_vocab = reverse_vocab
        self.embedding_dim = 100  # Assuming a standard embedding dimension (e.g., GloVe 100d)


    def hamming_distance(self, input_address):
        """
        Calculate the Hamming distance between input_address and stored addresses, considering ternary logic.
        'Don't care' values (-1) do not contribute to the distance.
        """
        no_care_adjustment = np.where(self.addresses == -1, 0, 1)  # Adjust for 'don't care' values
        distances = np.sum(np.abs(self.addresses - input_address) * no_care_adjustment, axis=1)
        return distances

    def update_radius(self):
        """
        Dynamically adjust the global and local Hamming radii based on access patterns to optimize memory retrieval.
        """
        self.global_hamming_radius += (np.mean(self.access_frequency) - self.global_hamming_radius) * self.radius_adjustment_factor
        self.global_hamming_radius = np.clip(self.global_hamming_radius, 1, self.address_dim / 2)
        self.local_hamming_radii += (self.access_frequency - self.local_hamming_radii) * self.radius_adjustment_factor
        self.local_hamming_radii = np.clip(self.local_hamming_radii, 1, self.address_dim / 2)

    def apply_decay(self):
        """
        Simulate forgetting by gradually decreasing the access frequency of memory slots, with adjustments for global
        activity level and local decay rates based on access patterns.
        """
        recent_activity = np.mean(self.access_frequency)
        self.global_activity_level = 0.9 * self.global_activity_level + 0.1 * recent_activity
        max_access = np.max(self.access_frequency) if np.max(self.access_frequency) > 0 else 1
        self.local_decay_rates = self.initial_decay_rate + (1 - (self.access_frequency / max_access)) * self.radius_adjustment_factor
        global_adjustment_factor = 1 - (self.global_activity_level / max_access)
        self.local_decay_rates *= global_adjustment_factor
        self.access_frequency -= self.local_decay_rates
        self.access_frequency = np.clip(self.access_frequency, 0, None)

    def write_to_memory(self, input_address, input_content):
        """
        Store a pattern in memory, with special consideration for ternary logic in both address and content.
        This method automatically applies decay to simulate forgetting before storing new content.
        """


        self.apply_decay()
        distances = self.hamming_distance(input_address)
        within_radius = distances <= self.local_hamming_radii
        for i in np.where(within_radius)[0]:
            adjustment_factor = 1 - distances[i] / self.local_hamming_radii[i]
            for j in range(self.address_dim):
                if input_content[j] != -1:  # Adjust only non-'don't care' values
                    self.contents[i, j] += adjustment_factor * input_content[j]
        self.access_frequency[within_radius] += 1
        self.update_radius()

    def read_from_memory(self, input_address):
        """
        Attempt to recall a stored pattern based on a given input address, using ternary logic and weighted content
        retrieval to reconstruct the most likely pattern.
        """
        distances = self.hamming_distance(input_address)
        within_radius = distances <= self.local_hamming_radii
        if np.any(within_radius):
            adjustment_factor = 1 - distances[within_radius] / self.local_hamming_radii[within_radius]
            weighted_content = np.dot(adjustment_factor, self.contents[within_radius])
            return np.where(weighted_content > 0, 1, np.where(weighted_content < 0, 0, -1))
        return np.zeros(self.address_dim) - 1  # Return 'don't care' for unmatched queries

    def store_sequence(self, sequence):
        """
        Stores sequences of patterns, enhancing semantic relationships through weighted links. This method is
        foundational for encoding episodic memory and sequence-based information.
        """
        for i in range(len(sequence) - 1):
            link = (tuple(sequence[i]), tuple(sequence[i + 1]))
            self.sequence_pointers[link] = self.sequence_pointers.get(link, 0) + 1

    def retrieve_sequence(self, start_address, sequence_length):
        """
        Probabilistically retrieves a sequence based on the stored weights, simulating episodic memory recall.
        This method allows for the dynamic reconstruction of sequences based on partial cues and semantic weights.
        """
        current_address = tuple(start_address)
        retrieved_sequence = [current_address]
        for _ in range(1, sequence_length):
            links = {link: weight for link, weight in self.sequence_pointers.items() if link[0] == current_address}
            if not links:
                break
            total_weight = sum(links.values())
            probabilities = np.array(list(links.values())) / total_weight
            next_address = random.choices(list(links.keys()), weights=probabilities, k=1)[0][1]
            retrieved_sequence.append(next_address)
            current_address = next_address
        return [self.decode_token(vec) for vec in retrieved_sequence]
     
    def encode_token(self, token_id, dont_care_positions=None):
        """
        Encodes a token into a ternary vector, with the capability to specify 'don't care' positions explicitly. This
        method is crucial for flexible memory encoding and handling partial information.
        """
        binary_vector = np.full(self.address_dim, -1, dtype=int)  # Initialize with 'don't cares'
        binary_repr = format(token_id, 'b').zfill(self.address_dim)[-self.address_dim:]
        for i, bit in enumerate(binary_repr):
            if dont_care_positions is None or i not in dont_care_positions:
                binary_vector[i] = int(bit)
        return binary_vector

    def decode_token(self, binary_vector):
        """
        Decodes a binary or ternary vector back into its token representation, using the reverse vocabulary. This
        method supports the retrieval of meaningful information from the memory's encoded data.
        """
        token_id = int("".join(str(bit) for bit in np.where(binary_vector >= 0, binary_vector, 0)), 2)
        if self.reverse_vocab:
            return self.reverse_vocab.get(token_id, "<unk>")
        else:
            raise ValueError("Reverse vocabulary not provided.")

    def normalize_embedding(embedding):
        """Normalize an embedding vector to a [-1, 1] range."""
        norm_embedding = (embedding - embedding.min()) / (embedding.max() - embedding.min())
        norm_embedding = 2 * norm_embedding - 1  # Scale to [-1, 1]
        return norm_embedding

    def embedding_to_ternary(embedding, lower_threshold=-0.5, upper_threshold=0.5):
        """Convert a normalized embedding to a ternary representation."""
        ternary_embedding = np.where(embedding > upper_threshold, 1,
                                    np.where(embedding < lower_threshold, -1, 0))
        return ternary_embedding
    # New method to process embeddings before storing or using them
    def process_embedding(self, embedding, mode='ternary'):
        """
        Normalize, convert, and reshape embeddings to be compatible with SDAM,
        using only NumPy for processing.
        
        Parameters:
        - embedding: The embedding vector to process, expected as a NumPy array.
        - mode: 'ternary' or 'binary' to specify the conversion method.
        
        Returns:
        - Processed embedding ready for SDAM operations.
        """
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Normalize embedding to [-1, 1]
        normalized_embedding = (embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding))
        normalized_embedding = 2 * normalized_embedding - 1  # Scale to [-1, 1]
        
        # Convert to ternary or binary representation
        if mode == 'ternary':
            converted_embedding = np.where(normalized_embedding > 0.5, 1,
                                        np.where(normalized_embedding < -0.5, -1, 0))
        else:  # binary
            converted_embedding = np.where(normalized_embedding > 0, 1, 0)
        
        # Reshape to match address_dim, padding or truncating as needed
        if converted_embedding.size < self.address_dim:
            padding = np.full((self.address_dim - converted_embedding.size,), -1)
            processed_embedding = np.concatenate([converted_embedding, padding])
        elif converted_embedding.size > self.address_dim:
            processed_embedding = converted_embedding[:self.address_dim]
        else:
            processed_embedding = converted_embedding
        
        return processed_embedding
    def imaginative_exploration(self, trigger_condition=False, n_seeds=5, exploration_depth=10, adjustment_factor=0.01):
        """
        Simulates imaginative exploration of memory by probabilistically retrieving sequences
        based on random starting points within the memory, and adjusts local decay rates or Hamming radii.
        
        Parameters:
        - trigger_condition: A boolean flag to determine if the exploration should be triggered.
        - n_seeds: Number of seeds to start the exploration.
        - exploration_depth: Depth of exploration from each seed.
        - adjustment_factor: The factor by which to adjust decay rates or Hamming radii.
        """
        if not trigger_condition:
            return

        seeds_indices = np.random.choice(len(self.addresses), size=n_seeds, replace=False)
        for seed_idx in seeds_indices:
            current_address = self.addresses[seed_idx]
            for _ in range(exploration_depth):
                next_address, can_continue = self.retrieve_simulated_sequence(current_address)
                if not can_continue:
                    break  # End the sequence if no logical continuation is found
                
                # Apply slight adjustments to decay rates or Hamming radii
                self.local_decay_rates[seed_idx] *= (1 + adjustment_factor)  # Example adjustment
                current_address = next_address


    def retrieve_simulated_sequence(self, start_address):
        """
        Retrieves the next address in a sequence, simulating a retrieval process.
        
        Parameters:
        - start_address: The starting address (vector) from which to retrieve the next step.
        
        Returns:
        - A tuple containing the next address in the sequence and a boolean indicating
        if the sequence can logically continue.
        """
        # Simulate retrieval by finding the closest memory address to `start_address`
        distances = self.hamming_distance(start_address)
        closest_idx = np.argmin(distances)  # Find the index of the closest memory address
        
        if distances[closest_idx] <= self.local_hamming_radii[closest_idx]:
            next_address = self.addresses[closest_idx]
            # Check if a logical next step exists (for simplicity, assume it always does)
            can_continue = True
        else:
            next_address = None
            can_continue = False

        return next_address, can_continue
