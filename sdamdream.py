import numpy as np
import random

class SparseDistributedAssociativeMemory:
    """
    Experimental SDM with hebbian learning concepts and ternary logic
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
        self.addresses = np.random.randint(-1, 2, size=(address_space_size, address_dim))
        self.contents = np.zeros((address_space_size, address_dim), dtype=np.float32)
        self.access_frequency = np.zeros(address_space_size, dtype=np.float32)
        self.sequence_pointers = np.zeros((address_space_size, address_space_size), dtype=np.int32)  # Change this line
        self.decay_rate = 0.01
        self.initial_decay_rate = self.decay_rate
        self.global_activity_level = 0.0
        self.local_decay_rates = np.full(address_space_size, self.initial_decay_rate, dtype=np.float32)
        self.radius_adjustment_factor = 0.05
        self.vocab = vocab
        self.reverse_vocab = reverse_vocab

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
        weighted_contents = np.zeros(self.address_dim)
        if np.any(within_radius):
            for idx in np.where(within_radius)[0]:
                adjustment_factor = 1 - distances[idx] / self.local_hamming_radii[idx]
                weighted_contents += adjustment_factor * self.contents[idx]
            # Normalize and convert aggregated content back to ternary logic
            retrieved_pattern = np.sign(weighted_contents)
            return retrieved_pattern
        return np.full(self.address_dim, -1)  # Return 'don't care' for unmatched queries
    
    def store_sequence(self, sequence_indices):
        """
        Stores sequences of patterns using indices to enhance semantic relationships.
        Assumes sequence_indices are valid indices within self.addresses.
        """
        for i in range(len(sequence_indices) - 1):
            start_idx, end_idx = sequence_indices[i], sequence_indices[i + 1]
            self.sequence_pointers[start_idx, end_idx] += 1

    def retrieve_sequence(self, start_index, sequence_length):
        current_index = start_index
        retrieved_indices = [current_index]
        for _ in range(1, sequence_length):
            # Use sequence pointers to guide retrieval
            transitions = self.sequence_pointers[current_index]
            if not np.any(transitions):
                break
            # Introduce a simple form of feedback by prioritizing more frequently accessed sequences
            # This could be refined based on a success metric if available
            probabilities = transitions / np.sum(transitions)
            next_index = np.argmax(probabilities)  # Prioritize the most common transition
            retrieved_indices.append(next_index)
            current_index = next_index
        return [self.decode_token(self.contents[idx]) for idx in retrieved_indices]  
    
    def encode_token(self, token_id):
        """
        Encodes a token ID into a ternary vector with all positions initially set to 'don't care'.
        Only positions corresponding to the binary representation of the token are set to 0 or 1.
        """
        ternary_vector = np.full(self.address_dim, -1)  # Initialize all positions with 'don't care'
        binary_repr = format(token_id, 'b').zfill(self.address_dim)[-self.address_dim:]
        for i, bit in enumerate(binary_repr):
            ternary_vector[i] = int(bit)  # Only change positions that have explicit 0 or 1 values
        return ternary_vector

    def decode_token(self, ternary_vector):
        """
        Decodes a ternary vector back into its corresponding token ID by treating 'don't care' values as 0.
        This reverse operation reconstructs the token ID from its ternary representation.
        """
        # Treat 'don't care' (-1) as 0 for the purpose of decoding
        binary_vector = ''.join(['1' if i == 1 else '0' for i in ternary_vector])
        token_id = int(binary_vector, 2)
        return self.reverse_vocab.get(token_id, "<unk>")

    @staticmethod
    def normalize_embedding(embedding):
        """
        Normalize an embedding vector to a [-1, 1] range, preparing it for ternary conversion.
        This method ensures that embedding values are appropriately scaled for memory storage.
        """
        norm_embedding = (embedding - embedding.min()) / (embedding.max() - embedding.min())
        norm_embedding = 2 * norm_embedding - 1  # Scale to [-1, 1]
        return norm_embedding
    
    @staticmethod 
    def embedding_to_ternary(embedding, lower_threshold=-0.5, upper_threshold=0.5):
        """
        Convert a normalized embedding to a ternary representation, using specified thresholds
        to determine the conversion from numerical values to ternary logic (-1, 0, 1).
        """
        ternary_embedding = np.where(embedding > upper_threshold, 1,
                                    np.where(embedding < lower_threshold, -1, 0))
        return ternary_embedding
    
    @staticmethod
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
    def imaginative_rehearsal(self, focus_factor=0.1, random_exploration_factor=0.05, reinforcement_factor=1.1):
        """
        Performs memory rehearsal by reinforcing frequently accessed or recently added memories,
        combined with random exploration.
        
        Parameters:
        - focus_factor: Portion of memories to be reinforced based on frequency or recency of access.
        - random_exploration_factor: Portion of memories to be randomly explored.
        - reinforcement_factor: Factor by which the memory counters are reinforced.
        """
        total_indices = self.address_space_size
        # Determine the number of indices for focused rehearsal and random exploration
        focus_indices_count = int(total_indices * focus_factor)
        random_indices_count = int(total_indices * random_exploration_factor)
        
        # Select indices for focused rehearsal based on access frequency
        focused_indices = np.argsort(self.access_frequency)[-focus_indices_count:]
        
        # Select indices for random exploration
        random_indices = np.random.choice(total_indices, size=random_indices_count, replace=False)
        
        # Combine focused and random indices, ensuring uniqueness
        combined_indices = np.unique(np.concatenate((focused_indices, random_indices)))
        
        for idx in combined_indices:
            # Simulate memory reinforcement by adjusting counters
            self.contents[idx] += np.where(self.contents[idx] != 0, reinforcement_factor * np.sign(self.contents[idx]), 0)
            
            # Adjust local decay rates to slow down forgetting of rehearsed memories
            self.local_decay_rates[idx] *= (1 - reinforcement_factor / 10)
            
            # Ensure counters and decay rates remain within bounds
            self.contents[idx] = np.clip(self.contents[idx], -1, 1)
            self.local_decay_rates[idx] = np.clip(self.local_decay_rates[idx], self.initial_decay_rate / 2, self.initial_decay_rate * 2)
        


