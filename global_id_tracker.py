class GlobalIDTracker:
    """ To get global identity (ID) from different annotated video sequences. """
    def __init__(self):
        self.global_id_counter = 0
        self.sequence_mapping = {}  # {(sequence_id, local_id): global_id}
        self.global_to_sequence = {}  # {global_id: (sequence_id, local_id)}

    def get_global_id(self, sequence_id, local_id):
        """Get or create a global ID for a local ID in a sequence"""
        key = (sequence_id, local_id)

        if key in self.sequence_mapping:
            return self.sequence_mapping[key]

        # Create new global ID
        global_id = self.global_id_counter
        self.sequence_mapping[key] = global_id
        self.global_to_sequence[global_id] = key
        self.global_id_counter += 1

        return global_id

    def get_sequence_info(self, global_id):
        """Get a sequence and local ID from global ID"""
        return self.global_to_sequence.get(global_id)

    def get_all_mappings(self):
        """Get all mappings"""
        return self.sequence_mapping.copy()


# Example usage
if __name__ == "__main__":
    # Sample data structure
    sequences = {
        "seq1": {
            "frame1": [{"id": 0}, {"id": 1}, {"id": 2}],
            "frame2": [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}],
        },
        "seq2": {
            "frame1": [{"id": 0}, {"id": 1}],
            "frame2": [{"id": 0}, {"id": 1}, {"id": 2}],
        },
        "seq3": {
            "frame1": [{"id": 0}],
        }
    }

    # Create tracker
    tracker = GlobalIDTracker()

    # Process all sequences
    for seq_name, frames in sequences.items():
        print(f"\nProcessing sequence: {seq_name}")
        for frame_name, objects in frames.items():
            for obj in objects:
                local_id = obj["id"]
                global_id = tracker.get_global_id(seq_name, local_id)
                print(f"  Frame {frame_name}: local_id={local_id} -> global_id={global_id}")

    # Get all mappings
    print("\n" + "=" * 50)
    print("All Mappings:")
    for (seq_id, local_id), global_id in tracker.get_all_mappings().items():
        print(f"  Sequence: {seq_id}, Local ID: {local_id} -> Global ID: {global_id}")