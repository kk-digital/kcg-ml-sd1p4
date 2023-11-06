import json
import os


class MemoryJSON:
    def __init__(self, filename, batch_size=100000, size_memory=1e9):  # Default size_memory is 1GB
        self.data = []
        self.filename = filename
        self.batch_size = batch_size
        self.size_memory = size_memory
        self.current_size = 0  # Running total of data size
        self.count = 0

        # Check if file exists and its size
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            with open(self.filename, 'w') as f:
                f.write("[")
            self.is_file_empty = True
        else:
            self.is_file_empty = False
            # remove the closing bracket
            with open(self.filename, 'rb+') as f:
                f.seek(-1, os.SEEK_END)
                f.truncate()

    def add(self, entry):
        # Maintain a running total of data size
        serialized_entry = json.dumps(entry)
        self.current_size += len(serialized_entry)

        self.data.append(serialized_entry)
        self.count += 1

        # Check batch size first before memory size for optimization
        if self.count >= self.batch_size or self.current_size > self.size_memory:
            self.dump_to_file()

    def dump_to_file(self):
        with open(self.filename, 'a') as f:
            # If file is not empty, prepend with a comma
            prefix = '' if self.is_file_empty else ','
            # Since data is already serialized, just join them
            serialized_data = prefix + ",".join(self.data)
            f.write(serialized_data)

            # Mark file as not empty for subsequent writes
            self.is_file_empty = False

        self.data.clear()
        self.current_size = 0
        self.count = 0

    def finalize(self):
        self.dump_to_file()
        # Ensure the file ends with a closing bracket
        with open(self.filename, 'a') as f:
            f.write("]")


# Example usage:
if __name__ == "__main__":
    memory_json = MemoryJSON('large_data.json', size_memory=1e6)  # Example with 1MB threshold

    # Add data
    for i in range(100):  # This is just an example, adjust the range accordingly
        memory_json.add({"id": i, "name": f"Name {i}"})

    # Dump any remaining data to file
    memory_json.finalize()
