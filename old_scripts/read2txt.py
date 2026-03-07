import os

def gather_files_contents(parent_folder, output_file):
    # Open output file once, write all contents there
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for root, dirs, files in os.walk(parent_folder):
            for filename in files:
                if filename.endswith(('.mustache', '.md', '.py', '.ini', '.json')):
                    file_path = os.path.join(root, filename)
                    # Write the file path as a header
                    relative_path = os.path.relpath(file_path, parent_folder)
                    out_f.write(f"\n=== FILE: {relative_path} ===\n\n")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            out_f.write(content)
                            out_f.write("\n\n")
                    except Exception as e:
                        out_f.write(f"Error reading file: {e}\n\n")

if __name__ == "__main__":
    parent_folder = "./"  # Replace with your folder
    output_file = "Story_gen.txt"   # Output filename
    gather_files_contents(parent_folder, output_file)
    print(f"File contents collected into {output_file}")
