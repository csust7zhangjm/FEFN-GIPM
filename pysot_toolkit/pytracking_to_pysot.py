import os

def process_files_in_folder(folder_path):
    output_folder = folder_path + "_output"
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            input_file = os.path.join(folder_path, file_name)
            output_file = os.path.join(output_folder, file_name)

            with open(input_file, "r") as input_fp, open(output_file, "w") as output_fp:
                for line in input_fp:
                    processed_line = ",".join(line.split())
                    output_fp.write(processed_line + "\n")

            print(f"Processed {file_name} successfully.")

    print("All files processed.")

# Specify the folder path where the .txt files are located
folder_path = "/root/code/TransT/results/uav_raw_result/Ours"

process_files_in_folder(folder_path)
