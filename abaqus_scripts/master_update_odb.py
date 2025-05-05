import os
import subprocess

velocities = [10, 20, 40]
base_path = "D:/AISI_1020/Vel_{}/odb/"
update_odb_script = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "Update_odb_batch.py")
)

for vel in velocities:
    input_path = base_path.format(vel)
    if not os.path.exists(input_path):
        print("Directory not found:", input_path)
        continue

    odb_files = [f for f in os.listdir(input_path) if f.lower().endswith('.odb')]
    odb_files.sort()
    for i in range(0, len(odb_files), 20):
        batch = odb_files[i:i+20]
        batch_file = os.path.abspath(f"odb_batch_{vel}_{i//20}.txt")
        with open(batch_file, "w") as f:
            for odb in batch:
                f.write(odb + "\n")
        cmd = [
            "abaqus", "cae", "-noGUI", update_odb_script, batch_file, input_path
        ]
        print("Запуск:", " ".join(cmd))
        subprocess.call(cmd)
        os.remove(batch_file)

        exit(0)
