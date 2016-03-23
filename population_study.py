import os, sys
import patient

print()

# Default to current working directory
directory = os.getcwd()


# Check to see if user gives a different directory
if sys.argv[1]:
    directory = sys.argv[1]

directory = os.path.abspath(directory)

# Check to make sure that the directory is valid
if not os.path.isdir(directory):
    print("Given directory", dir, "is not valid")
    print("Exiting...")
    sys.exit()

file_list = os.listdir(directory)
ecg_files = []  # list of ecg files
patients = []  # list of patient objects

# Create a list of all the ecg files
for file in file_list:
    if '.ecg' in file:
        ecg_files.append(os.path.join(directory,file))

if not ecg_files:
    print("No ecg files found in", directory)
    sys.exit()

for ecg_file in ecg_files:

    # Create the patient object
    print("Creating patient from", ecg_file)
    p = patient.Patient(profile=False,
                        ecg_file=ecg_file)

    # Add the patient to the list of patient objects
    patients.append(p)

    # Read the file
    p.load_ecg_data()

    for lead in p.leads:
        lead.get_heart_rate()
