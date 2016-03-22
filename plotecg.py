import sys

# Custom imports
import patient

try:
    funct = sys.argv[1]
except IndexError:
    print("Usage Examples: \n"
          "\tpython plotecg.py -HR 9004.ecg (Calculates heart rate and plots it)\n"
          "\tpython plotecg.py -L 1 2 3 9004.ecg (Plots the raw lead data)\n")
    sys.exit()

leads = None

if funct == "-HR":
    filename = sys.argv[2]
elif funct == "-L":
    length = len(sys.argv)
    if length == 3:
        filename = sys.argv[2]
    elif length == 4:
        filename = sys.argv[3]
        leads = int(sys.argv[2]) - 1
    elif length == 5:
        filename = sys.argv[4]
        leads = [int(sys.argv[2]) - 1, int(sys.argv[3]) - 1]
    elif length == 6:
        filename = sys.argv[5]
        leads = [int(sys.argv[2]) - 1, int(sys.argv[3]) - 1, int(sys.argv[4]) - 1]
    else:
        print("Please specify between 0 and 3 leads\n")
        sys.exit()

else:
    print("Please use -HR or -L\n")
    sys.exit()
try:
    f = open(filename, 'rb')
except IOError:
    print('%s cannot be opened', filename)
    sys.exit()

# Create the patient object
p = patient.Patient()

# Read the file
p.load_ecg_data(filename)

if funct == "-HR":
    for lead in p.active_leads:
        p.leads[lead].get_heart_rate()
    #p.plot_hr_data()
if funct == "-L":
    p.plot_ecg_leads_voltage(leads)

    # for lead in p.leads:
    #    lead.plot_lead()
