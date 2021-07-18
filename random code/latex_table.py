import json, os
import numpy as np

def original(config, details, output, stats):
    # output = output + "{ |c|c|c|c|c|c|c| }\n\t\t\\hline\n\t\tDICE & accuracy & AUC & con & dil & ero & dif\\\\\\hline\n"
    output = output + "{ |c|c|c|c|c|c| }\n\t\t\\hline\n\t\tDICE & ACC & AUC & con & dil & ero\\\\\\hline\n"

    o = "\\checkmark"
    means = np.round(stats[0], decimals=4)
    maxim = np.round(stats[1], decimals=4)
    minim = np.round(stats[2], decimals=4)
    for i in range(len(config)):
        s = np.round(details[i], decimals=4)
        c = config[i]["components"]

        output = output + "\t\t"
        for i in range(len(s)):
            if s[i] == maxim[i]:
                output = output + "\\textcolor{best}{\\textbf{" + str(s[i]) + "}} & "
            elif s[i] == minim[i]:
                output = output + "\\textcolor{worst}{\\textbf{" + str(s[i]) + "}} & "
            elif s[i] > means[i]:
                output = output + "\\textcolor{blue}{" + str(s[i]) + "} & "
            else:
                output = output + "\\textcolor{red}{" + str(s[i]) + "} & "
        
        # output = output + c[0]*o + " & " + c[1]*o + " & " + c[2]*o + " & " + c[3]*o + "\\\\\\hline\n"
        output = output + c[0]*o + " & " + c[1]*o + " & " + c[2]*o + "\\\\\\hline\n"
    
    return output

def alpha(config, details, output, stats):
    output = output + "{ |c|c|c|c|c| }\n\t\tDICE & ACC & AUC & $\\alpha$ & model\\\\\\hline\n"

    means = np.round(stats[0], decimals=4)
    maxim = np.round(stats[1], decimals=4)
    minim = np.round(stats[2], decimals=4)
    for i in range(len(config)):
        a = str(config[i]["alpha"])
        s = np.round(details[i], decimals=4)
        m = config[i]["name"][4:10]
        if m == "111010":
            m = "1"
        elif m == "111011":
            m = "2"
        else:
            m = "3"
        
        output = output + "\t\t"
        for i in range(len(s)):
            if s[i] == maxim[i]:
                output = output + "\\textcolor{best}{\\textbf{" + str(s[i]) + "}} & "
            elif s[i] == minim[i]:
                output = output + "\\textcolor{worst}{\\textbf{" + str(s[i]) + "}} & "
            elif s[i] > means[i]:
                output = output + "\\textcolor{blue}{" + str(s[i]) + "} & "
            else:
                output = output + "\\textcolor{red}{" + str(s[i]) + "} & "
        
        output = output + a + " & " + m + "\\\\\\hline\n"
    
    return output

def channels(config, details, output, stats):
    output = output + "{ |c|c|c|c|c| }\n\t\tDICE & ACC & AUC & channels & model\\\\\\hline\n"

    means = np.round(stats[0], decimals=4)
    maxim = np.round(stats[1], decimals=4)
    minim = np.round(stats[2], decimals=4)
    for i in range(len(config)):
        a = str(config[i]["channels"])
        s = np.round(details[i], decimals=4)
        m = config[i]["name"][4:10]
        if m == "111010":
            m = "1"
        elif m == "111011":
            m = "2"
        else:
            m = "3"
        
        output = output + "\t\t"
        for i in range(len(s)):
            if s[i] == maxim[i]:
                output = output + "\\textcolor{best}{\\textbf{" + str(s[i]) + "}} & "
            elif s[i] == minim[i]:
                output = output + "\\textcolor{worst}{\\textbf{" + str(s[i]) + "}} & "
            elif s[i] > means[i]:
                output = output + "\\textcolor{blue}{" + str(s[i]) + "} & "
            else:
                output = output + "\\textcolor{red}{" + str(s[i]) + "} & "
        
        output = output + a + " & " + m + "\\\\\\hline\n"
    
    return output

def conventional(config, details, output, stats):
    output = output + "{ |c|c|c|c|c| }\n\t\tDICE & ACC & AUC & channels\\\\\\hline\n"

    means = np.round(stats[0], decimals=4)
    maxim = np.round(stats[1], decimals=4)
    minim = np.round(stats[2], decimals=4)
    for i in range(len(config)):
        a = str(config[i]["channels"])
        s = np.round(details[i], decimals=4)

        output = output + "\t\t"
        for i in range(len(s)):
            if s[i] == maxim[i]:
                output = output + "\\textcolor{best}{\\textbf{" + str(s[i]) + "}} & "
            elif s[i] == minim[i]:
                output = output + "\\textcolor{worst}{\\textbf{" + str(s[i]) + "}} & "
            elif s[i] > means[i]:
                output = output + "\\textcolor{blue}{" + str(s[i]) + "} & "
            else:
                output = output + "\\textcolor{red}{" + str(s[i]) + "} & "
        
        output = output + a + "\\\\\\hline\n"
    
    return output

def alpha_stats(config, details, output, stats):
    alpha = ''.join(input("Give alpha: ").split('.'))
    new_config = []
    new_details = []
    for i in range(len(config)):
        if config[i]['name'][5:] == alpha and not config[i]['name'][:4] == '1000':
            new_config.append(config[i])
            new_details.append(details[i])
    
    details = np.array(new_details)
    means = np.median(details, axis=0)
    maximum = np.max(details, axis=0)
    minimum = np.min(details, axis=0)

    stats = [means, maximum, minimum]
    
    return original(new_config, details, output, stats)

def beta_stats(config, details, output, stats):
    output = output + "{ |c|c|c|c|c| }\n\t\t\\hline\n\t\tDICE & ACC & AUC & size & $\\lambda$\\\\\\hline\n"
    
    means = np.round(stats[0], decimals=4)
    maxim = np.round(stats[1], decimals=4)
    minim = np.round(stats[2], decimals=4)

    for i in range(len(config)):
        a = str(int(config[i]['lambda']) / 10)
        b = config[i]['name'][-1]
        s = np.round(details[i], decimals = 4)

        output = output + "\t\t"
        for i in range(len(s)):
            if s[i] == maxim[i]:
                output = output + "\\textcolor{best}{\\textbf{" + str(s[i]) + "}} & "
            elif s[i] == minim[i]:
                output = output + "\\textcolor{worst}{\\textbf{" + str(s[i]) + "}} & "
            elif s[i] > means[i]:
                output = output + "\\textcolor{blue}{" + str(s[i]) + "} & "
            else:
                output = output + "\\textcolor{red}{" + str(s[i]) + "} & "
        
        output = output + b + ' & ' + a + "\\\\\\hline\n"
    
    return output

def dilero(config, details, output, stats):
    output = output + "{ |c|c|c|c|c|c|c|c|c| }\n\t\t\\hline\n\t\tDICE & ACC & AUC & dil l2 & ero l2 & dil l3 & ero l3 & dil l4 & ero l4\\\\\\hline\n"

    o = "\\checkmark"
    means = np.round(stats[0], decimals=4)
    maxim = np.round(stats[1], decimals=4)
    minim = np.round(stats[2], decimals=4)
    for i in range(len(config)):
        s = np.round(details[i], decimals=4)
        c = config[i]["components"]

        output = output + "\t\t"
        for i in range(len(s)):
            if s[i] == maxim[i]:
                output = output + "\\textcolor{best}{\\textbf{" + str(s[i]) + "}} & "
            elif s[i] == minim[i]:
                output = output + "\\textcolor{worst}{\\textbf{" + str(s[i]) + "}} & "
            elif s[i] > means[i]:
                output = output + "\\textcolor{blue}{" + str(s[i]) + "} & "
            else:
                output = output + "\\textcolor{red}{" + str(s[i]) + "} & "
        
        output = output + c[0]*o + " & " + c[1]*o + " & " + c[2]*o + " & " + c[3]*o + " & " + c[4]*o + " & " + c[5]*o + "\\\\\\hline\n"
    
    return output

def channels(config, details, times, parameters, output):
    output = output + "{ |c|c|c|c|c|c|c| }\n\t\t\\hline\n\t\tDICE & ACC & AUC & test & train & params & channels\\\\\\hline\n"
    current = 0
    sets = [i for i in range(5)]
    details = np.array([details[current + i] for i in sets])
    times = np.array([times[current + i] for i in sets])
    config = [config[current + i] for i in sets]
    parameters = [parameters[current + i] for i in sets]

    for i in range(5):
        s = np.round(details[i], decimals=4)
        t = np.round(times[i], decimals=4)
        p = parameters[i]
        c = 14

        output = output + '\t\t'
        for j in range(len(s)):
            output = output + f'{s[j]} & '
        
        output = output + f'{t[0]} & {t[1]} & {p} & {c} \\\\\\hline\n'

    return output


FOLDER = os.path.join("output", 'conv-chan-14')

config = []
with open(os.path.join(FOLDER, "conf_code"), "r") as file:
    config = json.loads(file.read())

details = []
with open(os.path.join(FOLDER, "listed_details"), "r") as file:
    details = json.loads(file.read())

times = []; parameters = []
for i in range(len(config)):
    with open (os.path.join(FOLDER, "JSON files", config[i]['name'] + '_details')) as file:
        data = file.read().split('\n')[-4:-1]
        parameters.append(int(data[0].split(': ')[-1]))
        times.append([float(data[1].split(': ')[-1]), float(data[2].split(': ')[-1])])

details = np.array(details)
times = np.array(times)
parameters = np.array(parameters)

# means = np.mean(details, axis=0)
# maximum = np.max(details, axis=0)
# minimum = np.min(details, axis=0)

# stats = [means, maximum, minimum]

output = "\\begin{table}[H]\n\t\\centering\n\t\\begin{tabular}"

# output = original(config, details, output, stats)
# output = alpha(config, details, output, stats)
# output = channels(config, details, output, stats)
# output = conventional(config, details, output, stats)
# output = alpha_stats(config, details, output, stats)
# output = beta_stats(config, details, output, stats)
# output = dilero(config, details, output, stats)
output = channels(config, details, times, parameters, output)

output = output + "\t\\end{tabular}\n\t\\caption{}\\label{}\n\\end{table}"

with open(os.path.join(FOLDER, f"latex_table_mean"), "w") as file:
    file.write(output)

# half = len(config)/2
# wins = [[0 if listed_details[i][j] > listed_details[i + half][j] else 1 for j in range(3)] for i in range(half)]

# victories = sum(wins)
# model = [sum([[wins[3*i + j][k] for k in range(3)] for i in range(len(config) / 3)]) for j in range(3)]

# details = np.array(details)
# details = details.transpose()