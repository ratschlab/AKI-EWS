import numpy as np


stepsize = 60 * 6
maxtime = 3 * 24 * 60
min_lim = 0.01


def endpoints_to_samples(endpoints, stepsize=stepsize):
    samples = []
    for k_patient in endpoints.keys():
        current_t = 0
        current_state = np.nan
        sample = []
        for state, time in endpoints[k_patient]:
            delta_t = 0
            while current_t < time:
                if not np.isnan(current_state):
                    if len(sample) > 0:
                        if str(current_state) + str(int(delta_t / stepsize)) != sample[-1]:
                            sample.append(str(current_state) + str(int(delta_t / stepsize)))
                    else:
                        sample.append(str(current_state) + str(int(delta_t / stepsize)))
                current_t = current_t + stepsize
                delta_t = delta_t + stepsize
            current_state = state
        samples.append(sample)
    return samples

def printTable(myDict):
    print(myDict)
    import pandas as pd
    df = pd.DataFrame(myDict)
    print(df)


def generate_distributions(samples):
    from pomegranate import MarkovChain, DiscreteDistribution, ConditionalProbabilityTable
    states = set()
    for patient in samples:
        states = states.union(set(patient))
    states = sorted(list(states))

    initial_distribution = {}
    for i in states:
        initial_distribution[i] = 1 / (len(states))
    d1 = DiscreteDistribution(initial_distribution)

    initial_weights = []
    for i in states:
        for j in states:
            initial_weights.append([i, j, 1 / len(states)**2])
    d2 = ConditionalProbabilityTable(initial_weights, [d1])

    return d1, d2

def endpoints_to_probabilities(endpoints):
    samples = []
    for k_patient in endpoints.keys():
        current_t = 0
        current_state = np.nan
        sample = []
        for state, time in endpoints[k_patient]:
            while current_t < time:
                if not np.isnan(current_state):
                    sample.append(str(current_state))
                current_t = current_t + stepsize
            current_state = state
        samples.append(sample)

    states = ['-1','0','1','2','3']
    counts = {}
    for state in states:
        counts[state] = {}
        for stat in states:
            counts[state][stat] = 1

    for patient in samples:
        prev_st = 'na'
        for state in patient:
            if prev_st != 'na':
                counts[prev_st][state] += 1
            prev_st = state   

    for state in counts.keys():
        normalization = sum([counts[state][i] for i in counts[state].keys()])
        for transition in counts[state].keys():
            counts[state][transition] /= normalization
    return counts



def samples_to_probabilities(samples):
    states = set()
    for patient in samples:
        states = states.union(set(patient))
    states = sorted(list(states))

    counts = {}
    for state in states:
        counts[state] = {}

    for patient in samples:
        prev = '-1'
        for state in patient:
            if prev != '-1':
                if not state in counts[prev].keys():
                    counts[prev][state] = 0
                counts[prev][state] += 1
            prev = state

    for state in counts.keys():
        normalization = sum([counts[state][i] for i in counts[state].keys()])
        for transition in counts[state].keys():
            counts[state][transition] /= normalization
            # counts[state][transition] = - np.log(counts[state][transition])
    return counts


def plot_transition(counts):
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LogNorm
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(4, 7)
    ax1 = fig.add_subplot(gs[0, :-1])
    ax2 = fig.add_subplot(gs[1, :-1])
    ax3 = fig.add_subplot(gs[2, :-1])
    ax4 = fig.add_subplot(gs[3, :-1])
    ax5 = fig.add_subplot(gs[:, -1])
    ax5.set_axis_off()
    axarr = [ax1, ax2, ax3, ax4, ax5]
    for start in range(4):
        timeaxis = int((maxtime) / stepsize)
        transition_matrix = np.zeros((4, timeaxis))
        for time in range(timeaxis):
            for end in range(4):
                if str(start) + str(time) in counts.keys():
                    if end != start:
                        if str(end) + '0' in counts[str(start) + str(time)].keys():
                            transition_matrix[end, time] = counts[str(start) + str(time)][str(end) + '0']
                    else:
                        if str(end) + str(time + 1) in counts[str(start) + str(time)].keys():
                            transition_matrix[end, time] = counts[str(start) + str(time)][str(end) + str(time + 1)]

        # axarr[start].imshow(transition_matrix, cmap='hot', interpolation='nearest',aspect="auto")
        # print(np.min(transition_matrix[np.nonzero(transition_matrix)]))
        transition_matrix[transition_matrix < min_lim] = min_lim
        img = axarr[start].imshow(transition_matrix, cmap='viridis', norm=LogNorm(vmin=min_lim, vmax=1), interpolation='nearest', aspect='auto')
        if start == 3:
            axarr[start].set_xticks(range(timeaxis))
            axarr[start].set_xticklabels([str(int(i * stepsize / 60)) for i in range(timeaxis)])
            axarr[start].set_xlabel('hours since last transition')
            cbar = plt.colorbar(img, ax=axarr[-1])
            cbar.ax.get_yaxis().labelpad = 15
            cbar.set_label('observed probability for transition', rotation=270)
        else:
            axarr[start].set_xticklabels([])
        axarr[start].set_ylabel('KDIGO stage')
        axarr[start].set_title('KDIGO stage ' + str(start))
        # axarr[start].set_yticks([0.5,1.5,2.5,3.5])
        # axarr[start].set_yticklabels(range(4))
        axarr[start].set_yticks(range(4))
    plt.tight_layout()
    plt.savefig('transitions.pdf')


def generate_endpoints():
    underlying_p = [[0.95, 0.03, 0.01, 0.01], [0.3, 0.6, 0.08, 0.02], [0.1, 0.3, 0.5, 0.1], [0.1, 0.2, 0.2, 0.5]]
    statenames = [0, 1, 2, 3]
    endpoints = {}
    r = np.random.RandomState(42)
    length = 25
    for i in range(4000):
        patient = [(0, 0)]
        state = 0
        for j in range(length):
            new = r.choice(statenames, replace=True, p=underlying_p[state])
            if new != state:
                patient.append((new, j))
                state = new
        patient.append((-1, length))
        endpoints[i] = patient
    return endpoints


def load_endpoints():
    base_ep = '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/3b_endpoints_KDIGO/'
    base_ep = '../../endpoints_hirid/'

    import pandas as pd
    reference_time = pd.Timedelta(np.timedelta64(1, 'm'))

    from glob import glob
    import pickle
    KDIGO_dict = {}
    for file in glob(base_ep + '*.pkl'):
        KDIGO_dict = {**KDIGO_dict, **pickle.load(open(file, 'rb'))}  # merge to single dict

    for pid in KDIGO_dict.keys():
        if len(KDIGO_dict[pid]) > 0:
            new_ep = []
            start_of_obs = KDIGO_dict[pid][0][1]
            for ep in KDIGO_dict[pid]:
                new_ep.append((ep[0], (ep[1] - start_of_obs) / reference_time))
            KDIGO_dict[pid] = new_ep
    return KDIGO_dict


endpoints = load_endpoints()
printTable(endpoints_to_probabilities(endpoints))
samples = endpoints_to_samples(endpoints)
counts = samples_to_probabilities(samples)
plot_transition(counts)

# print(counts)
