from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import Counter
from typing import List
import seaborn as sns
from pyannote.core import Segment, Annotation, notebook

@dataclass
class turn:
    from_speaker: str
    to_speaker: str
    distance: float  # negative distance is an overlap

@dataclass
class turn_graph_edge:
    from_speaker: str
    to_speaker: str 
    frequency: int

@dataclass
class SdiTurnGraphEdge:
    from_speaker: str
    to_speaker: str
    frequency: int
    sequence: List[str] = field(default_factory=list)  # Stores the sequence of speakers in this edge if needed

def analyze_turns(rttmreader,start_time=None, end_time=None):
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = rttmreader.get_max_time()
    
    rttms_sorted = rttmreader.get_rttms_sorted_by_time()
    turns = []

    current = []
    last_a = rttms_sorted[0]
    current.append(last_a)

    for a in rttms_sorted:
        # drop speakers not present any more
        current = [x for x in current if x.end > a.begin]
        # check if a is present in current. 
        new_speaker = not any(x.speaker == a.speaker for x in current)            
        if new_speaker:
            # add turn for all current speakers
            for c in current:
                # check interval
                if a.begin >= start_time and a.begin < end_time:
                    d = round(a.begin-c.end,3)
                    turns.append(turn(c.speaker,a.speaker,d))
            # no current speakers
            if not current and last_a.speaker != a.speaker:
                d = round(a.begin-last_a.end,3)
                turns.append(turn(last_a.speaker,a.speaker,d))
            current.append(a)

        last_a = a
    return turns

def analyze_rttm(rttmreader, interval_length=None):
    stats_per_interval = defaultdict(lambda: defaultdict(lambda: {
        'total_time': 0.0,
        'segments': 0,
        'segments_duration': [],
        'speaker_ratio': 0.0,
        'number_of_turns': 0,
        'length_of_turns': [],
        'average_length_of_turns': 0.0,
        'turn_switching_frequency': 0,
        'interruptions_frequency': 0,
        'interruptions_duration': 0.0,
        'average_segment_length': 0.0,
        'variance_segment_length': 0.0
    }))

    if interval_length is None:
        interval_length = rttmreader.get_max_time()

    num_intervals = int(np.ceil(rttmreader.get_max_time() / interval_length))

    for interval_index in range(num_intervals):
        start_time = interval_index * interval_length
        end_time = start_time + interval_length
        interval_annotations = rttmreader.filter_rttms(start_time, end_time)
        interval_turns = analyze_turns(rttmreader,start_time, end_time)
        last_speaker = None

        for annotation in interval_annotations:
            speaker = annotation.speaker
            duration = annotation.duration
            current_stats = stats_per_interval[interval_index][speaker]

            current_stats['total_time'] += duration
            current_stats['segments_duration'].append(duration)
            current_stats['segments'] += 1 

            if last_speaker != speaker:
                if last_speaker is not None:
                    stats_per_interval[interval_index][last_speaker]['turn_switching_frequency'] += 1
                current_stats['number_of_turns'] += 1
                current_stats['length_of_turns'].append(duration)  # Start new turn
            else:
                if current_stats['length_of_turns']:
                    current_stats['length_of_turns'][-1] += duration  # Continue current turn
                else:
                    current_stats['length_of_turns'].append(duration)  # In case first segment for speaker

            last_speaker = speaker

        for turn in interval_turns:
            if turn.distance < 0:  # Handle interruptions
                from_speaker = turn.from_speaker
                interruption_duration = abs(turn.distance)
                stats_per_interval[interval_index][from_speaker]['interruptions_frequency'] += 1
                stats_per_interval[interval_index][from_speaker]['interruptions_duration'] += interruption_duration

        total_interval_spoken_time = sum(stat['total_time'] for stat in stats_per_interval[interval_index].values())
        for speaker, stats in stats_per_interval[interval_index].items():
            stats['average_length_of_turns'] = np.mean(stats['length_of_turns']) if stats['length_of_turns'] else 0
            stats['variance_length_of_turns'] = np.var(stats['length_of_turns']) if stats['length_of_turns'] else 0
            stats['speaker_ratio'] = stats['total_time'] / total_interval_spoken_time if total_interval_spoken_time else 0
            stats['average_segment_length'] = np.mean(stats['segments_duration']) if stats['segments_duration'] else 0
            stats['variance_segment_length'] = np.var(stats['segments_duration']) if stats['segments_duration'] else 0

    data_frames = stats_to_dataframe(stats_per_interval)
    
    return stats_per_interval, data_frames



def stats_to_dataframe(stats_per_interval):
    # Get the enhanced statistics per interval
    
    # Initialize a list to hold all rows of the DataFrame
    data = []
    
    # Iterate through intervals and speakers to create a record for each
    for interval_index in stats_per_interval:
        for speaker, stats in stats_per_interval[interval_index].items():
            # Append a new record for this speaker and interval
            data.append({
                'Interval_Index': interval_index,
                'Speaker': speaker,
                'Total_Speaking_Time': stats['total_time'],
                'Speaker_Ratio': stats['speaker_ratio'],
                'Duration_of_Segments': stats['segments_duration'],
                'Number_of_Segments': stats['segments'],
                'Average_Segment_Length': stats['average_segment_length'],
                'Variance_in_Segment_Length': stats['variance_segment_length'],
                'Number_of_Turns': stats['number_of_turns'],
                'Length_of_Turns': stats['length_of_turns'],
                'Average_Turn_Length': stats['average_length_of_turns'],
                'Variance_in_Turn_Length': stats['variance_length_of_turns'],
                'Turn_Switching_Frequency': stats['turn_switching_frequency'],
                'Interruption_Frequency': stats['interruptions_frequency'],
                'Interruption_Duration': stats['interruptions_duration']                
            })
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    
    # Sort the DataFrame by interval and speaker for better readability
    df.sort_values(by=['Interval_Index', 'Speaker'], inplace=True)
    
    return df

def plot_all_speakers(df, metric, interval_range=None, selected_speakers=None, plot_type='bar', stacked=False):
    if interval_range is not None:
        start_interval, end_interval = interval_range
        df = df[(df['Interval_Index'] >= start_interval) & (df['Interval_Index'] <= end_interval)]
    
    if selected_speakers is not None:
        df = df[df['Speaker'].isin(selected_speakers)]
    
    unique_intervals = df['Interval_Index'].unique()
    unique_speakers = df['Speaker'].unique()
    
    fig = go.Figure()
    
    if plot_type == 'bar':
        fig = px.bar(df, x='Interval_Index', y=metric, color='Speaker', barmode='stack' if stacked else 'group')
    elif plot_type == 'line':
        fig = px.line(df, x='Interval_Index', y=metric, color='Speaker', markers=True)
    elif plot_type == 'stream':
        for speaker in unique_speakers:
            speaker_df = df[df['Speaker'] == speaker]
            fig.add_trace(go.Scatter(x=speaker_df['Interval_Index'], y=speaker_df[metric], mode='lines', stackgroup='one', name=speaker))

    fig.update_layout(title=f'{metric} for Speakers {selected_speakers if selected_speakers else "All"} Over Time',
                      xaxis_title='Interval Index',
                      yaxis_title=metric)
    fig.show()


def count_unique_sequences(turns):
    sequence_counts = Counter()
    for turn in turns:
        sequence_tuple = (turn.from_speaker, turn.to_speaker)
        sequence_counts[sequence_tuple] += 1
    return sequence_counts



def plot_sankey(sequence_counts, threshold=3):
    labels, source, target, value = create_sankey_data(sequence_counts, threshold)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ))])

    fig.update_layout(title_text="Flow of Speaker Turn Sequences", font_size=10)
    fig.show()


def create_sankey_data(sequence_counts, threshold=3):
    labels = []
    source = []
    target = []
    value = []
    
    # Create labels and lookup dictionary
    label_dict = {}
    for seq, count in sequence_counts.items():
        if count >= threshold:
            for speaker in seq:
                if speaker not in label_dict:
                    label_dict[speaker] = len(label_dict)
                    labels.append(speaker)
            for i in range(len(seq) - 1):
                source.append(label_dict[seq[i]])
                target.append(label_dict[seq[i + 1]])
                value.append(count)
                
    return labels, source, target, value


def turns_to_matrix(turns):
    nodes, edges = turns_to_graph(turns)
    node_count = len(nodes)
    adm = np.zeros((node_count, node_count))
    prob_matrix = np.zeros((node_count, node_count))
    node_dict = dict(zip(nodes, range(node_count)))
    for e in edges:
        from_index = int(node_dict[e.from_speaker])
        to_index = int(node_dict[e.to_speaker])
        adm[from_index, to_index] = e.frequency
    return nodes, adm


def turns_to_graph(turns, order=1):
    edges = []
    nodes = set()
    for turn in turns:
        nodes.add(turn.from_speaker)
        nodes.add(turn.to_speaker)
        if order > 1 and hasattr(turn, 'sequence'):
            sequence = turn.sequence
        else:
            sequence = [turn.from_speaker, turn.to_speaker]

        edge_found = next((e for e in edges if e.from_speaker == turn.from_speaker and e.to_speaker == turn.to_speaker), None)
        if edge_found:
            edge_found.frequency += 1
        else:
            edges.append(SdiTurnGraphEdge(turn.from_speaker, turn.to_speaker, 1, sequence))
    return list(nodes), edges


def plot_turns_interaction_matrix(nodes, matrix):
    sorted_indices = np.argsort(nodes)
    sorted_nodes = np.array(nodes)[sorted_indices]
    sorted_matrix = matrix[sorted_indices, :][:, sorted_indices]

    plt.figure(figsize=(8, 6))
    if np.all(matrix <= 1):  # Check if all values <= 1 (probably probabilities)
        title = 'Turn Probability Matrix'
    else:
        title = 'Turns Interaction Matrix'

    sns.heatmap(sorted_matrix, annot=True, fmt=".1f", cmap="YlGnBu", xticklabels=sorted_nodes, yticklabels=sorted_nodes)
    plt.xlabel('To Speaker')
    plt.ylabel('From Speaker')
    plt.title(title)
    plt.show()


def get_pyannotion(rttmreader):
    pyannotetion = Annotation()
    for rttm in rttmreader.get_rttms():
        pyannotetion[Segment(rttm.begin, rttm.end)] = rttm.speaker
    return pyannotetion


def speaker_segment_graph(rttmreader, start_time=None, end_time=None, interval=None, interval_length=None):
    pyannote_annotation = get_pyannotion(rttmreader)

    # If interval and interval length are given, calculate start_time and end_time
    if interval is not None and interval_length is not None:
        if isinstance(interval, tuple):
            # Calculate start and end times based on interval indices
            start_time = interval[0] * interval_length
            end_time = interval[1] * interval_length  # Treat end time as exclusive
        else:
            # Calculate start and end times for a single interval index
            start_time = interval * interval_length
            end_time = (interval + 1) * interval_length

    # Find the minimum start and maximum end time from all segments if no time indications are given
    if start_time is None and end_time is None:
        all_segments = list(pyannote_annotation.itersegments())
        if all_segments:
            start_time = min(segment.start for segment in all_segments)
            end_time = max(segment.end for segment in all_segments)
        else:
            raise ValueError("No segments available to determine the full time range.")

    # Speakers and their indices for Y-axis positioning
    speakers = sorted(pyannote_annotation.labels())
    speaker_indices = {speaker: idx + 1 for idx, speaker in enumerate(speakers)}

    # Create figure
    fig = go.Figure()

    # Iterate through the speakers, collect their segments, and add a trace for each
    for idx, speaker in enumerate(speakers):
        segments_x, segments_y = [], []
        for segment, _, label in pyannote_annotation.itertracks(yield_label=True):
            if label == speaker:
                # Filter segments by start_time and end_time, if given
                if segment.end >= start_time and segment.start <= end_time:
                    segments_x.extend([segment.start, segment.end, None])  # None for discontinuity
                    segments_y.extend([speaker_indices[speaker]] * 2 + [None])

        fig.add_trace(go.Scatter(
            x=segments_x, y=segments_y,
            mode='lines', name=speaker,
            line=dict(color=px.colors.qualitative.Plotly[idx % len(px.colors.qualitative.Plotly)]),
        ))

    # Update the layout
    fig.update_layout(
        title="Speaker Segments",
        xaxis_title="Time (s)",
        xaxis=dict(range=[start_time, end_time] if start_time and end_time else None),
        yaxis=dict(title="Speaker", tickvals=list(speaker_indices.values()), ticktext=list(speaker_indices.keys())),
    )

    # Show the plot
    fig.show()
