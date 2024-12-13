import os
import time

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd


def build_dict_from_excel(excel_file, discard):
    codes = pd.read_excel(excel_file)
    current_keys, current_level, frequencies = [], -1, {}
    for _, row in codes.iterrows():
        for idx, v in enumerate(row.__iter__()):
            if len(v.strip()) == 0:
                continue
            if idx == current_level:
                current_keys[-1] = v
                break
            elif idx < current_level:
                # update current keys and level
                current_keys = current_keys[:idx] + [v]
                current_level = idx
                break
            else: # idx > current_level
                current_level += 1
                current_keys.append(v)
                break
        if current_keys[-1] not in discard:
            try:
                frequencies[tuple(current_keys)] = int(row.values.tolist()[-1])
            except Exception:
                frequencies[tuple(current_keys)] = 0
    return frequencies


def build_tree_from_dict(data):
    tree = {}
    for key_tuple, frequency in data.items():
        current_level = tree
        for lvl, key in enumerate(key_tuple):
            if key not in current_level:
                current_level[key] = {"name": key, "frequency": frequency, "sub_frequency": 0, "level": lvl, "children": {}}
            current_level = current_level[key]["children"]
    tree = list(tree.values())[0]
    tree['frequency'] = 0

    def add_subtree_freqs(tree):
        if len(tree["children"]) == 0:
            return tree["frequency"]
        else:
            for _, sub_tree in tree["children"].items():
                tree['sub_frequency'] += add_subtree_freqs(sub_tree)
        return tree['sub_frequency'] + tree['frequency']
    
    add_subtree_freqs(tree)
    return tree


def recursively_walk_tree(tree, print_nodes=True):
    freq = str(tree['frequency'])
    if tree['sub_frequency'] > 0:
        freq += f"+{tree['sub_frequency']}"
    node_str = "    " * tree['level'] + f"{tree['name']} ({freq})"
    if print_nodes:
        print(node_str)
    if len(tree["children"]) == 0:
        return 1
    else:
        sub_size = 0
        for sub_tree in tree['children'].values():
            sub_size += recursively_walk_tree(sub_tree, print_nodes=print_nodes)
        return sub_size
    

def merge_rare(tree, threshold=2):
    new = {'name': tree['name'], 'frequency': tree['frequency'], 'sub_frequency': tree['sub_frequency'], 'level': tree['level'], 'children': {}}
    other, keep = 0, []
    for child in tree['children'].values():
        child_freq = child['frequency'] + child['sub_frequency']
        if child_freq <= threshold:
            other += child_freq
        else:
            keep.append( (child_freq, merge_rare(child)) )
    # store in reversed order
    if other > 0:
        new['children']['Others'] = {'name': 'Others', 'frequency': other, 'sub_frequency': 0, 'level': tree['level'] + 1, 'children': {}}
    for _, child in sorted(keep, key=lambda x: x[0]):
        new['children'][child['name']] = child
    return new


def generate_qtree_code(node):    
    sub_freq = '' if node["sub_frequency"] == 0 else f'+{node["sub_frequency"]}'
    node_label = f'{{{node["name"]} ({node["frequency"]}{sub_freq})}}'
    if len(node["children"]) < 2:
        node_label = f'{{{node["name"]} ({node["frequency"]+node["sub_frequency"]})}}'
        return f"[.{node_label} ]"
    else:
        children_code = " ".join(generate_qtree_code(child) for child in node["children"].values())
        return f"[.{node_label} {children_code} ]"
    

def find_tree(tree, code):
    if tree['name'] == code:
        return tree
    for child in tree['children'].values():
        ret = find_tree(child, code)
        if ret:
            return ret
    return None


def find_in_tree(tree, code, parent):
    if tree['name'] == parent and code in tree['children']:
        return True
    if len(tree['children']) == 0:
        return False
    return any([find_in_tree(child, code, parent) for child in tree['children'].values()])
    

fam_quotes = { 
    'General Codes': ("To use AI [\dots] to counter the shortage of skilled workers", 'I14, p. 4'),
    'Types of Daily Work': ("develop an app to detect tolerable products in the supermarket", 'I10, p. 4'),
    'AI Use Cases': ("monitoring the machine condition such that we can make predictions", 'I14, p. 4'),
    'ML Methods': ("the AI evaluates whether the typed text contains specific data", 'I3, p. 44'),
    'ML Tools & Brands': ("I used scikit-learn models and also worked with TensorFlow", 'I13, p. 4'),
    'Requirements on AI': ("My boss doesn't care much about the process, he wants results", 'I13, p. 160'),
    'Benefits': ("Your label hhelps me to decide immediately, it saves a lot of time", 'I9, p. 219'),
    'Limitations': ("I don't get how the value is included in the overall scoring", 'I16, p. 58'),
    'Property Importance': ("the primary objectives: reducing time and enhancing accuracy", 'I7, p. 98'),
    'Associations': ("like I'm looking for a washing machine at the DIY store", 'I3, p. 84'),
    'Target Audience': ("the addressees are likely to be people who are intensively involved", 'I14, p. 64'),
    'Workflows and Use': ("different agendas and newsletters as a regular source of information", 'I16, p. 70'),
    'General Comparison': ("It is time-consuming -- that is the disadvantage of other approaches", 'I9, p. 219'),
    'Reasons for Trust': ("if it has a university stamp on it, it seems more trustworthy", 'I11, p. 144'),
    'Who Needs Trust': ("it helps to understand how the model works if you are a developer", 'I13, p. 20'),
    'Dimensions of Trust': ("trust in AI, or trust in a label -- these are two different things", 'I11, p. 152')
}


PLOT_WIDTH = 900
PLOT_HEIGHT = PLOT_WIDTH // 4

print('Generating figures')
####### DUMMY OUTPUT #######
# for setting up pdf export of plotly
fig=px.scatter(x=[0, 1, 2], y=[0, 1, 4])
fig.write_image("dummy.pdf")
time.sleep(0.5)
os.remove("dummy.pdf")

discard = ['ANON', 'Katha dky', 'Money Quotes']
excel_dict = build_dict_from_excel('MAXQDA24 project - Codesystem.xlsx', discard)
TREE = build_tree_from_dict(excel_dict)
tot_num, tot_freqs = str(recursively_walk_tree(TREE)), str(TREE['frequency'] + TREE['sub_frequency'])
TREE_MERGED = merge_rare(TREE)

# interviewee overview table
interviewees = pd.read_csv('interviewees.csv')
interviewees['id'] = [f'I{idx+1}' for idx in interviewees.index]
tab_cols = ['Job Title', 'Company Type', 'Employees', 'Gender', 'Age', 'AI Skills']
table_rows = [' $AND '.join(['ID'] + tab_cols) + r' \\', r'\toprule']
for idx, row in interviewees[tab_cols].iterrows():
    values = [f'I{idx+1}'] + row.values.astype(str).tolist()
    table_rows.append(' $AND '.join(values) + r' \\')
tab_final = r'\begin{tabular}{lllllll}' + '\n    ' + '\n    '.join(table_rows) + '\n' + r'\end{tabular}'
with open('paper_results/tab_interviewees.tex', 'w') as tab:
    tab.write(tab_final.replace('&', '\&').replace('$AND', '&'))

# generate overview table
table_rows = [' $AND '.join(['Code Family', 'RQ', 'Size', 'Occ', 'Quote']) + r' \\', r'\toprule']
for fam, fam_codes in TREE['children'].items():
    for code, subtree in fam_codes['children'].items():
        quote = fam_quotes.get(code, ('', 'n.a.'))
        depth, fmt_quote = str(recursively_walk_tree(subtree)), r'\small \q{' + quote[0] + r'}' + f' ({quote[1].replace(" ", "~")})'
        freq = str(subtree['frequency'] + subtree['sub_frequency'])
        table_rows.append(' $AND '.join([code, fam.split(' ')[0], depth, freq, fmt_quote]) + r' \\')
table_rows = table_rows + [r'\midrule', ' $AND '.join(['Total', ' ', tot_num, tot_freqs, ' '])]
tab_final = r'\begin{tabular}{lllll}' + '\n    ' + '\n    '.join(table_rows) + '\n' + r'\end{tabular}'
with open('paper_results/tab_codesystem_overview.tex', 'w') as tab:
    tab.write(tab_final.replace('&', '\&').replace('$AND', '&'))

# generate code family trees
for fam, fam_codes in TREE_MERGED['children'].items():
    for code, subtree in fam_codes['children'].items(): 
        tikz_code = '\n'.join([
            r'\begin{tikzpicture}[grow=right,level distance=180pt,scale=1,transform shape]',
            r'\Tree ' + generate_qtree_code(subtree).replace('&', r'\&'),
            r'\end{tikzpicture}' ])
        with open(f'paper_results/codes_{fam.split(" ")[0]}_{code.replace(" ", "_")}.tex', 'w') as tf:
            tf.write(tikz_code)

# Q4 reasons for trust
reasons = find_tree(TREE_MERGED, 'Reasons for Trust')
authorities = find_tree(TREE_MERGED, 'Trustable Authorities')
reason_data = {n: r['frequency'] + r['sub_frequency'] for n, r in authorities['children'].items()}
reason_data.update({n: r['frequency'] + r['sub_frequency'] for n, r in reasons['children'].items() if n!='Trustable Authorities'})
reason_data['Others'] = sum([d['frequency'] + d['sub_frequency'] for d in [reasons['children']['Others'], authorities['children']['Others']]])
fig = go.Figure(go.Bar(y=list(reason_data.keys()), x=list(reason_data.values()), orientation='h'))
fig.add_shape(type="rect", x0=0, y0=0.5, x1=15, y1=4.5, line=dict(color="RoyalBlue"))
fig.add_annotation(text="Authorities", x=16, y=2.5, showarrow=False, textangle=-90)
fig.update_layout(width=PLOT_WIDTH*0.5, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 0}, showlegend=False)
fig.show()
fig.write_image("paper_results/q4_trust_reasons.pdf")

# Q3 competitors used
used = {n.replace('Used: ', ''): c['frequency'] + c['sub_frequency'] for n, c in find_tree(TREE_MERGED, 'Workflows and Use')['children'].items() if 'Used' in n or 'Other' in n}
fig = go.Figure(go.Bar(y=list(used.keys()), x=list(used.values()), orientation='h'))
fig.update_layout(width=PLOT_WIDTH*0.5, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 0}, showlegend=False)
fig.show()
fig.write_image("paper_results/q3_competitors.pdf")

# Q2 benefits VS limitations
sentiment_codes = ['Benefits', 'Room for Improvements', 'Limitations']
sentiment_trees = {code: find_tree(TREE, code) for code in sentiment_codes}
codes_per_id = pd.read_csv('codes_per_id.csv')
results = []
for id, data in codes_per_id.groupby('id'):
    counts = {code: 0 for code in sentiment_codes}
    for _, row in data.iterrows():
        for code, subtree in sentiment_trees.items():
            if find_in_tree(subtree, row['code'], row['parent']):
                counts[code] += row['count']
                break
    idx = interviewees[interviewees['id'] == id].index[0]
    for code, count in counts.items():
        interviewees.loc[idx,code] = count
fig = go.Figure()
fig.add_trace(go.Bar(x=interviewees['id'], y=interviewees['Benefits'], name='Benefits'))
fig.add_trace(go.Bar(x=interviewees['id'], y=interviewees['Room for Improvements'] * -1, name='Room for Improvements'))
fig.add_trace(go.Bar(x=interviewees['id'], y=interviewees['Limitations'] * -1, name='Limitations'))
fig.update_layout(barmode='relative', width=PLOT_WIDTH*0.6, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                  legend=dict(orientation="h", yanchor="top", y=1.2, xanchor="center", x=0.5),
                  yaxis={'title': 'Code Occurrences', 'tick0': -20, 'dtick': 5, 'tickformat': '',
                         'tickvals': [-20, -15, -10, -5, 0, 5, 10, 15, 20], 'ticktext': ['20', "15", '10', "5", '0', "5", '10', "15", '20']})
fig.show()
fig.write_image("paper_results/q2_sentiment.pdf")

# Q2 property importance
codes = {'Energy, Resources & Sustainability': 'Resources', 'Predictive Quality': 'Pred Qual', 'Temporal Performance': 'Temp Perf', 'Consistency & Robustness': 'Robustness'}
fams = ['Q1 - Who and Why?', 'Q2 - How to Label?']
prop_importance = {fam: [] for fam in fams}
for code in codes:
    for fam in fams:
        node = find_tree(TREE['children'][fam], code)
        prop_importance[fam].append( node['frequency'] + node['sub_frequency'] )
        if 'Q2' in fam:
            prop_importance[fam][-1] = prop_importance[fam][-1] * -1
fig = go.Figure()
fig.add_trace(go.Bar(x=list(codes.values()), y=prop_importance['Q1 - Who and Why?'], name='generally discussing AI'))
fig.add_trace(go.Bar(x=list(codes.values()), y=prop_importance['Q2 - How to Label?'], name='facing labels'))
fig.update_layout(barmode='relative', width=PLOT_WIDTH*0.4, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                  legend=dict(orientation='h', yanchor="top", y=1.2, xanchor="center", x=0.5),
                  yaxis={'title': 'Code Occurrences', 'tick0': -20, 'dtick': 5, 'tickformat': '',
                         'tickvals': [-20, -15, -10, -5, 0, 5, 10, 15, 20], 'ticktext': ['20', "15", '10', "5", '0', "5", '10', "15", '20']})
fig.show()
fig.write_image("paper_results/q2_prop_importances.pdf")

# Q1 figure
daily_dicts = {}
for code in ['General Codes', 'Types of Daily Work', 'ML Methods', 'AI Use Cases', 'ML Tools & Brands', 'Requirements on AI']:
    daily_dicts[code] = TREE_MERGED['children']['Q1 - Who and Why?']['children'][code]['children']
fig = make_subplots(rows=2, cols=3, subplot_titles=list(daily_dicts.keys()), horizontal_spacing=0.2, vertical_spacing=0.12)
for i, (code, dict) in enumerate(daily_dicts.items()):
    keys = [key[:20] + '..' if len(key) > 20 else key for key in dict.keys()]
    values = [info['frequency'] + info['sub_frequency'] for info in dict.values()]
    fig.add_trace(
        go.Bar(x=values, y=keys, orientation='h', name=code),
        row=i//3+1, col=i%3+1
    )
fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*2.0, margin={'l': 0, 'r': 0, 'b': 0, 't': 18}, showlegend=False)
fig.show()
fig.write_image("paper_results/q1_bars.pdf")




# data = data.drop('Job Title', axis=1)

# fig = make_subplots(
#     rows=2, cols=2,
#     subplot_titles=data.columns.tolist()
# )

# # Add a histogram for each column
# for i, column in enumerate(data.columns):
#     if data[column].dtype == 'object':
#         print(column)
#         # Categorical data: bar chart of counts
#         fig.add_trace(
#             px.histogram(data, x=column).data[0], 
#             row=i//2+1, col=i%2 + 1
#         )
#     else:
#         # Numeric data: histogram
#         fig.add_trace(
#             px.histogram(data, x=column, nbins=10).data[0], 
#             row=i//2+1, col=i%2 + 1
#         )

# # Update layout
# fig.update_layout(
#     title_text="Distribution of Interviewee Attributes",
#     showlegend=False,
#     height=500, width=600,
#     template='plotly_white'
# )

# # Show plot
# fig.show()
