import os

import pandas as pd
import numpy as np

qualification = { # according to German Qualifications Framework
    1: 'Prevocationally trained',
    2: 'Vocationally trained',
    3: 'Interm. school graduated',
    4: 'High school graduated',
    5: 'Expert or Specialist',
    6: 'Bachelor or Senior',
    7: 'Master or Diploma',
    8: 'Doctorate'
}

ai_skills = {
    1: 'Novice (interested but no deeper understanding)',
    2: 'Beginner (general understanding but no practical experience)',
    3: 'User (practical experience with AI-as-a-service)',
    4: 'Engineer (basic experience with developing AI models)',
    5: 'Expert (extensive experience with developing AI models)'
}

renamed = {
    "04_1_company_info":                                ('Company Type',            None),
    "04_2_n_employees":                                 ('Employees',               None),
    '05_angestellt-als':                                ('Job Title',               None),
    '08_geschlecht':                                    ('Gender',                  lambda v: 'female' if 'weib' in v else 'male'),
    '09_alter':                                         ('Age',                     lambda v: str(int(v)) if not np.isnan(v) else '---'),
    '10_hoechste-berufsqualifikation':                  ('Highest Qualification',   lambda v: qualification[int(v.split(' ')[0].split('(')[0])]),
    '11_technische-ki-kenntnisse-selbsteinschaetzung':  ('Numeric AI Skills', lambda v: int(v.split(' ')[0].split('(')[0]))
}

renamed_cols = {col: ren[0] for col, ren in renamed.items()}
col_map_funcs = {func[0]: func[1] for func in renamed.values() if func[1] is not None}
data = pd.read_csv('.anon_exports/applicants.csv')
data = data.drop([col for col in data.columns if col not in renamed], axis=1).rename(renamed_cols, axis=1)
for col, func in col_map_funcs.items():
    data[col] = data[col].map(func)
data['Self-Assessed AI Skills'] = data['Numeric AI Skills'].map(lambda v: ai_skills[v])
data['AI Skills'] = data['Self-Assessed AI Skills'].map(lambda v: v.split(' ')[0].split('(')[0])
data['Job Title (Qualification)'] = data['Job Title'] + ' (' + data['Highest Qualification'] + ')'
data.sort_values(['Age']).to_csv('interviewees.csv', index=False)

coded_segments = []
rename = {'Code': 'code', 'Obercode': 'parent', 'Cod. Seg. (alle Dokumente)': 'count'}
for code_file in os.listdir('.anon_exports'):
    if '_Codes.xlsx' in code_file:
        id = code_file.split('_')[0]
        data = pd.read_excel(os.path.join('.anon_exports', code_file))
        data = data[list(rename.keys())]
        data['id'] = id
        coded_segments.append(data)
segments = pd.concat(coded_segments, axis=0)
segments = segments.rename(rename, axis=1)
segments.to_csv('codes_per_id.csv', index=False)
