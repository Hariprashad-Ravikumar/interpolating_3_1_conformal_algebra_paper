import re
import numpy as np

# Load Latex table
with open('main_paper.tex', 'r') as f:
    content = f.read()

# Extract IFD table
start_str = r'\begin{tabular}{ |c||c|c|c|c|c|c|c|c|c|c|c|c|c|c|c| }'
caption_pos = content.find('Conformal algebra $(3+1)$ in the IFD limit')
table_start = content.rfind(start_str, 0, caption_pos)
table_end = content.find(r'\end{tabular}', table_start)
table_content = content[table_start:table_end]

lines = table_content.split('\n')
latex_rows = []
for line in lines:
    if line.strip().startswith(r'\rule'):
        row = line.split('&')
        cleaned_row = []
        for cell in row:
            cell = re.sub(r'\\rule\{.*?\}\{.*?\}', '', cell)
            cell = cell.replace('$', '').replace('\\\\', '').strip()
            # If the cell is fully wrapped in {} like {i\mathfrak{K}_{{3}}}
            if cell.startswith('{') and cell.endswith('}'):
                count = 0
                match = True
                for idx, char in enumerate(cell):
                    if char == '{': count += 1
                    elif char == '}': count -= 1
                    if count == 0 and idx < len(cell) - 1:
                        match = False
                        break
                if match:
                    cell = cell[1:-1].strip()
            cleaned_row.append(cell)
        latex_rows.append(cleaned_row)

# Generator Definitions
g = np.zeros((6,6))
g[0,1] = -1; g[1,0] = -1
g[2,2] = 1; g[3,3] = -1; g[4,4] = -1; g[5,5] = -1

def J(a, b):
    mat = np.zeros((6,6), dtype=complex)
    for c in range(6):
        for d in range(6):
            mat[c,d] = 1j * ( (1 if a==c else 0)*g[b,d] - g[a,d]*(1 if b==c else 0) )
    return mat

D = -J(0,1)
P0 = np.sqrt(2)*J(1,2); P1 = np.sqrt(2)*J(1,3); P2 = np.sqrt(2)*J(1,4); P3 = np.sqrt(2)*J(1,5)
K_tilde_0 = -np.sqrt(2)*J(0,2); K_tilde_3 = -np.sqrt(2)*J(0,5)
K_tilde_1 = -np.sqrt(2)*J(0,3); K_tilde_2 = -np.sqrt(2)*J(0,4)
M01 = J(2,3); M02 = J(2,4); M03 = J(2,5)
M31 = J(5,3); M23 = J(4,5); M12 = J(3,4)

K1 = -M01; K2 = -M02; K3 = M03
J1 = M23; J2 = M31; J3 = M12

# IFD Limit Generators (delta = 0)
s = 0.0
c = 1.0

K_p = K_tilde_0*c - K_tilde_3*s
K_m = K_tilde_0*s + K_tilde_3*c
P_p = P0*c + P3*s
P_m = P0*s - P3*c
D_p = D*c + K3*s
D_m = D*s - K3*c
K_hat_1 = -K1*s - J2*c
K_hat_2 = -K2*s + J1*c
D_hat_1 = -K1*c + J2*s
D_hat_2 = -K2*c - J1*s

def comm(A, B):
    return np.dot(A, B) - np.dot(B, A)

# Generators map for IFD
gens_map = {
    '\\mathfrakK_0': K_p, 'P_0': P_p, 'D': D_p,
    '\\mathfrakK_3': K_m, 'P_3': P3, 'K_3': K3,
    'P_1': P1, 'P_2': P2, '\\mathfrakK_1': K_tilde_1, '\\mathfrakK_2': K_tilde_2,
    'K^1': K1, 'K^2': K2, 
    'J^2': J2, 'J^1': J1, 'J^3': J3
}

def latex_to_expr(expr):
    expr = expr.replace('{', '').replace('}', '')
    if expr == '0': return '0'
    expr = expr.replace('i', '1j*')
    expr = expr.replace('21j*', '2*1j*')
    expr = expr.replace('-1j*', '-1*1j*')
    
    # Sort keys by length to avoid partial replacement (e.g., -K_{3} before -K_{1})
    tokens = {}
    for idx, k in enumerate(sorted(gens_map.keys(), key=len, reverse=True)):
        token = f"__GEN_{idx}__"
        if k in expr:
            expr = expr.replace(k, token)
            tokens[token] = k
    
    for token, k in tokens.items():
        expr = expr.replace(token, f"gens_map[{repr(k)}]")
    
    expr = expr.replace('1j*gens_map', '1j*gens_map')
    return expr

gen_names = [r[0].replace('{', '').replace('}', '') for r in latex_rows[1:]]

discrepancies = []
for i, l_row in enumerate(latex_rows[1:]):
    gen_i = gen_names[i]
    for j in range(1, len(l_row)):
        gen_j = gen_names[j-1]
        l_cell = l_row[j].replace('{', '').replace('}', '')
        
        actual_comm = comm(gens_map[gen_i], gens_map[gen_j])
        
        try:
            py_expr = latex_to_expr(l_cell)
            expected_comm = eval(py_expr)
            if not np.allclose(actual_comm, expected_comm):
                # find the correct mathematical expression for it
                actual_id = "UNKNOWN"
                for name, mat in gens_map.items():
                    if np.allclose(actual_comm, 1j*mat): actual_id = f"i{name}"
                    elif np.allclose(actual_comm, -1j*mat): actual_id = f"-i{name}"
                    elif np.allclose(actual_comm, 2j*mat): actual_id = f"2i{name}"
                    elif np.allclose(actual_comm, -2j*mat): actual_id = f"-2i{name}"
                    
                discrepancies.append(f"Row {gen_i}, Col {gen_j}:\n  Latex: {l_cell}\n  Math: {actual_id}")
        except Exception as e:
            discrepancies.append(f"Row {gen_i}, Col {gen_j}:\n  Latex: {l_cell}\n  Eval Error: {e}\n  Parsed as: {py_expr}")

if discrepancies:
    print("Found Discrepancies:")
    for d in discrepancies:
        print(d)
else:
    print("All IFD entries match perfectly!")
