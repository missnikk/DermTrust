import json

with open('ml/failure_audit_results.json') as f:
    data = json.load(f)

header = '%-25s | %-10s | %-6s | %-10s | %-8s | %-6s | %-7s | %-12s | %-22s'
print(header % ('File', 'Top Class', 'Conf', '2nd Class', '2nd Conf', 'Margin', 'Entropy', 'Centroid', 'Outcome'))
print('-' * 115)
for r in data:
    c_str = f"({r['cam_centroid'][0]:.1f},{r['cam_centroid'][1]:.1f})"
    print(header % (
        r['filename'],
        r['top_class'],
        f"{r['top_confidence']*100:.1f}%",
        r['top2_class'],
        f"{r['top2_confidence']*100:.1f}%",
        f"{r['margin']*100:.1f}%",
        f"{r['entropy']:.2f}",
        c_str,
        r['gate_outcome']
    ))
