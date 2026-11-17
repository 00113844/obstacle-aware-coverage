import ast, sys, re, nbformat
nb = nbformat.read(sys.argv[1], as_version=4)
nbformat.validate(nb)
bad = 0
for i, c in enumerate(nb.cells):
    if c.cell_type == "code":
        src = "\n".join(l for l in c.source.splitlines() if not l.lstrip().startswith(("!", "%")))
        try: ast.parse(src)
        except SyntaxError as e: bad += 1; print("SYNTAX cell", i, e)
    else:
        s = re.sub(r"\$\$.*?\$\$", "", c.source, flags=re.S)
        s = re.sub(r"`[^`]*`", "", s)
        if s.count("$") % 2: bad += 1; print("UNBALANCED $ in md cell", i)
print("cells:", len(nb.cells), "problems:", bad)
