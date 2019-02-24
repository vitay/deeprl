import pypandoc
import re, yaml

# Meta information
metadata = yaml.load(open("src/config.yaml", "r"))
title = metadata["title"]
author = metadata["author"]
email = metadata["email"]
css = metadata["css"]

# List of files to compile
files = metadata["files"]

# HTML file where the references will be shown
reference_file = files[-1].replace('.md', ".html").split('-')[1]
bibs = metadata["bibliography"]
bib_pandoc = []
for bib in bibs:
    bib_pandoc.append("--bibliography=" + bib )

# Path to MathJax
mathjax = metadata["mathjax"]

# Gather the files
print("Gathering the files...")
text = ""
for idx, f in enumerate(files):
    filename = "./src/"+f
    with open(filename, "r") as rfile:
        text += rfile.read()
        if not f == files[-1]:
            text += """

<!--PAGEBREAK-->

"""
with open('document.md','w') as wfile:
    with open("src/config.yaml", "r") as rfile:
        header = rfile.read()
    wfile.write("---\n" + header + "---\n\n" + text)

# Arguments to pandoc
filters = ['pandoc-crossref', 'pandoc-citeproc']
pdoc_args = [
    '--template=assets/toc.html',
    '--toc',
    '--mathjax',
    '--number-sections',
    '--metadata=crossrefYaml:"assets/pandoc-crossref.yaml"', # stopped working...
    '--metadata=autoSectionLabels:true',
    '--metadata=numberSections:true',
    '--metadata=secPrefix:Section',
    '--metadata=figPrefix:Fig.',
    '--metadata=eqnPrefix:Eq.',
    '--metadata=figureTitle:Figure',
    '--metadata=linkReferences:true',
    '--metadata=link-citations:true',
    '--metadata=autoEqnLabels:false',
] + bib_pandoc

# Convert the file to html
print("Convert the whole document to html...")
content = pypandoc.convert_text(
    text,
    format='md',
    to='html',
    extra_args=pdoc_args,
    filters=filters)

# Separate the TOC from the content
toc, content = content.split("<!--TOCBREAK-->")

# Open the template
with open("assets/template.html", "r") as rfile:
    template = rfile.read()
output = template %{
    'title': title,
    'author': author,
    'email': email,
    'css': css,
    'mathjax': mathjax,
    'toc': toc,
    'body': content
}

# Write the standalone file to disc
with open('DeepRL.html','w') as wfile:
    wfile.write(output)

# Analyse the toc files separately
print("Convert the single files...")
for f in files:
    # Read the file
    filename = "./src/"+f
    with open(filename, "r") as rfile:
        section = rfile.read()
    local_toc = pypandoc.convert_text(
        section,
        format='md',
        to='html',
        extra_args=pdoc_args,
        filters=filters)
    for line in local_toc.split("\n"):
        if re.search(r'toc-section-number', line):
            # Extract the section tag
            try:
                tag = re.findall(r"href=\"\#sec:[\w-]+\"", line)[0] # Changed in pandoc?
                new_tag = tag.replace('"#', '"./'+f.replace(".md", ".html").split('-')[1] + "#")
                # Update the TOC
                toc = toc.replace(tag, new_tag)
                # Update the total content
                content = content.replace(tag, new_tag)
            except Exception as e:
                print("Could not translate", tag)

# Update the refs
refs = re.findall(r"href=\"\#ref-[\w]*\"", content)
for ref in refs:
    content = content.replace(ref, ref.replace("#", reference_file+"#"))


# Split the content into files
parts = content.split("<!--PAGEBREAK-->")
for idx, f in enumerate(files):
    parts[idx] += """
<br>
<div class="arrows">
<a href="%(prev)s" class="previous">&laquo; Previous</a>
<a href="%(next)s" class="next">Next &raquo;</a>
</div>
"""% {
    'prev': files[idx-1].replace('.md', '.html').split('-')[1] if idx > 0 else "#",
    'next': files[idx+1].replace('.md', '.html').split('-')[1] if idx < len(files) -1 else "#",
}
    part = template %{
        'title': title,
        'author': author,
        'email': email,
        'css': css,
        'mathjax': mathjax,
        'toc': toc,
        'body': parts[idx]
    }
    ofile = f.replace('.md', '.html').split('-')[1]
    with open(ofile, "w") as wfile:
        wfile.write(part)
