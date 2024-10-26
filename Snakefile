from bs4 import BeautifulSoup
from openai import OpenAI
import os
from norerun import norerun
import tiktoken
MODEL = "gpt-3.5-turbo"
commentary_categories = [
    'Chasidut',
    'Commentary',
    'Guides',
    'Halakhah',
    'Jewish Thought',
    'Kabbalah',
    'Liturgy',
    'Midrash',
    'Mishnah',
    'Musar',
    'Quoting Commentary',
    'Responsa',
    'Second Temple',
    'Talmud',
    'Tanakh',
    'Targum',
    'Tosefta'
    ]

checkpoint get_parashot_csv:
    output:
        "parashot.csv"
    run:
        import pandas as pd

        # Raw URL to access the CSV file
        url = 'https://raw.githubusercontent.com/Sefaria/Sefaria-Project/master/data/tmp/parsha.csv'
        # Load the CSV file into a DataFrame
        parashot = pd.read_csv(url)
        parashot.index.name="n"

        # Save the DataFrame to CSV
        parashot.to_csv(output[0], index=True)


def parashot_list(wildcards):
    import pandas as pd
    parashotFile = checkpoints.get_parashot_csv.get().output[0] 
    parashot = pd.read_csv(parashotFile)
    parashot = parashot[parashot['ref'].str.contains(wildcards.book)]
    return parashot['en'].tolist()

def parasha_verse(wildcards):
    import pandas as pd
    parashotFile = checkpoints.get_parashot_csv.get().output[0] 
    parashot = pd.read_csv(parashotFile)
    parashot.index=parashot['en']
    return (parashot.loc[wildcards.parasha]['ref'])


def parasha_lines(wildcards):
    import json
    from pprint import pprint
    import pandas as pd
    parashotFile = checkpoints.get_parasha.get(lang="english", parasha=wildcards["parasha"]).output[0] 
    parashot = json.loads(open(parashotFile).read())['versions'][0]['text']
    if type(parashot[0])==str:
        parashot = [parashot]
    #parashot.index=parashot['en']
    table = pd.read_csv(checkpoints.get_parashot_csv.get().output[0])
    table.index=table['en']
    ref = table.loc[wildcards["parasha"]]['ref'].split()[-1].split('-')[0]
    book = table.loc[wildcards["parasha"]]['ref'].split()[0]
    verse,line = ref.split(':')
    verse,line = int(verse),int(line)
    out = []
    for Verse in parashot:
        for _ in Verse:
            out.append((book,verse, line))
            line +=1
        verse += 1
        line = 1
    return (out)
        
def remove_footnotes_en(s):
    # Parse the string with BeautifulSoup
    if type(s) == list: s = " ".join(s)
    if s==[]:s=""
    s = s.replace("<br>", " ")
    soup = BeautifulSoup(s, "html.parser")
    
    # Iterate through all <i> tags with class "footnote"
    for footnote_tag in soup.find_all("i", class_="footnote"):
        # Get the text inside the footnote tag, strip any extra whitespace
        footnote_text = footnote_tag.get_text(strip=False)
        
        # Replace the footnote tag with its text wrapped in brackets
        footnote_tag.replace_with(f" ({footnote_text}) ")
    for footnote_marker in soup.find_all("sup",class_="footnote-marker"):
        footnote_marker.decompose()

    # Get the cleaned-up text without any HTML tags
    return soup.get_text()

def remove_footnotes_heb(s):
    s = s.replace("<br>", " ")

    # Parse the string with BeautifulSoup
    soup = BeautifulSoup(s, "html.parser")
    # Find and remove all <i> tags with class "footnote"
    for footnote_tag in soup.find_all("i", class_="footnote"):
        footnote_tag.replace_with(" ")
    for footnote_marker in soup.find_all("sup",class_="footnote-marker"):
        footnote_marker.decompose()

    # Get the cleaned up HTML
    return str(soup)
def trim_to_max_tokens(text, max_tokens=10000, model="gpt-3.5-turbo"):
    # Load the tokenizer for the given model
    encoding = tiktoken.encoding_for_model(model)
    
    # Tokenize the text
    tokens = encoding.encode(text)
    
    # Trim the tokens if they exceed the max_tokens limit
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    
    # Decode the tokens back to text
    trimmed_text = encoding.decode(tokens)
    return trimmed_text
@norerun
def summarize_text(text,refereces=False):
    # Get the API key from the environment variable
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    if refereces:
        text = trim_to_max_tokens( f"Please summarize the following text in no more than one paragraph, keep references in brackets indicating from which commentary it came (is at the beginning of each paragraph), Do it succintly, do not include introductions, just bulletpoint statements of specifics:\n\n{text}")
    else:
        text = trim_to_max_tokens( f"Please summarize the following text in no more than one paragraph, Do it succintly, do not include introductions, just bulletpoint statements of specifics, not as a debate between sources, but mention specifically what they say:\n\n{text}")

    chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content":text,
                }
            ],
            model="gpt-3.5-turbo",
        )
    
    return chat_completion.choices[0].message.content





checkpoint get_parasha:
    input:
        parashotFile="parashot.csv"
    output:
        "sefaria/{lang}_{parasha}.json"
    run:
        import urllib.parse
        import requests
        import pandas as pd
        parashot = pd.read_csv(input.parashotFile)
        parashot.index=parashot['en']
        verses = (parashot.loc[wildcards.parasha]['ref'])

        # Original verse reference
        encoded_reference = urllib.parse.quote(verses)
        url = f"https://www.sefaria.org/api/v3/texts/{encoded_reference}?version={wildcards.lang}"
        headers = {"accept": "application/json"}

        response = requests.get(url, headers=headers)
        with open(output[0], "w") as f:
            f.write(response.text)
rule get_commentary:
    output:
        #"sefaria/commentary_{parasha}.json",
        "sefaria/commentary/{book}/{verse}/{line}.json"
    run:
        import urllib.parse
        import requests
        import pandas as pd
        encoded_reference = urllib.parse.quote(f"{wildcards.book} {wildcards.verse}:{wildcards.line}")
        url = f"https://www.sefaria.org/api/links/{encoded_reference}"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)
        if response.text.startswith('{"error":'):
            raise Exception(response.text)
        with open(output[0], "w") as f:
            f.write(response.text)

ruleorder: get_commentary>get_parasha
rule parse_commentary:
    input:
        #commentary="sefaria/commentary_{parasha}.json",
        parts = lambda wildcards: [f"sefaria/commentary/{book}/{verse:02}/{line:03}.json" for book, verse, line in parasha_lines(wildcards)],
    output:
        "sefaria/commentary_{parasha}.csv",
    run:
        import json
        from pprint import pprint
        import pandas as pd
        out = []
        for fname in input.parts:
            with open(fname) as f:
                commentsaries = json.load(f)
                for commentary in commentsaries:
                    ref = commentary["anchorRefExpanded"][-1]
                    #if "Rav Hirsch on Torah" in commentary["ref"]: continue

                    out.append({
                        "verse" : int(ref.split()[-1].split(":")[0]),
                        "line" : int(ref.split()[-1].split(":")[1]),
                        "category" : commentary["category"],
                        "source" : commentary["ref"],
                        "text" : BeautifulSoup(remove_footnotes_en(commentary["text"]), "lxml").text
                    })

        df = pd.DataFrame(out)
        df = df[df.category != "Reference"]
        df = df[df.text != ""]
        df.to_csv(output[0], index=False)

        for category in df.category.unique():
            print(category, len(df[df.category == category]))

rule summarise_commentary:
    input:
        "sefaria/commentary_{parasha}.csv",
    output:
        "sefaria/sumarized_commentary_{parasha}.csv"
    run:
        import pandas as pd
        from tqdm import tqdm
        df = pd.read_csv(input[0])

        @norerun
        def openai_summarize(text,parasha,verse,line,source,category):
            # Get the API key from the environment variable
            client = OpenAI(
                # This is the default and can be omitted
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
            text = trim_to_max_tokens( f"Please summarize the following text from {category} {source}, talking about {parasha} {verse}:{line} in no more than one setnce, Do it succintly:\n\n{text}")
            
            chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content":text,
                        }
                    ],
                    model="gpt-3.5-turbo",
                )
            
            return chat_completion.choices[0].message.content
        sumaries = []
        for i,row in tqdm(df.iterrows(), total=len(df)):
            summary = openai_summarize(row.text, wildcards.parasha, row.verse, row.line, row.source,row.category)

            sumaries.append(summary)

        df["summary"] = sumaries

        df.to_csv(output[0], index=False)




rule prepare_text_block_for_summary:
    input:
        commentary="sefaria/commentary_{parasha}.csv"
    output:
        "sefaria/summaries/{parasha}/text_block_{category}.pkl"
    run:
        import pandas as pd
        import pickle
        with open(input.commentary) as f:
            comments = pd.read_csv(f)
        comments = comments[comments.category == wildcards.category]
        out={}
        for _, row in comments.iterrows():
            if not (int(row.verse), int(row.line)) in out:
                out[(int(row.verse), int(row.line))] = ""

            out[(int(row.verse), int(row.line))] += f"{row.source}\n{row.text}\n\n"

        with open(output[0], "wb") as f:
            pickle.dump(out, f)
rule get_summaries:
    input:
        "sefaria/summaries/{parasha}/text_block_{category}.pkl"
    output:
        "sefaria/summaries/{parasha}/summary_{category}.pkl"
    params:
        max_tokens_input = 15000,
        max_tokens_output = 750,
        temperature = 0.5,
        preprompt = "Please summarize the following text in no more than one paragraph,"
                    " keep references in brackets indicating from which commentary it came "
                    "(is at the beginning of each paragraph), Do it succintly, do not "
                    "include introductions, just statements of specifics.",
        model=MODEL,
    run:
        import pickle
        from openai import OpenAI
        from tqdm import tqdm
        from time import sleep

        with open(input[0], "rb") as f:
            summaries = pickle.load(f)
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        out = {}
        for verse in tqdm(summaries.keys(), total=len(summaries),desc=f"Summarizing {wildcards.category} for {wildcards.parasha}"):
            text = ( f"{params.preprompt} The text is from the {wildcards.category}.\n\n{summaries[verse]}")
            text = trim_to_max_tokens(text, max_tokens=params.max_tokens_input, model=params.model)

            chat_completion = client.chat.completions.create(
                messages=[
                            {
                                "role": "user",
                                "content":text,
                            }
                        ],
                model=params.model,
                max_tokens=params.max_tokens_output,
                temperature=params.temperature
                )
            
    
            summary= chat_completion.choices[0].message.content
            out[verse] = summary

            sleep(0.5)
        
        with open(output[0], "wb") as f:
            pickle.dump(out, f)

rule make_metasummary:
    input:
        expand("sefaria/summaries/{{parasha}}/summary_{category}.pkl", category=commentary_categories)
    output:
        "sefaria/summaries/{parasha}/meta_summary.pkl"
    params:
        max_tokens_input = 15000,
        max_tokens_output = 750,
        temperature = 0.5,
        preprompt = "Please summarize the following text in no more than one paragraph,"
                    " keep references in brackets indicating from which commentary it came "
                    "(is at the beginning of each paragraph), Do it succintly, do not "
                    "include introductions, just statements of specifics.",
        model=MODEL,
    run:
        import pickle
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        out = {}
        summaries = (pd.concat([pd.Series(pd.read_pickle(i),name=i.split("summary_")[-1].split(".")[0]) for i in input],axis=1))
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        for verse, row in tqdm(summaries.iterrows(), total=len(summaries),desc=f"Summarizing all commentary for {wildcards.parasha}"):
            block = ""
            for category,text in row.dropna().items():
                block += f"{category}: {text}\n\n"
            text = ( f"{params.preprompt}\n\n{block}")
            text = trim_to_max_tokens(text, max_tokens=params.max_tokens_input, model=params.model)

            chat_completion = client.chat.completions.create(
                messages=[
                            {
                                "role": "user",
                                "content":text,
                            }
                        ],
                model=params.model,
                max_tokens=params.max_tokens_output,
                temperature=params.temperature
                )
            summary= chat_completion.choices[0].message.content
            out[verse] =   summary
        
        with open(output[0], "wb") as f:
            pickle.dump(out, f)
        
rule make_Elliott_dictionary:
    input:
        "ElliottFriedman.txt",
    output:
        "ElliottFriedman.pk"
    run:
        import pickle

        REF = {}
        def keepOnlyNumbers(s):
            return int("".join([i for i in s if i.isdigit() ]))
            


        with open(input[0]) as f:
            book=""
            for line in f:
                if line == "\n":
                    continue
                if line.startswith(">"):
                    book = line[1:].strip()
                    continue
                autor,parts = line.split("~")
                for part in parts.split(";"): 
                    if part.strip() == "": continue
                    try:
                        verse,a = part.split(":")
                    except:
                        print(book,repr(a))
                        raise ValueError(part)
                    verse = keepOnlyNumbers(verse)
                    for b in a.split(","):
                        if "-" in b:
                            try:
                                start,end = b.split("-")
                            except:
                                print(book,repr(b))
                                raise ValueError(b)
                            start = keepOnlyNumbers(start)
                            end = keepOnlyNumbers(end)
                            for c in range(start,end+1):
                                if not (book,verse,c) in REF:
                                    REF[(book,verse,c)]=[]
                                REF[(book,verse,c)].append(autor)
                        else:
                            b = keepOnlyNumbers(b)
                            if not (book,verse,b) in REF:
                                REF[(book,verse,b)]=[]
                            REF[(book,verse,b)].append(autor)
        with open(output[0], "wb") as f:
            pickle.dump(REF, f)


rule fix_docx:
    input:
        docx=".BOOK_{book}.docx"
    output:
        docx="BOOK_{book}.docx"
    run:

        from docx import Document
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Inches

        # Open the document
        doc = Document(input.docx)
       
        # Set the page width and height
        doc.sections[0].page_width      = Inches(8.5)
        doc.sections[0].page_height     = Inches(11)
        doc.sections[0].top_margin      = Inches(0.25)
        doc.sections[0].bottom_margin   = Inches(0.25)


        # Access the "Heading 1" style
        for style in doc.styles:
            if style.name == "Heading 1":

                # Modify font settings
                font = style.font
                font.bold = True
                font.size = Pt(16)  # Optional: Set the font size
                font.color.rgb = RGBColor(0, 0, 0)  # Set the font color to black

                # Modify paragraph settings
                paragraph_format = style.paragraph_format
                paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER  # Center the text
                paragraph_format.space_after = Pt(2)  # Optional: Adjust space after paragraph

                # Add a page break before every Heading 1 paragraph
                style.paragraph_format.page_break_before = True
            elif style.name == "Heading 2":

                # Modify font settings
                font = style.font
                font.bold = True
                font.size = Pt(14)  # Optional: Set the font size
                font.color.rgb = RGBColor(100, 100, 100)  # Set the font color to black

                # Modify paragraph settings
                paragraph_format = style.paragraph_format
                paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER  # Center the text
                paragraph_format.space_after = Pt(12)  # Optional: Adjust space after paragraph

                # Add a page break before every Heading 1 paragraph
                style.paragraph_format.page_break_before = False   
            elif style.name == "Compact":
                font = style.font
                font.name = "SBL Hebrew"
                paragraph_format = style.paragraph_format
                paragraph_format.keep_together = True  # Ensure paragraph stays on the same page


        # Function to set vertical alignment for a cell
        def set_vertical_alignment(cell, alignment="top"):
            tc = cell._tc  # Access the underlying XML for the table cell
            tcPr = tc.get_or_add_tcPr()  # Get or create the table cell properties (tcPr)
            
            # Create or find the vertical alignment element
            vAlign = tcPr.find(qn('w:vAlign'))
            if vAlign is None:
                vAlign = OxmlElement('w:vAlign')
                tcPr.append(vAlign)
            
            # Set the alignment value ("top", "center", "bottom")
            vAlign.set(qn('w:val'), alignment)

        # Function to set table borders to white (or transparent)
        def set_table_borders_white(table):
            tbl = table._tbl  # Access the underlying XML for the table
            
            # Get or create the table properties (tblPr)
            tblPr = tbl.tblPr
            
            # Create or access the borders element (tblBorders)
            tblBorders = tblPr.find(qn('w:tblBorders'))
            if tblBorders is None:
                tblBorders = OxmlElement('w:tblBorders')
                tblPr.append(tblBorders)
            
            # Modify or create the borders: top, left, bottom, right, insideH (horizontal), insideV (vertical)
            for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
                border = tblBorders.find(qn(f'w:{border_name}'))
                if border is None:
                    border = OxmlElement(f'w:{border_name}')
                    tblBorders.append(border)
                
                # Set border color to white and border size to 0 to make it effectively transparent
                border.set(qn('w:val'), 'single')
                border.set(qn('w:sz'), '4')  # Border size (use 0 for completely invisible)
                border.set(qn('w:space'), '0')
                border.set(qn('w:color'), 'FFFFFF')  # Set to white

        # Iterate through all the tables in the document
        for table in doc.tables:
            # Set the border color to white/transparent
            set_table_borders_white(table)
            
            # Iterate through all rows and cells
            for row in table.rows:
                # Access the row properties and disable splitting across pages
                row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
                for cell in row.cells:
                    # Set each cell to be aligned to the top
                    set_vertical_alignment(cell, alignment="top")

        # Save the modified document
        doc.save(output.docx)
        print("Done!")
        print("")
        print("remember to do these things in word, it is too complacated to do in snakemake:")
        print("    - Pages numbers (centered).")
        print("    - Table of contents, only one level.") 
        print("    - Save as PDF.")
        print("")
        print("Ready for the presses!")
        print("")
        print("That all folks!")

rule make_mega_json:
    output:
        json = "src/mega_{parasha}.json",

    input:
        meta_summary = "sefaria/summaries/{parasha}/meta_summary.pkl",
        summaries    = expand("sefaria/summaries/{{parasha}}/summary_{source}.pkl", source=commentary_categories),
        commentary   = "sefaria/commentary_{parasha}.csv",
        english      ="sefaria/english_{parasha}.json",
        hebrew       = "sefaria/hebrew_{parasha}.json",
    run:
        import pandas as pd
        from sortedcontainers import SortedDict
        import pickle
        import json
        from tqdm import tqdm
        import itertools



        # Create the main HTML document
        meta = parasha_lines(wildcards)
        commentary = pd.read_csv(input.commentary)
        with doc.body:

            out = SortedDict()
            with open(input.meta_summary, "rb") as f:
                meta_summary = (pickle.load(f))
            with open(input.english) as f: english = list(itertools.chain.from_iterable(json.load(f)['versions'][0]['text']))
            with open(input.hebrew) as f:  hebrew  = list(itertools.chain.from_iterable(json.load(f)['versions'][0]['text']))
            summaries = {}
            for source,_summary in zip(commentary_categories,input.summaries):
                with open(_summary, "rb") as f:
                    summaries[source] = (pickle.load(f))
            

            for (book, verse, line),hebrew_line, english_line in tqdm(zip(meta,hebrew, english), total=len(meta)):

                sub1 = SortedDict()
                for source,s in summaries.items():
                    c = (commentary[(commentary.category == source) & (commentary.verse == verse) & (commentary.line == line)])

                    c = (SortedDict(zip(c["source"], c["text"])))
                    if (verse, line) in s:  
                        sub1[source] = {
                            "summary":s[(verse, line)],
                            "commentaries": c
                            }
                if not verse in out:
                    out[verse] = SortedDict()

                out[verse][line] = {
                    "hebrew": hebrew_line, 
                    "english": english_line, 
                    "summary": meta_summary[(verse, line)],
                    "commentaries": sub1}

        with open(output.json, "w") as f: 
            json.dump(dict(out), f, indent=2)   


  

rule all:
    input:
        expand("BOOK_{book}.docx",
            book=[
                "Genesis",
                "Exodus",
                "Leviticus",
                "Numbers",
                "Deuteronomy"
            ])

